import threading
from functools import lru_cache, partial
from pathlib import Path
from typing import Union, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from ConfigSpace import Configuration
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModel

from automl.core.approaches.base_approach import Approach
from automl.core.registry import register_approach
from automl.core.trainers.torch_trainer import TorchTrainer
from automl.core.types import DatasetSplit, TrainResult, PredictionResult
from automl.logger import get_logger

logger = get_logger()

_tokenizer_cache = threading.local()


def _load_tokenizer(path: str) -> PreTrainedTokenizerBase:
    """Load (and cache) one tokenizer instance per THREAD.

    `prepare()` is called once per hyperparameter-optimization trial (the
    ifBO loop in optimizer.py can run hundreds of these), so reusing a
    tokenizer instead of re-reading it from disk every call matters. This
    used to be a single `lru_cache`-wrapped instance shared by the whole
    process, which is safe as long as calls are sequential - but a fast
    (Rust-backed) tokenizer's `__call__` mutates its own truncation/padding
    config in place first (`max_length` varies per trial's sampled
    `max_seq_length`), so two trials tokenizing concurrently on separate
    threads (see IfboOptimizer's parallel-trial loop) race on that shared
    mutable state and crash with `RuntimeError: Already borrowed`. Caching
    one instance per thread instead keeps the "load once, reuse many
    times" benefit within a thread while giving each concurrently-running
    trial its own private tokenizer to mutate.
    """
    cached = getattr(_tokenizer_cache, "tokenizer", None)
    if cached is None or getattr(_tokenizer_cache, "path", None) != path:
        logger.info(
            f"Loading tokenizer from '{path}' for thread "
            f"{threading.get_ident()} (not yet cached on this thread)."
        )
        cached = AutoTokenizer.from_pretrained(path)
        _tokenizer_cache.tokenizer = cached
        _tokenizer_cache.path = path
    else:
        logger.debug(
            f"Reusing thread-local tokenizer for '{path}' "
            f"(thread {threading.get_ident()})."
        )
    return cached


# Full (untruncated) per-text encodings, cached by tokenizer path so they
# survive across HPO trials. Keyed by tokenizer path -> {text: input_ids}.
_full_encoding_cache: dict[str, dict[str, list[int]]] = {}
_full_encoding_cache_lock = threading.Lock()


def _encode_texts_cached(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_path: str,
) -> list[list[int]]:
    """Tokenize `texts` once, fully untruncated, caching each result by
    (tokenizer_path, text) so a later call for the *same* text - just under
    a different trial's `max_seq_length` - is answered from this in-memory
    dict instead of re-running the tokenizer.

    `prepare()` is called once per HPO trial (hundreds, per ifBO), and each
    call previously re-tokenized the *entire* train+val corpus from scratch.
    Every trial resamples train/val from the same fixed underlying pool of
    texts (see `IfboOptimizer.train_single_configuration`, which reseeds
    `train_test_split` per trial but draws from the same dataset), so after
    the first trial nearly every text is already cached here and the cost
    collapses to dict lookups. Truncation to the trial's `max_seq_length` is
    applied afterwards, in `TextSequenceDataset`, by slicing these cached
    full-length ids - so this cache is sized once by corpus size, not by
    the number of trials or `max_seq_length` values seen.
    """
    cache = _full_encoding_cache.setdefault(tokenizer_path, {})

    missing_texts = []
    missing_positions = []
    result: list[Optional[list[int]]] = [None] * len(texts)
    for i, text in enumerate(texts):
        cached_ids = cache.get(text)
        if cached_ids is not None:
            result[i] = cached_ids
        else:
            missing_texts.append(text)
            missing_positions.append(i)

    if missing_texts:
        logger.debug(
            f"Tokenization cache for '{tokenizer_path}': "
            f"{len(texts) - len(missing_texts)}/{len(texts)} texts already "
            f"cached, encoding {len(missing_texts)} new text(s)."
        )
        encoded = tokenizer(
            missing_texts,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        with _full_encoding_cache_lock:
            for pos, text, ids in zip(missing_positions, missing_texts, encoded):
                ids = cache.setdefault(text, ids)
                result[pos] = ids
    else:
        logger.debug(
            f"Tokenization cache for '{tokenizer_path}': all {len(texts)} "
            f"text(s) served from cache."
        )

    return result  # type: ignore[return-value]


def _truncate_ids(
    ids: list[int], max_seq_len: int, sep_token_id: Optional[int]
) -> list[int]:
    """Reproduce `tokenizer(text, truncation=True, max_length=max_seq_len)`
    from fully-encoded `ids` (`[CLS] + content + [SEP]`): if truncation is
    needed, keep the first `max_seq_len - 1` tokens and re-append
    `sep_token_id`, matching the tokenizer's own right-truncation-before-SEP
    behavior, instead of just cutting off SEP with a plain slice.
    """
    if len(ids) <= max_seq_len:
        return ids
    if sep_token_id is None:
        return ids[:max_seq_len]
    return ids[: max_seq_len - 1] + [sep_token_id]


@lru_cache(maxsize=1)
def _load_pretrained_word_embeddings(model_name: str) -> torch.Tensor:
    """Load (and cache) just the pretrained token embedding matrix for
    `model_name`, used to warm-start the BiLSTM's embedding layer instead
    of starting from scratch.
    """
    logger.info(f"Loading pretrained embedding matrix from '{model_name}'.")
    model = AutoModel.from_pretrained(model_name)
    weight = model.get_input_embeddings().weight.detach().clone()
    del model
    logger.debug(f"Pretrained embedding matrix for '{model_name}': shape={tuple(weight.shape)}.")
    return weight


@lru_cache(maxsize=4)
def _pretrained_svd(model_name: str, vocab_size: int):
    logger.info(
        f"Computing SVD of pretrained embeddings for '{model_name}' "
        f"(vocab_size={vocab_size}); this runs once per (model_name, vocab_size)."
    )
    matrix = _load_pretrained_word_embeddings(model_name)
    if matrix.size(0) >= vocab_size:
        matrix = matrix[:vocab_size]
    else:
        pad = torch.empty(vocab_size - matrix.size(0), matrix.size(1))
        nn.init.normal_(pad, mean=0.0, std=0.02)
        matrix = torch.cat([matrix, pad], dim=0)
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    logger.debug(f"SVD complete for '{model_name}' (vocab_size={vocab_size}).")
    return centered, vt  # computed once, ever, per (model_name, vocab_size)


@lru_cache(maxsize=16)
def _pretrained_embedding_init(
    model_name: str, vocab_size: int, target_dim: int
) -> torch.Tensor:
    """Build an (vocab_size, target_dim) init matrix from `model_name`'s
    pretrained embeddings.

    `target_dim` is the tuned `seq_embed_dim` hyperparameter (32-512) and
    essentially never matches the transformer's native hidden size (768
    for distilbert), so a straight copy isn't possible. Instead, we PCA the
    pretrained matrix down to `target_dim`: this keeps the directions of
    highest variance in the pretrained embedding space, so tokens that are
    semantically close before the projection stay close after it too -
    still a much better starting point than random init, which is what
    makes the LSTM re-learn token semantics from scratch every trial.
    """
    logger.debug(
        f"Building pretrained embedding init for '{model_name}' "
        f"(vocab_size={vocab_size}, target_dim={target_dim})."
    )
    centered, vt = _pretrained_svd(model_name, vocab_size)
    if target_dim >= centered.size(1):
        logger.debug(
            f"target_dim={target_dim} >= native dim={centered.size(1)}; "
            f"padding with {target_dim - centered.size(1)} randomly-initialized dim(s)."
        )
        extra = torch.empty(centered.size(0), target_dim - centered.size(1))
        nn.init.normal_(extra, mean=0.0, std=0.02)
        return torch.cat(
            [centered, extra], dim=1
        )  # note: not mean-restored, matches original behavior
    return centered @ vt[:target_dim].T


def _collate_sequences(batch, pad_value: int = 0):
    """Pad a batch to the length of its longest sequence.

    TextSequenceDataset now stores un-padded (truncated) token ids, so
    padding happens here, per-batch, instead of once for the whole dataset
    at a fixed `max_seq_length`. If most texts are much shorter than
    `max_seq_length`, this avoids materializing (and later training over)
    a large amount of pure padding.
    """
    if isinstance(batch[0], tuple):
        sequences, labels = zip(*batch)
        labels = torch.stack(labels)
    else:
        sequences, labels = batch, None

    lengths = [seq.size(0) for seq in sequences]
    max_len = max(max(lengths), 1)  # guard against an all-empty batch
    padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : seq.size(0)] = seq

    return (padded, labels) if labels is not None else padded


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = True,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_embeddings)
                self.embedding.weight[0].zero_()  # keep padding_idx row zero
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * directions, num_classes)

    def forward(self, input_ids):
        # Derive real sequence lengths from padding and pack the batch
        # before running the LSTM. Previously the LSTM ran (and stored
        # activations for backprop) over every padded position too - with
        # a large max_seq_length and mostly-short texts, that's a lot of
        # wasted compute and, more importantly, wasted autograd memory.
        # Packing skips padded positions entirely in both directions.
        lengths = (input_ids != self.embedding.padding_idx).sum(dim=1).clamp(min=1)

        emb = self.embedding(input_ids)  # (B, L, E)
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, c_n) = self.lstm(packed)  # h_n: (num_layers*D, B, H)

        # Use last layer’s hidden state, concatenate both directions
        if self.lstm.bidirectional:
            last_fwd = h_n[-2, :, :]  # (B, H)
            last_bwd = h_n[-1, :, :]  # (B, H)
            h = torch.cat([last_fwd, last_bwd], dim=1)
        else:
            h = h_n[-1, :, :]
        h = self.dropout(h)
        logits = self.fc(h)  # (B, num_classes)
        return logits


class TextSequenceDataset(torch.utils.data.Dataset):

    DEFAULT_LABEL_MASK = -100

    def __init__(
        self,
        full_input_ids: list[list[int]],
        labels,
        max_seq_len: int,
        sep_token_id: Optional[int] = None,
    ):
        # `full_input_ids` is already tokenized (see `_encode_texts_cached`),
        # fully untruncated and WITHOUT attention_mask / token_type_ids -
        # only input_ids is ever used downstream, so keeping the other two
        # fields around wastes roughly 2/3 of the memory this dataset used
        # to hold. Truncation to this trial's `max_seq_len` happens here,
        # per-sample, so the same cached full encoding can be reused by
        # every trial regardless of its sampled `max_seq_length`.
        input_ids_list = [
            _truncate_ids(ids, max_seq_len, sep_token_id) for ids in full_input_ids
        ]

        # Store every sequence back-to-back in ONE contiguous int32 buffer
        # (+ offsets) rather than as a Python list of per-sample tensors.
        # Two separate wins:
        #  1. int32 instead of int64 halves the raw storage size (vocab
        #     sizes like distilbert's ~30k fit comfortably in int32; we
        #     upcast to int64 lazily, per-sample, only when a batch is
        #     actually read).
        #  2. A single tensor (vs. a Python list/list-of-tensors) avoids
        #     the classic PyTorch DataLoader multiprocessing pitfall where
        #     touching many individual Python objects' refcounts in worker
        #     processes forces the OS to copy-on-write pages that were
        #     otherwise shared with the parent process, silently
        #     multiplying memory usage by ~num_workers.
        lengths = torch.tensor([len(ids) for ids in input_ids_list], dtype=torch.long)
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
        self.input_ids = (
            torch.cat([torch.tensor(ids, dtype=torch.int32) for ids in input_ids_list])
            if input_ids_list
            else torch.empty(0, dtype=torch.int32)
        )
        self._len = len(input_ids_list)

        if labels is not None:
            # Precompute the label tensor once (mapping NaN -> mask value)
            # instead of storing a raw Python list and re-checking
            # `pd.isna` on every __getitem__ call. This also removes the
            # same copy-on-write risk described above for the label list.
            label_series = pd.Series(labels)
            filled = label_series.fillna(self.DEFAULT_LABEL_MASK).astype("int64")
            self.labels = torch.from_numpy(filled.to_numpy().copy())
        else:
            self.labels = None

        logger.debug(
            f"Built TextSequenceDataset: {self._len} sample(s), "
            f"max_seq_len={max_seq_len}, total tokens={self.input_ids.numel()}."
        )

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        x = self.input_ids[start:end].to(torch.long)
        if self.labels is None:
            return x
        return x, self.labels[idx]


@register_approach("sequence-dl")
class SequenceDLApproach(Approach[torch.nn.Module, dict]):

    TOKENIZER_PATH = "./tokenizers/distilbert-base-uncased"
    EMBEDDING_MODEL_NAME = "distilbert-base-uncased"

    def __init__(
        self,
        config: Union[Configuration, dict],
        num_classes: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(config, num_classes, device, **kwargs)
        num_workers: Optional[int] = kwargs.get("num_workers", None)
        self._num_worker: int = (
            num_workers
            if num_workers is not None
            else self.get_param_value("num_workers")  # FIXME: Not a configuration param
        )
        self._stochastic_epochs: bool = bool(kwargs.get("stochastic_epochs", False))
        self._stochastic_epoch_fraction: Optional[float] = kwargs.get(
            "stochastic_epoch_fraction", None
        )
        self.vocab = None
        self.model = None
        # self.label_encoder = LabelEncoder()
        self.trainer: Optional[TorchTrainer] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._pad_id: int = 0
        self._sep_token_id: Optional[int] = None

    def initialize(self):
        pass

    def prepare_training(self, train: DatasetSplit):
        return train  # handled in prepare()

    def prepare_validation(self, val: DatasetSplit):
        return val

    def prepare(self, train: DatasetSplit, val: DatasetSplit):
        # Hyperparams from config / defaults
        max_seq_len = int(self.get_param_value("max_seq_length"))
        embed_dim = int(self.get_param_value("seq_embed_dim"))
        hidden_dim = int(self.get_param_value("hidden_dim"))
        num_layers = int(self.get_param_value("seq_num_layers"))
        dropout = float(self.get_param_value("dropout"))
        batch_size = int(self.get_param_value("batch_size"))

        logger.debug(
            f"[{self.name}] prepare(): max_seq_len={max_seq_len}, "
            f"embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, dropout={dropout}, "
            f"batch_size={batch_size}, num_workers={self._num_worker}."
        )

        # Build vocab on train text
        train_texts = train.texts  # adjust column name as needed

        # Build datasets
        train_labels = train.labels
        val_texts = val.texts
        val_labels = val.labels

        logger.debug(
            f"[{self.name}] prepare(): {len(train_texts)} train text(s), "
            f"{len(val_texts)} val text(s)."
        )

        # train_labels = self.label_encoder.fit_transform(train_labels)
        # val_labels = self.label_encoder.transform(val_labels)

        # TODO: Add to configspace
        self.tokenizer = _load_tokenizer(self.TOKENIZER_PATH)
        assert self.tokenizer is not None, "Tokenizer is None"
        self._pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        self._sep_token_id = self.tokenizer.sep_token_id

        # Encode once per unique text (cached across trials, see
        # `_encode_texts_cached`); only truncation to this trial's
        # `max_seq_len` happens below, in `TextSequenceDataset`.
        train_full_ids = _encode_texts_cached(
            train_texts, self.tokenizer, self.TOKENIZER_PATH
        )
        val_full_ids = _encode_texts_cached(
            val_texts, self.tokenizer, self.TOKENIZER_PATH
        )

        train_ds = TextSequenceDataset(
            train_full_ids, train_labels, max_seq_len, self._sep_token_id
        )
        val_ds = TextSequenceDataset(
            val_full_ids, val_labels, max_seq_len, self._sep_token_id
        )

        collate_fn = partial(_collate_sequences, pad_value=self._pad_id)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._num_worker,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._num_worker > 0,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._num_worker,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._num_worker > 0,
            collate_fn=collate_fn,
        )

        # Build model
        vocab_size = self.tokenizer.vocab_size
        pretrained_embeddings = _pretrained_embedding_init(
            self.EMBEDDING_MODEL_NAME, vocab_size, embed_dim
        )
        self.model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=self._num_classes,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            pretrained_embeddings=pretrained_embeddings,
        )
        assert self.model is not None
        self.model.to(self._device)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.debug(
            f"[{self.name}] Built BiLSTMClassifier: vocab_size={vocab_size}, "
            f"num_classes={self._num_classes}, num_params={num_params}, "
            f"device={self._device.type}."
        )

        # Trainer
        # epochs = self.get_param_value("epochs")

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "vocab": self.vocab,
        }

    def train(
        self,
        prepared_result,
        epochs: int = 10,
        load_path: Optional[Path] = None,
        evaluate_validation=True,
        **kwargs,
    ) -> TrainResult:
        assert self.model is not None

        optimizer_name = self.get_param_value("optimizer")
        lr = self.get_param_value("learning_rate")
        weight_decay = self.get_param_value("weight_decay")

        scheduler = self.get_param_value("scheduler")
        warmup_ratio = float(self.get_param_value("warmup_ratio"))
        max_grad_norm = float(self.get_param_value("max_grad_norm"))

        optimizer_args = {"lr": lr, "weight_decay": weight_decay}

        # remove value which are None
        # scheduler_args = {k: v for k, v in scheduler_args.items() if v is not None}

        logger.debug(
            f"[{self.name}] train(): optimizer={optimizer_name}, lr={lr}, "
            f"weight_decay={weight_decay}, scheduler={scheduler}, "
            f"warmup_ratio={warmup_ratio}, max_grad_norm={max_grad_norm}, "
            f"epochs={epochs}, evaluate_validation={evaluate_validation}, "
            f"load_path={load_path}."
        )

        if self.trainer is None:
            logger.debug(f"[{self.name}] No existing trainer; creating a new TorchTrainer.")
            trainer = TorchTrainer(
                model=self.model,
                approach_name=self.name,
                train_loader=prepared_result["train_loader"],
                val_loader=prepared_result["val_loader"],
                device=self._device,
                optimizer=optimizer_name,
                optimizer_args=optimizer_args,
                scheduler=scheduler,
                epochs=epochs,
                evaluate_validation=evaluate_validation,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                stochastic_epochs=self._stochastic_epochs,
                stochastic_epoch_fraction=self._stochastic_epoch_fraction,
            )
            self.trainer = trainer
        else:
            logger.debug(f"[{self.name}] Reusing existing trainer for continued training.")

        assert self.trainer is not None
        result = self.trainer.train(
            load_path=load_path,
            save_path=None,  # or some path if you want val-best checkpoint
        )
        logger.info(f"[{self.name}] train() finished after {epochs} epoch(s).")
        return result

    @torch.no_grad()
    def predict(self, data: pd.DataFrame | DataLoader) -> PredictionResult:
        assert self.model is not None, "Model is not initialized"
        assert self.tokenizer is not None, "Tokenizer is not initialized"

        max_seq_length = self.get_param_value("max_seq_length")
        batch_size = self.get_param_value("batch_size")
        self.model.eval()

        if isinstance(data, pd.DataFrame):
            logger.debug(
                f"[{self.name}] predict(): building DataLoader from a "
                f"{len(data)}-row DataFrame (max_seq_length={max_seq_length}, "
                f"batch_size={batch_size})."
            )
            texts = data["text"].tolist()
            labels = data["label"].tolist()

            full_ids = _encode_texts_cached(
                texts, self.tokenizer, self.TOKENIZER_PATH
            )
            ds = TextSequenceDataset(
                full_ids,
                labels=labels,
                max_seq_len=max_seq_length,
                sep_token_id=self._sep_token_id,
            )

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=self._device.type == "cuda",
                collate_fn=partial(_collate_sequences, pad_value=self._pad_id),
            )
        else:
            logger.debug(f"[{self.name}] predict(): using caller-provided DataLoader.")
            loader = data

        all_preds = []
        all_labels = []

        for batch in loader:
            x, y = batch
            x = x.to(self._device, non_blocking=True)
            # `y` is only ever compared/concatenated on CPU later - it never
            # needs to touch the GPU at all, so we no longer copy it there.

            logits = self.model(x)
            preds = torch.argmax(logits, dim=-1)

            # Move each batch's predictions to CPU immediately instead of
            # letting a list of GPU-resident tensors grow for the entire
            # pass. For large prediction sets this bounds peak GPU memory
            # to ~one batch instead of the whole dataset.
            all_preds.append(preds.cpu())
            all_labels.append(y)

        # One GPU -> CPU transfer per batch, already done above.
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        # y_pred_orig = self.label_encoder.inverse_transform(y_pred)

        logger.debug(f"[{self.name}] predict(): produced {len(y_pred)} prediction(s).")

        return {
            "y_pred": y_pred,
            "y_true": y_true,
        }
