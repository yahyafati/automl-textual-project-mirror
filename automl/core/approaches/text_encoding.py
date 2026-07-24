"""
Shared tokenization/dataset helpers for text approaches that consume raw
token-id sequences (`sequence_dl.py`'s BiLSTM and `transformer.py`'s
fine-tuned transformer both build on this).

Pulled out into its own module because both approaches are invoked once per
HPO trial (the ifBO loop in optimizer.py can run hundreds of these) and both
need the same two perf-sensitive properties: a per-thread tokenizer instance
(fast tokenizers mutate their own truncation/padding config in place, so a
single shared instance races under concurrent trials - see `load_tokenizer`)
and a full-corpus encoding cache so repeated trials over the same underlying
texts don't re-tokenize from scratch every time (see `encode_texts_cached`).
"""

import threading
from typing import Optional

import pandas as pd
import torch
from transformers import PreTrainedTokenizerBase

from automl.logger import get_logger

logger = get_logger()

_tokenizer_cache = threading.local()


def load_tokenizer(path: str) -> PreTrainedTokenizerBase:
    """Load (and cache) one tokenizer instance per THREAD.

    See module docstring: two trials tokenizing concurrently on separate
    threads race on a fast tokenizer's shared mutable truncation/padding
    state and crash with `RuntimeError: Already borrowed`. Caching one
    instance per thread keeps the "load once, reuse many times" benefit
    within a thread while giving each concurrently-running trial its own
    private tokenizer to mutate.
    """
    from transformers import AutoTokenizer

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


def encode_texts_cached(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_path: str,
) -> list[list[int]]:
    """Tokenize `texts` once, fully untruncated, caching each result by
    (tokenizer_path, text) so a later call for the *same* text - just under
    a different trial's `max_seq_length` - is answered from this in-memory
    dict instead of re-running the tokenizer.

    Every trial resamples train/val from the same fixed underlying pool of
    texts, so after the first trial nearly every text is already cached
    here and the cost collapses to dict lookups. Truncation to the trial's
    `max_seq_length` is applied afterwards (see `truncate_ids`), by slicing
    these cached full-length ids - so this cache is sized once by corpus
    size, not by the number of trials or `max_seq_length` values seen.
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


def truncate_ids(
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


def collate_sequences(batch, pad_value: int = 0):
    """Pad a batch to the length of its longest sequence.

    Sequences are stored un-padded (truncated) token ids, so padding
    happens here, per-batch, instead of once for the whole dataset at a
    fixed `max_seq_length`. If most texts are much shorter than
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


class TextSequenceDataset(torch.utils.data.Dataset):

    DEFAULT_LABEL_MASK = -100

    def __init__(
        self,
        full_input_ids: list[list[int]],
        labels,
        max_seq_len: int,
        sep_token_id: Optional[int] = None,
    ):
        # `full_input_ids` is already tokenized (see `encode_texts_cached`),
        # fully untruncated and WITHOUT attention_mask / token_type_ids -
        # only input_ids is ever used downstream, so keeping the other two
        # fields around wastes roughly 2/3 of the memory this dataset used
        # to hold. Truncation to this trial's `max_seq_len` happens here,
        # per-sample, so the same cached full encoding can be reused by
        # every trial regardless of its sampled `max_seq_length`.
        input_ids_list = [
            truncate_ids(ids, max_seq_len, sep_token_id) for ids in full_input_ids
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
