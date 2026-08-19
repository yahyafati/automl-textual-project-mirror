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
    ids: list[int],
    max_seq_len: int,
    sep_token_id: Optional[int],
    strategy: str = "right",
    ellipsis_ids: Optional[list[int]] = None,
) -> list[int]:
    """Reproduce `tokenizer(text, truncation=True, max_length=max_seq_len)`
    from fully-encoded `ids` (`[CLS] + content + [SEP]`), truncating the
    *content* span (everything but CLS/SEP) according to `strategy`:

    - "right" (default): keep the first `max_seq_len - 2` content tokens,
      dropping the tail - matches the tokenizer's own
      right-truncation-before-SEP behavior.
    - "left": keep the last `max_seq_len - 2` content tokens, dropping the
      head.
    - "center": keep a prefix and a suffix of content with `ellipsis_ids`
      spliced in between, dropping the middle.

    If `sep_token_id` is None there's no CLS/SEP structure to preserve, so
    the same strategy is applied to the raw `ids` directly.
    """
    if len(ids) <= max_seq_len:
        return ids

    if sep_token_id is None:
        if strategy == "left":
            return ids[-max_seq_len:]
        if strategy == "center":
            return _splice_center(ids, max_seq_len, ellipsis_ids)
        return ids[:max_seq_len]

    cls_id = ids[0]
    content = ids[1:-1]
    budget = max_seq_len - 2  # room left over for CLS + SEP
    if budget <= 0:
        return ids[: max_seq_len - 1] + [sep_token_id]

    if strategy == "left":
        new_content = content[-budget:]
    elif strategy == "center":
        new_content = _splice_center(content, budget, ellipsis_ids)
    else:
        new_content = content[:budget]

    return [cls_id] + new_content + [sep_token_id]


def _splice_center(
    ids: list[int], budget: int, ellipsis_ids: Optional[list[int]]
) -> list[int]:
    """Keep a head and tail slice of `ids` totalling `budget` tokens, with
    `ellipsis_ids` spliced in between in place of the dropped middle."""
    ellipsis_ids = ellipsis_ids or []
    content_budget = budget - len(ellipsis_ids)
    if content_budget <= 0:
        return ids[-budget:] if budget > 0 else []
    head = (content_budget + 1) // 2
    tail = content_budget - head
    return ids[:head] + ellipsis_ids + (ids[-tail:] if tail else [])


_ellipsis_ids_cache: dict[str, list[int]] = {}
_ellipsis_ids_cache_lock = threading.Lock()


def get_ellipsis_ids(
    tokenizer: PreTrainedTokenizerBase, tokenizer_path: str
) -> list[int]:
    """Token id(s) for "..." under `tokenizer`, cached by `tokenizer_path`
    (mirrors `encode_texts_cached`'s per-tokenizer caching) - used to mark
    the dropped middle span in center-truncated sequences."""
    cached = _ellipsis_ids_cache.get(tokenizer_path)
    if cached is not None:
        return cached
    ids = tokenizer.encode("...", add_special_tokens=False)
    with _ellipsis_ids_cache_lock:
        _ellipsis_ids_cache.setdefault(tokenizer_path, ids)
    return _ellipsis_ids_cache[tokenizer_path]


def expand_with_truncation_augmentation(
    full_input_ids: list[list[int]],
    labels,
    max_seq_len: int,
    sep_token_id: Optional[int],
    ellipsis_ids: Optional[list[int]] = None,
) -> tuple[list[list[int]], Optional[list]]:
    """Data augmentation for training data: every item whose full
    (untruncated) length exceeds `max_seq_len` is emitted as 3 copies -
    right-truncated, left-truncated, and center-truncated with an ellipsis -
    instead of a single one-sided truncation, so the model sees content
    from all parts of long inputs rather than only ever the first
    `max_seq_len` tokens. Items already within `max_seq_len` are passed
    through as a single, unaugmented copy. Labels are duplicated alongside
    their source item.

    Only meant for training data - callers must NOT use this for
    validation/test data, since it changes the number and order of items
    relative to the input, which val/predict need to stay 1:1 with.
    """
    labels_list = list(labels) if labels is not None else None
    expanded_ids: list[list[int]] = []
    expanded_labels: Optional[list] = [] if labels_list is not None else None

    for i, ids in enumerate(full_input_ids):
        if len(ids) > max_seq_len:
            variants = [
                truncate_ids(ids, max_seq_len, sep_token_id, strategy="right"),
                truncate_ids(ids, max_seq_len, sep_token_id, strategy="left"),
                truncate_ids(
                    ids,
                    max_seq_len,
                    sep_token_id,
                    strategy="center",
                    ellipsis_ids=ellipsis_ids,
                ),
            ]
        else:
            variants = [ids]
        expanded_ids.extend(variants)
        if expanded_labels is not None:
            expanded_labels.extend([labels_list[i]] * len(variants))

    return expanded_ids, expanded_labels


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

    lengths: list[int] = [seq.size(0) for seq in sequences]
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
        input_ids_list = [
            truncate_ids(ids, max_seq_len, sep_token_id) for ids in full_input_ids
        ]

        lengths = torch.tensor([len(ids) for ids in input_ids_list], dtype=torch.long)
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
        self.input_ids = (
            torch.cat([torch.tensor(ids, dtype=torch.int32) for ids in input_ids_list])
            if input_ids_list
            else torch.empty(0, dtype=torch.int32)
        )
        self._len = len(input_ids_list)

        if labels is not None:
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
