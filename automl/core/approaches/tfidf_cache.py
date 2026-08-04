"""
On-disk cache for fitted TF-IDF vectorizers and their transformed matrices.

`TfidfFFNNApproach.prepare()` (see `tfidf_ffnn.py`) re-fits/re-transforms
from scratch on every call. But the optimizer creates a fresh `Approach` and
calls `prepare()` again for every trial (`base_optimizer.py`), including
when successive-halving/Hyperband *promotes the same configuration* to a
higher budget and when the incumbent is re-evaluated - cases where the text
corpus and every TF-IDF hyperparameter are identical to a prior call and
only `epochs` changed. Re-tokenizing in those cases is pure waste.

This module memoizes `fit_transform`/`transform` results on disk, keyed by
the vectorizer's constructor params plus a hash of the exact texts, so a
repeat call with an unchanged (params, corpus) pair loads the previous
result instead of re-tokenizing. Cached on disk (not in memory) so the
saving also survives across separate optimizer process runs, not just
within one.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from automl.logger import get_logger

logger = get_logger()

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "tfidf_vectorizer"

_key_locks: dict[str, threading.Lock] = {}
_key_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    # Guards against two threads racing to fit the *same* (params, corpus)
    # pair concurrently (harmless duplicate work) and against one thread
    # reading a cache file another thread is mid-write on.
    with _key_locks_guard:
        return _key_locks.setdefault(key, threading.Lock())


def cache_dir() -> Path:
    override = os.environ.get("AUTOML_TFIDF_CACHE_DIR")
    path = Path(override) if override else _DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hash_texts(texts: list[str]) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(len(texts)).encode())
    for text in texts:
        hasher.update(b"\x00")
        hasher.update(text.encode("utf-8", errors="surrogatepass"))
    return hasher.hexdigest()


def _hash_key(payload: dict[str, Any]) -> str:
    # default=str covers non-JSON-native values in TfidfVectorizer's
    # get_params() output (e.g. the `dtype` class object).
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def _atomic_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".joblib")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        joblib.dump(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _safe_load(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        # Corrupt/partial entry (e.g. process was killed mid-write, or the
        # cached sklearn objects were pickled by an incompatible version) -
        # ignore it and let the caller recompute & overwrite it.
        logger.warning(f"Ignoring unreadable TF-IDF cache entry at {path}: {e}")
        return None


def get_or_fit_train(
    vectorizer: TfidfVectorizer,
    char_vectorizer: Optional[TfidfVectorizer],
    texts: list[str],
) -> tuple[sp.csr_matrix, TfidfVectorizer, Optional[TfidfVectorizer], str]:
    """Fit `vectorizer` (and `char_vectorizer`, if given) on `texts`, unless
    an identical (params, corpus) pair was already fit before - in which
    case load the fitted vectorizer(s) and transformed matrix from disk.

    Returns `(X_train, fitted_vectorizer, fitted_char_vectorizer, train_key)`.
    `train_key` identifies this (params, corpus) pair and must be passed to
    `get_or_transform()` for the corresponding validation/test split.
    """
    key_payload: dict[str, Any] = {
        "vectorizer": vectorizer.get_params(),
        "char_vectorizer": char_vectorizer.get_params() if char_vectorizer else None,
        "texts": _hash_texts(texts),
    }
    train_key = _hash_key(key_payload)
    entry_path = cache_dir() / f"train-{train_key}.joblib"

    with _lock_for(train_key):
        cached = _safe_load(entry_path)
        if cached is not None:
            logger.debug(f"TF-IDF train cache hit ({train_key}), skipping fit.")
            return (
                cached["X"],
                cached["vectorizer"],
                cached.get("char_vectorizer"),
                train_key,
            )

        logger.debug(f"TF-IDF train cache miss ({train_key}), fitting vectorizer(s).")
        X = vectorizer.fit_transform(texts)
        if char_vectorizer is not None:
            X_char = char_vectorizer.fit_transform(texts)
            X = sp.hstack([X, X_char])
        X = X.tocsr()

        _atomic_dump(
            {"X": X, "vectorizer": vectorizer, "char_vectorizer": char_vectorizer},
            entry_path,
        )
        return X, vectorizer, char_vectorizer, train_key


def get_or_transform(
    train_key: str,
    vectorizer: TfidfVectorizer,
    char_vectorizer: Optional[TfidfVectorizer],
    texts: list[str],
) -> sp.csr_matrix:
    """`vectorizer.transform(texts)` (plus `char_vectorizer`, if given),
    memoized on disk under a key derived from `train_key` (identifying the
    fitted vectorizer) and a hash of `texts`.
    """
    val_key = _hash_key({"train_key": train_key, "texts": _hash_texts(texts)})
    entry_path = cache_dir() / f"transform-{val_key}.joblib"

    with _lock_for(val_key):
        cached = _safe_load(entry_path)
        if cached is not None:
            logger.debug(f"TF-IDF transform cache hit ({val_key}).")
            return cached["X"]

        logger.debug(f"TF-IDF transform cache miss ({val_key}).")
        X = vectorizer.transform(texts)
        if char_vectorizer is not None:
            X_char = char_vectorizer.transform(texts)
            X = sp.hstack([X, X_char])
        X = X.tocsr()

        _atomic_dump({"X": X}, entry_path)
        return X
