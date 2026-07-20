"""Corpus-trained byte-level BPE tokenizer for the recurrent (RNN/GRU/LSTM) models.

Unlike the borrowed ``distilbert-base-uncased`` WordPiece vocab (~30.5k tokens
feeding a *randomly initialised* embedding table), this trains a small BPE
vocab on the actual training split. Fewer, domain-specific tokens mean a much
smaller embedding table and far more gradient updates per embedding.

Byte-level BPE (GPT-2/RoBERTa style) is used so every input is representable
from the 256-byte base alphabet -- there is effectively no out-of-vocabulary
loss regardless of vocab size.

``[PAD]`` is fixed at id 0 to match ``padding_idx=0`` in ``RNNClassifier``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    normalizers,
    decoders,
)

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


class BPETokenizer:
    """Thin wrapper over a ``tokenizers.Tokenizer`` byte-level BPE model."""

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self.pad_id = tokenizer.token_to_id(PAD_TOKEN)
        self.unk_id = tokenizer.token_to_id(UNK_TOKEN)
        # PAD must be 0 so it lines up with the embedding's padding_idx.
        assert self.pad_id == 0, f"[PAD] must be id 0, got {self.pad_id}"

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 8000,
        lowercase: bool = True,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Train a fresh byte-level BPE tokenizer on ``texts``.

        Args:
            texts: training corpus (an iterable of strings; consumed once).
            vocab_size: target vocab size incl. special tokens (e.g. 8000, 16000).
            lowercase: lowercase + strip accents before tokenizing (uncased vocab).
            min_frequency: minimum pair frequency for a merge to be learned.
        """
        tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))

        norm = [normalizers.NFD()]
        if lowercase:
            norm += [normalizers.Lowercase(), normalizers.StripAccents()]
        tok.normalizer = normalizers.Sequence(norm)

        # add_prefix_space so a leading word tokenizes the same as a mid-sentence one.
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            # [PAD] first -> id 0; [UNK] -> id 1.
            special_tokens=[PAD_TOKEN, UNK_TOKEN],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [e.ids for e in self._tok.encode_batch(texts)]

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def __len__(self) -> int:
        return self._tok.get_vocab_size()

    def save(self, path: Union[str, Path]) -> None:
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))
