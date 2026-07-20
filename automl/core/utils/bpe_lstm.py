import numpy as np
import torch
from torch.utils.data import Dataset


class BPETokenizedDataset(Dataset):
    """Holds pre-encoded, variable-length BPE token-id sequences for dynamic padding.

    Texts are encoded ONCE up front (fast batched Rust call) and truncated to
    ``max_length``. Padding is deferred to ``make_bpe_collate`` so each batch is
    padded only to its own longest sequence. Ids are stored as int32 arrays to
    keep memory modest on the large corpora (yelp/dbpedia).
    """

    def __init__(self, texts, labels, tokenizer, max_length=256):
        encoded = tokenizer.encode_batch(list(texts))
        self.ids = [np.asarray(ids[:max_length], dtype=np.int32) for ids in encoded]
        self.labels = list(labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


def make_bpe_collate(pad_id: int = 0):
    """Return a collate_fn that pads each batch to its own longest sequence.

    Produces ``{"input_ids": LongTensor[B, L], "lengths": LongTensor[B],
    "labels": Tensor[B]}``. ``lengths`` (true token counts, clamped to >=1)
    let the model pack sequences and ignore padding. Labels keep their natural
    dtype so NaN placeholder labels (unlabelled test set) survive as floats
    rather than being corrupted by a forced long cast.
    """

    def collate(batch):
        seqs, labels = zip(*batch)
        lengths = [max(len(s), 1) for s in seqs]
        maxlen = max(lengths)
        input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            if len(s):
                input_ids[i, : len(s)] = torch.as_tensor(s, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "labels": torch.tensor(labels),
        }

    return collate
