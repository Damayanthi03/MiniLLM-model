"""Loading text, splitting it and producing mini-batches."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from config import resolve_dataset_path
from model.tokenizer import CharTokenizer


def load_text(path) -> str:
    p = resolve_dataset_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Dataset file is empty: {p}")
    return text


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class TextData:
    """Encodes text and holds a training split and a held-out validation split.

    The LAST `val_fraction` of the text is the validation set; the model never
    trains on it, so validation loss measures how well it handles unseen text.
    """

    def __init__(self, text: str, tokenizer: CharTokenizer, val_fraction: float, block_size: int):
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n_train = int(len(ids) * (1.0 - val_fraction))
        self.train = ids[:n_train]
        self.val = ids[n_train:]
        needed = block_size + 2
        if len(self.train) < needed or len(self.val) < needed:
            raise ValueError(
                f"The text is too short for context length {block_size}: it has {len(ids)} characters "
                f"({len(self.train)} train / {len(self.val)} validation) but each split needs at least "
                f"{needed}. Use a longer text or a smaller context length."
            )

    def split(self, name: str) -> torch.Tensor:
        if name == "train":
            return self.train
        if name == "val":
            return self.val
        raise ValueError("split must be 'train' or 'val'")

    def get_batch(self, split: str, batch_size: int, block_size: int, device, generator=None):
        """Random windows.  y is x shifted one character to the right (next-char targets)."""
        data = self.split(split)
        starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
        x = torch.stack([data[s : s + block_size] for s in starts.tolist()])
        y = torch.stack([data[s + 1 : s + 1 + block_size] for s in starts.tolist()])
        return x.to(device), y.to(device)
