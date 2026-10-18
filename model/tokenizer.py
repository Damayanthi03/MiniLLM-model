"""A tiny character-level tokenizer.

Every distinct character in the training text becomes one token id.
"""
from __future__ import annotations

from typing import Iterable


class CharTokenizer:
    def __init__(self, chars: Iterable[str]):
        self.chars: list[str] = sorted(set(chars))
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self.itos: dict[int, str] = {i: c for i, c in enumerate(self.chars)}

    # -- construction ------------------------------------------------------
    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(text)

    def to_dict(self) -> dict:
        return {"chars": list(self.chars)}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        return cls(data["chars"])

    # -- core API ----------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def unknown_chars(self, text: str) -> set[str]:
        return {c for c in text if c not in self.stoi}

    def encode(self, text: str) -> list[int]:
        """Text -> list of ids.  Characters not in the vocabulary are skipped."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: Iterable[int]) -> str:
        """List of ids -> text."""
        return "".join(self.itos[int(i)] for i in ids)
