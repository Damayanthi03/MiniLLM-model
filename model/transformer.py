"""Decoder-only Transformer (GPT style), written from scratch in PyTorch.

Data flow for a batch of token ids  (B = batch, T = length, C = embedding):

    ids (B,T) -> token emb + position emb -> [Block x N] -> LayerNorm -> Linear -> logits (B,T,V)

Each Block is:   x = x + Attention(LayerNorm(x));   x = x + FeedForward(LayerNorm(x))
All weights start random (no pretrained weights anywhere).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t may only look at positions <= t."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # queries, keys, values
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)      # mix the heads back together
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular matrix: True where attention is allowed.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer(
            "causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, heads, T, T)
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)                              # each row sums to 1
        out = self.attn_dropout(weights) @ v                             # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))
        if return_attention:
            return out, weights
        return out


class FeedForward(nn.Module):
    """Position-wise two-layer network (expands to 4x, then projects back)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One Transformer block with residual connections and layer norm."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if return_attention:
            attn_out, weights = self.attn(self.ln1(x), return_attention=True)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, weights
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        if cfg.vocab_size < 1:
            raise ValueError("cfg.vocab_size must be set from the tokenizer before building the model")
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: output layer re-uses the token-embedding matrix.
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Random initialisation - this is what "trained from scratch" means.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.config.block_size}")
        pos = torch.arange(T, device=idx.device)
        return self.drop(self.tok_emb(idx) + self.pos_emb(pos))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids.  Returns (logits (B,T,V), loss or None)."""
        x = self._embed(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def attention_maps(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """Real attention weights of every layer, each shaped (B, heads, T, T)."""
        was_training = self.training
        self.eval()
        x = self._embed(idx)
        maps = []
        for block in self.blocks:
            x, weights = block(x, return_attention=True)
            maps.append(weights)
        self.train(was_training)
        return maps

    def count_parameters(self, trainable_only: bool = True) -> int:
        # .parameters() yields the tied embedding/output matrix only once.
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
