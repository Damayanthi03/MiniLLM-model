"""Autoregressive text generation with temperature and top-k sampling."""
from __future__ import annotations

import torch

from model.tokenizer import CharTokenizer
from model.transformer import MiniLM


@torch.no_grad()
def generate_ids(
    model: MiniLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Repeatedly predict the next token and append it.

    temperature < 1 -> safer / more repetitive,  > 1 -> more random.
    top_k keeps only the k most likely tokens before sampling.
    temperature <= 0 means greedy decoding (always the most likely token).
    """
    model.eval()
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]                 # the model only sees the last block_size tokens
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]                       # scores for the next token
        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                kth_best = torch.topk(logits, k).values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < kth_best, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def generate_text(
    model: MiniLM,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 20,
    seed: int | None = None,
) -> str:
    """Return only the newly generated continuation (not the prompt)."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    if not ids:
        # Empty prompt (or only unknown characters): start from a newline if
        # the vocabulary has one, otherwise from the first token.
        ids = [tokenizer.stoi.get("\n", 0)]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    out = generate_ids(model, idx, max_new_tokens, temperature, top_k, generator)
    return tokenizer.decode(out[0, len(ids):].tolist())
