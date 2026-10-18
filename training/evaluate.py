"""Evaluation on held-out text, perplexity and sample generations."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import torch

from model.checkpoint import LoadedModel
from model.generate import generate_text
from model.transformer import MiniLM
from training.dataset import TextData, load_text, text_fingerprint


def perplexity(loss: float) -> float:
    """exp(cross-entropy).  Valid because the loss is an average negative
    log-likelihood in nats per character.  Clamped to avoid float overflow."""
    return math.exp(min(float(loss), 50.0))


@torch.no_grad()
def evaluate_validation(model: MiniLM, data: TextData, batch_size: int, device) -> float:
    """Average loss over ALL non-overlapping windows of the validation split (deterministic)."""
    model.eval()
    block = model.config.block_size
    tokens = data.val
    n_windows = (len(tokens) - 1) // block
    if n_windows < 1:
        raise ValueError("Validation split is shorter than one context window")
    starts = [i * block for i in range(n_windows)]
    total, count = 0.0, 0
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        x = torch.stack([tokens[s : s + block] for s in chunk]).to(device)
        y = torch.stack([tokens[s + 1 : s + 1 + block] for s in chunk]).to(device)
        _, loss = model(x, y)
        total += loss.item() * x.numel()
        count += x.numel()
    return total / count


@torch.no_grad()
def estimate_train_loss(model: MiniLM, data: TextData, batch_size: int, num_batches: int, device, generator=None) -> float:
    """Average loss over random training windows (dropout off)."""
    model.eval()
    block = model.config.block_size
    losses = []
    for _ in range(num_batches):
        x, y = data.get_batch("train", batch_size, block, device, generator)
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def run_evaluation(
    loaded: LoadedModel,
    text: str | None = None,
    num_samples: int = 3,
    prompt_chars: int = 24,
    sample_chars: int = 160,
    temperature: float = 0.8,
    top_k: int = 20,
    batch_size: int = 16,
    seed: int = 1234,
) -> dict:
    """Real evaluation of a saved model on its held-out text.  Returns a report dict."""
    meta = loaded.meta
    tcfg = meta["train_config"]
    model, tokenizer = loaded.model, loaded.tokenizer
    block = model.config.block_size
    if text is None:
        text = load_text(tcfg["dataset_path"])

    data = TextData(text, tokenizer, tcfg["val_fraction"], block)
    val_loss = evaluate_validation(model, data, batch_size, loaded.device)
    gen = torch.Generator().manual_seed(seed)
    train_loss = estimate_train_loss(model, data, batch_size, 10, loaded.device, gen)
    baseline = math.log(tokenizer.vocab_size)            # loss of guessing uniformly at random

    # Sample generations: prompts are real held-out snippets, so we can show
    # what the model wrote next to what the text actually says.
    val_text = tokenizer.decode(data.val.tolist())
    samples = []
    span = prompt_chars + sample_chars
    max_pos = max(0, len(val_text) - span)
    for k in range(num_samples):
        pos = int(max_pos * k / max(1, num_samples - 1)) if num_samples > 1 else 0
        prompt = val_text[pos : pos + prompt_chars]
        reference = val_text[pos + prompt_chars : pos + span]
        generated = generate_text(
            model, tokenizer, prompt, max_new_tokens=max(1, len(reference)),
            temperature=temperature, top_k=top_k, seed=seed + k,
        )
        samples.append({"prompt": prompt, "generated": generated, "actual_text": reference})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_saved_at": meta.get("saved_at"),
        "steps_completed": meta.get("steps_completed"),
        "training_completed": meta.get("completed"),
        "n_params": meta.get("n_params"),
        "vocab_size": tokenizer.vocab_size,
        "val_loss": val_loss,
        "val_perplexity": perplexity(val_loss),
        "train_loss": train_loss,
        "train_perplexity": perplexity(train_loss),
        "baseline_loss": baseline,
        "baseline_perplexity": float(tokenizer.vocab_size),
        "train_chars": int(len(data.train)),
        "val_chars": int(len(data.val)),
        "dataset_changed_since_training": (
            meta.get("dataset_sha") is not None and meta.get("dataset_sha") != text_fingerprint(text)
        ),
        "sampling": {"temperature": temperature, "top_k": top_k},
        "samples": samples,
    }
