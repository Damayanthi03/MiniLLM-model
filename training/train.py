"""The real training loop.

Every step: take a batch -> forward pass -> cross-entropy loss -> backward pass
(backpropagation) -> AdamW update.  Nothing is simulated.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import torch

import config
from config import ModelConfig, TrainConfig, get_device
from model.checkpoint import save_checkpoint
from model.tokenizer import CharTokenizer
from model.transformer import MiniLM
from training.dataset import TextData, load_text, text_fingerprint
from training.evaluate import estimate_train_loss, evaluate_validation


def get_lr(step: int, max_steps: int, base_lr: float) -> float:
    """Linear warm-up, then cosine decay down to 10% of the base rate."""
    warmup = max(1, min(50, max_steps // 10))
    min_lr = 0.1 * base_lr
    if step <= warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_step(model: MiniLM, optimizer: torch.optim.Optimizer, x, y, grad_clip: float = 1.0) -> float:
    """One real gradient update.  Returns the batch loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()                                   # backpropagation
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()                                  # AdamW weight update
    return loss.item()


def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    checkpoint_path=None,
    progress_callback=None,
    stop_event=None,
    device=None,
    text: str | None = None,
    log_path=config.TRAIN_LOG_PATH,
) -> dict:
    """Train a fresh, randomly initialised MiniLM and save a checkpoint.

    progress_callback(event: dict) is called with real numbers as training runs.
    stop_event (threading.Event) lets the UI stop training early.
    Returns a result dict; result["completed"] is False if stopped early.
    """
    checkpoint_path = checkpoint_path or config.CHECKPOINT_PATH
    train_cfg.validate()
    device = torch.device(device) if device is not None else get_device()
    if train_cfg.num_threads:
        torch.set_num_threads(max(1, int(train_cfg.num_threads)))
    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    dataset_name = "inline text" if text is not None else Path(train_cfg.dataset_path).name
    if text is None:
        text = load_text(train_cfg.dataset_path)
    tokenizer = CharTokenizer.from_text(text)
    model_cfg = replace(model_cfg, vocab_size=tokenizer.vocab_size)
    data = TextData(text, tokenizer, train_cfg.val_fraction, model_cfg.block_size)

    model = MiniLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    gen = torch.Generator().manual_seed(train_cfg.seed)
    history = {"step": [], "train_loss": [], "val_loss": []}
    t0 = time.time()

    def emit(**event):
        if progress_callback is not None:
            event.setdefault("max_steps", train_cfg.max_steps)
            event["elapsed"] = time.time() - t0
            progress_callback(event)

    def evaluate_now(step: int, lr: float, batch_loss):
        tr = estimate_train_loss(model, data, train_cfg.batch_size, train_cfg.eval_batches, device, gen)
        va = evaluate_validation(model, data, train_cfg.batch_size, device)
        history["step"].append(step)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        emit(step=step, batch_loss=batch_loss, train_loss=tr, val_loss=va, lr=lr)

    # Loss before any learning (should be close to ln(vocab_size)).
    evaluate_now(0, 0.0, None)
    initial_loss = history["val_loss"][0]

    steps_done, stopped = 0, False
    for step in range(1, train_cfg.max_steps + 1):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        lr = get_lr(step, train_cfg.max_steps, train_cfg.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = data.get_batch("train", train_cfg.batch_size, model_cfg.block_size, device, gen)
        batch_loss = train_step(model, optimizer, x, y, train_cfg.grad_clip)
        steps_done = step
        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps:
            evaluate_now(step, lr, batch_loss)
        else:
            emit(step=step, batch_loss=batch_loss, lr=lr)

    if history["step"][-1] != steps_done:                 # stopped between evaluations
        evaluate_now(steps_done, get_lr(max(steps_done, 1), train_cfg.max_steps, train_cfg.learning_rate), None)

    elapsed = time.time() - t0
    completed = not stopped
    extra = {
        "device": str(device),
        "dataset_name": dataset_name,
        "dataset_sha": text_fingerprint(text),
        "dataset_chars": len(text),
        "elapsed_seconds": elapsed,
        "initial_loss": initial_loss,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
    }
    save_checkpoint(
        checkpoint_path, model, tokenizer, model_cfg, train_cfg, history, steps_done, completed, extra
    )
    result = {
        "completed": completed,
        "steps_completed": steps_done,
        "n_params": model.count_parameters(),
        "vocab_size": tokenizer.vocab_size,
        "checkpoint_path": str(checkpoint_path),
        **extra,
        "history": history,
    }
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
