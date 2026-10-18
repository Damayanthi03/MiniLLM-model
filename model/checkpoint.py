"""Saving and loading trained models."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from model.tokenizer import CharTokenizer
from model.transformer import MiniLM


class CheckpointError(RuntimeError):
    """The checkpoint file exists but cannot be used."""


class CheckpointNotFoundError(FileNotFoundError):
    """No checkpoint has been saved yet (the model has not been trained)."""


@dataclass
class LoadedModel:
    model: MiniLM
    tokenizer: CharTokenizer
    meta: dict
    device: torch.device


def save_checkpoint(
    path,
    model: MiniLM,
    tokenizer: CharTokenizer,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    history: dict,
    steps_completed: int,
    completed: bool,
    extra: dict | None = None,
) -> Path:
    """Write the weights plus everything needed to rebuild the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "tokenizer": tokenizer.to_dict(),
        "history": history,
        "steps_completed": int(steps_completed),
        "completed": bool(completed),
        "n_params": model.count_parameters(),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomic: never leaves a half-written checkpoint
    return path


def load_checkpoint(path, device=None) -> LoadedModel:
    """Rebuild the model from a checkpoint.

    Raises CheckpointNotFoundError if nothing was saved yet and
    CheckpointError if the file is unreadable.
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointNotFoundError(
            f"No checkpoint found at {path}. Train the model first on the Model Training page."
        )
    device = torch.device(device) if device is not None else torch.device("cpu")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
        model_cfg = ModelConfig(**payload["model_config"])
        tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
        model = MiniLM(model_cfg)
        model.load_state_dict(payload["model_state"])
    except Exception as exc:  # corrupted / incompatible file
        raise CheckpointError(f"Checkpoint at {path} could not be loaded: {exc}") from exc
    model.to(device)
    model.eval()
    meta = {k: v for k, v in payload.items() if k != "model_state"}
    return LoadedModel(model=model, tokenizer=tokenizer, meta=meta, device=device)
