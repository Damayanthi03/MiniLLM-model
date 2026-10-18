"""State shared by all pages: the training manager, the loaded model and the status."""
from __future__ import annotations

import streamlit as st

import config
from config import ModelConfig, get_device
from model.checkpoint import CheckpointError, LoadedModel, load_checkpoint
from model.tokenizer import CharTokenizer
from model.transformer import MiniLM
from training.dataset import load_text
from training.manager import TrainingManager


@st.cache_resource
def get_manager() -> TrainingManager:
    """One manager for the whole server - this is what prevents duplicate training jobs."""
    return TrainingManager()


@st.cache_resource(show_spinner=False)
def _load_cached(path: str, mtime_ns: int) -> LoadedModel:
    # mtime_ns is only part of the cache key: a new checkpoint file reloads the model.
    return load_checkpoint(path, get_device())


def get_loaded_model() -> tuple[LoadedModel | None, str | None]:
    """Returns (loaded_model, error_message).  Both None when no checkpoint exists yet."""
    path = config.CHECKPOINT_PATH
    if not path.exists():
        return None, None
    try:
        return _load_cached(str(path), path.stat().st_mtime_ns), None
    except CheckpointError as exc:
        return None, str(exc)
    except Exception as exc:  # never crash the page because of a bad file
        return None, f"Unexpected error while loading the checkpoint: {exc}"


def get_status() -> dict:
    """label/kind used for the status badge: Not trained | Training | Trained (+ edge cases)."""
    manager = get_manager()
    loaded, error = get_loaded_model()
    if manager.is_running():
        label, kind = "Training", "running"
    elif error:
        label, kind = "Checkpoint error", "error"
    elif loaded is None:
        label, kind = "Not trained", "idle"
    elif loaded.meta.get("completed"):
        label, kind = "Trained", "ok"
    else:
        label, kind = "Partially trained", "warn"
    return {"label": label, "kind": kind, "loaded": loaded, "error": error}


@st.cache_resource(show_spinner=False)
def untrained_reference() -> dict:
    """Real (but untrained) architecture built on the sample text - used before any training."""
    text = load_text(config.DEFAULT_DATASET)
    tokenizer = CharTokenizer.from_text(text)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = MiniLM(cfg)
    return {"model": model, "tokenizer": tokenizer, "config": cfg, "chars": len(text)}


def fmt_loss(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def fmt_int(value) -> str:
    return "—" if value is None else f"{int(value):,}"
