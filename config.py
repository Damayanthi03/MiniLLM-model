"""Central configuration for MiniLM.

Everything tunable lives here: file locations, model size and training
settings.  Keeping it in one place makes the project easy to explain.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Folders and files
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"

SAMPLE_TEXT_PATH = DATA_DIR / "sample_text.txt"
CUSTOM_TEXT_PATH = DATA_DIR / "custom_text.txt"
CHECKPOINT_PATH = CHECKPOINT_DIR / "minilm.pt"
TRAIN_LOG_PATH = OUTPUT_DIR / "training_log.json"
EVAL_REPORT_PATH = OUTPUT_DIR / "evaluation_report.json"

# Dataset paths are stored relative to the project root so a checkpoint keeps
# working after the project is moved (e.g. Replit -> your laptop).
DEFAULT_DATASET = "data/sample_text.txt"
CUSTOM_DATASET = "data/custom_text.txt"


def resolve_dataset_path(path: str | Path) -> Path:
    """Turn a stored dataset path into a real path on this machine."""
    p = Path(path)
    if not p.is_absolute():
        return ROOT / p
    if p.exists():
        return p
    # Absolute path from another machine: fall back to our data folder.
    return DATA_DIR / p.name


# --------------------------------------------------------------------------
# Model and training settings
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 0        # filled in from the tokenizer
    block_size: int = 128      # context length (characters the model can see)
    n_embd: int = 128          # embedding dimension
    n_head: int = 4            # attention heads
    n_layer: int = 2           # transformer blocks
    dropout: float = 0.1

    def validate(self) -> None:
        for name in ("block_size", "n_embd", "n_head", "n_layer"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class TrainConfig:
    max_steps: int = 1500
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20       # random batches used to estimate training loss
    val_fraction: float = 0.1    # last 10% of the text is held out
    seed: int = 1337
    num_threads: int = 4         # keeps CPU use reasonable on shared machines
    dataset_path: str = DEFAULT_DATASET

    def validate(self) -> None:
        if self.max_steps < 1 or self.batch_size < 1 or self.eval_interval < 1:
            raise ValueError("max_steps, batch_size and eval_interval must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError("val_fraction must be between 0 and 0.5")


# Presets shown in the UI.  They only change how long we train - the training
# code is exactly the same.  Run time depends entirely on the machine's CPU.
PRESETS = {
    "Quick demo": {"max_steps": 300, "batch_size": 32, "eval_interval": 50},
    "Standard": {"max_steps": 1500, "batch_size": 32, "eval_interval": 100},
    "Extended": {"max_steps": 5000, "batch_size": 32, "eval_interval": 250},
}


def get_device():
    """Use CUDA if it is available, otherwise the CPU.  Never requires a GPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
