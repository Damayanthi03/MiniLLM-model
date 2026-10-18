"""Tiny configurations so the tests run in seconds on a CPU."""
from config import ModelConfig, TrainConfig

REPEATING_TEXT = "the cat sat on the mat. " * 80       # 1,920 characters


def tiny_model_cfg(vocab_size: int = 0) -> ModelConfig:
    return ModelConfig(vocab_size=vocab_size, block_size=16, n_embd=32, n_head=2, n_layer=2, dropout=0.0)


def tiny_train_cfg(steps: int = 60) -> TrainConfig:
    return TrainConfig(
        max_steps=steps, batch_size=8, learning_rate=3e-3, eval_interval=30,
        eval_batches=2, num_threads=1, dataset_path="data/sample_text.txt",
    )
