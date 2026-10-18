import pytest
import torch

from model.checkpoint import CheckpointError, CheckpointNotFoundError, load_checkpoint, save_checkpoint
from model.generate import generate_text
from model.tokenizer import CharTokenizer
from model.transformer import MiniLM
from tests.helpers import REPEATING_TEXT, tiny_model_cfg, tiny_train_cfg


@pytest.fixture()
def saved(tmp_path):
    torch.manual_seed(0)
    tok = CharTokenizer.from_text(REPEATING_TEXT)
    cfg = tiny_model_cfg(tok.vocab_size)
    model = MiniLM(cfg)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tok, cfg, tiny_train_cfg(), {"step": [], "train_loss": [], "val_loss": []}, 7, True)
    return path, model, tok


def test_checkpoint_roundtrip_gives_identical_outputs(saved):
    path, model, tok = saved
    loaded = load_checkpoint(path, "cpu")
    x = torch.randint(0, tok.vocab_size, (2, 16))
    with torch.no_grad():
        a, _ = model.eval()(x)
        b, _ = loaded.model(x)
    assert torch.allclose(a, b, atol=1e-6)
    assert loaded.tokenizer.chars == tok.chars
    assert loaded.meta["steps_completed"] == 7 and loaded.meta["completed"] is True
    assert loaded.meta["n_params"] == model.count_parameters()


def test_missing_checkpoint_raises_clear_error(tmp_path):
    with pytest.raises(CheckpointNotFoundError) as err:
        load_checkpoint(tmp_path / "nope.pt", "cpu")
    assert "train" in str(err.value).lower()


def test_corrupted_checkpoint_raises_checkpoint_error(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"this is not a checkpoint")
    with pytest.raises(CheckpointError):
        load_checkpoint(bad, "cpu")


def test_generation_from_loaded_checkpoint(saved):
    path, _, tok = saved
    loaded = load_checkpoint(path, "cpu")
    out = generate_text(loaded.model, loaded.tokenizer, "the cat", max_new_tokens=25, temperature=0.8, top_k=5, seed=3)
    assert len(out) == 25
    assert set(out) <= set(tok.chars)


def test_generation_is_repeatable_with_a_seed_and_greedy_is_deterministic(saved):
    path, _, _ = saved
    m = load_checkpoint(path, "cpu")
    a = generate_text(m.model, m.tokenizer, "the", 20, 1.0, 10, seed=5)
    b = generate_text(m.model, m.tokenizer, "the", 20, 1.0, 10, seed=5)
    assert a == b
    g1 = generate_text(m.model, m.tokenizer, "the", 20, temperature=0)
    g2 = generate_text(m.model, m.tokenizer, "the", 20, temperature=0)
    assert g1 == g2


def test_top_k_one_equals_greedy(saved):
    path, _, _ = saved
    m = load_checkpoint(path, "cpu")
    greedy = generate_text(m.model, m.tokenizer, "the", 15, temperature=0)
    top1 = generate_text(m.model, m.tokenizer, "the", 15, temperature=1.0, top_k=1, seed=9)
    assert greedy == top1


def test_generation_handles_empty_and_unknown_prompts(saved):
    path, _, _ = saved
    m = load_checkpoint(path, "cpu")
    assert len(generate_text(m.model, m.tokenizer, "", 10)) == 10
    assert len(generate_text(m.model, m.tokenizer, "ZZZ###", 10)) == 10


def test_generation_longer_than_context_window_works(saved):
    path, _, _ = saved
    m = load_checkpoint(path, "cpu")
    assert len(generate_text(m.model, m.tokenizer, "the cat", 60)) == 60      # block_size is only 16
