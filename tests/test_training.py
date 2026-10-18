import threading

import torch

from model.checkpoint import load_checkpoint
from model.tokenizer import CharTokenizer
from model.transformer import MiniLM
from training.dataset import TextData
from training.evaluate import perplexity, run_evaluation
from training.manager import TrainingManager
from training.train import get_lr, train, train_step
from tests.helpers import REPEATING_TEXT, tiny_model_cfg, tiny_train_cfg


def test_batches_are_shifted_by_one_character():
    tok = CharTokenizer.from_text(REPEATING_TEXT)
    data = TextData(REPEATING_TEXT, tok, 0.1, 16)
    x, y = data.get_batch("train", 4, 16, "cpu")
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_train_and_validation_splits_do_not_overlap():
    tok = CharTokenizer.from_text(REPEATING_TEXT)
    data = TextData(REPEATING_TEXT, tok, 0.1, 16)
    assert len(data.train) + len(data.val) == len(REPEATING_TEXT)
    assert len(data.val) == len(REPEATING_TEXT) - int(len(REPEATING_TEXT) * 0.9)


def test_one_training_step_updates_weights():
    torch.manual_seed(0)
    tok = CharTokenizer.from_text(REPEATING_TEXT)
    data = TextData(REPEATING_TEXT, tok, 0.1, 16)
    model = MiniLM(tiny_model_cfg(tok.vocab_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x, y = data.get_batch("train", 8, 16, "cpu")
    loss = train_step(model, optimizer, x, y)
    after = model.state_dict()
    assert loss > 0
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert "tok_emb.weight" in changed
    assert any(k.startswith("blocks.0.attn") for k in changed)
    assert any(k.startswith("blocks.1.ffn") for k in changed)


def test_full_training_run_reduces_loss_and_saves_checkpoint(tmp_path):
    ckpt = tmp_path / "m.pt"
    events = []
    result = train(
        tiny_model_cfg(), tiny_train_cfg(60), checkpoint_path=ckpt,
        progress_callback=events.append, device="cpu", text=REPEATING_TEXT, log_path=tmp_path / "log.json",
    )
    h = result["history"]
    assert result["completed"] and result["steps_completed"] == 60
    assert h["step"][0] == 0 and h["step"][-1] == 60
    assert h["train_loss"][-1] < h["train_loss"][0]
    assert h["val_loss"][-1] < h["val_loss"][0]
    assert ckpt.exists() and (tmp_path / "log.json").exists()
    assert len(events) >= 60                      # real progress events, one per step
    loaded = load_checkpoint(ckpt, "cpu")
    assert loaded.meta["completed"] is True and loaded.meta["steps_completed"] == 60


def test_learning_rate_schedule_warms_up_then_decays():
    lrs = [get_lr(s, 1000, 1e-3) for s in (1, 50, 500, 1000)]
    assert lrs[0] < lrs[1] and lrs[1] >= lrs[2] >= lrs[3] > 0


def test_stop_event_saves_partial_checkpoint(tmp_path):
    stop = threading.Event()
    stop.set()
    result = train(
        tiny_model_cfg(), tiny_train_cfg(60), checkpoint_path=tmp_path / "p.pt",
        stop_event=stop, device="cpu", text=REPEATING_TEXT, log_path=None,
    )
    assert result["completed"] is False and result["steps_completed"] == 0
    assert load_checkpoint(tmp_path / "p.pt", "cpu").meta["completed"] is False


def test_perplexity_is_exp_of_loss():
    import math

    assert abs(perplexity(math.log(7.0)) - 7.0) < 1e-9
    assert perplexity(1e6) < float("inf")          # clamped, no overflow


def test_evaluation_report_uses_real_numbers(tmp_path):
    ckpt = tmp_path / "e.pt"
    train(tiny_model_cfg(), tiny_train_cfg(30), checkpoint_path=ckpt, device="cpu", text=REPEATING_TEXT, log_path=None)
    loaded = load_checkpoint(ckpt, "cpu")
    report = run_evaluation(loaded, text=REPEATING_TEXT, num_samples=2, prompt_chars=8, sample_chars=20)
    assert report["val_loss"] > 0 and report["val_perplexity"] > 1
    assert len(report["samples"]) == 2 and report["samples"][0]["generated"]


def test_manager_refuses_duplicate_jobs(tmp_path):
    manager = TrainingManager()
    cfg = tiny_train_cfg(5000)
    cfg.eval_interval = 500
    assert manager.start(tiny_model_cfg(), cfg, device="cpu", checkpoint_path=tmp_path / "a.pt", text=REPEATING_TEXT, log_path=None)
    assert manager.is_running()
    assert not manager.start(tiny_model_cfg(), cfg, device="cpu", checkpoint_path=tmp_path / "b.pt", text=REPEATING_TEXT, log_path=None)
    manager.stop()
    manager.join(60)
    assert not manager.is_running()
    assert manager.snapshot()["status"] == "stopped"


def test_manager_reports_errors_instead_of_crashing(tmp_path):
    manager = TrainingManager()
    cfg = tiny_train_cfg(10)
    manager.start(tiny_model_cfg(), cfg, device="cpu", checkpoint_path=tmp_path / "x.pt", text="too short", log_path=None)
    manager.join(30)
    snap = manager.snapshot()
    assert snap["status"] == "error" and "too short" in snap["error"].lower()
