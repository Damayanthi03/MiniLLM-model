"""The web app must import and render every page, with and without a checkpoint."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import config  # noqa: E402
from model.checkpoint import save_checkpoint  # noqa: E402
from tests.helpers import REPEATING_TEXT, tiny_model_cfg, tiny_train_cfg  # noqa: E402
from training.train import train  # noqa: E402

PAGE_NAMES = ["Dashboard", "Text Generation", "Model Training", "Model Information", "Evaluation"]


def _run(page: str) -> AppTest:
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=90)
    at.session_state["nav"] = page
    at.run()
    return at


@pytest.mark.parametrize("page", PAGE_NAMES)
def test_every_page_renders_without_a_checkpoint(page, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_PATH", tmp_path / "missing.pt")
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGE_NAMES)
def test_every_page_renders_with_a_trained_checkpoint(page, tmp_path, monkeypatch):
    ckpt = tmp_path / "trained.pt"
    train(tiny_model_cfg(), tiny_train_cfg(30), checkpoint_path=ckpt, device="cpu", text=REPEATING_TEXT, log_path=None)
    monkeypatch.setattr(config, "CHECKPOINT_PATH", ckpt)
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_corrupted_checkpoint_does_not_crash_the_app(tmp_path, monkeypatch):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"garbage")
    monkeypatch.setattr(config, "CHECKPOINT_PATH", bad)
    for page in PAGE_NAMES:
        at = _run(page)
        assert not at.exception, (page, [e.value for e in at.exception])
