"""Model Training page - starts the real training function in a background thread."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from config import PRESETS, ModelConfig, TrainConfig, get_device
from training.dataset import load_text
from ui.theme import callout, cards_row, esc, metric_card, page_header, render, section
from views.common import fmt_loss, get_loaded_model, get_manager

DEFAULT_PRESET = "Standard"
DATASET_SAMPLE = "Built-in sample text"
DATASET_CUSTOM = "My uploaded text"


def _apply_preset() -> None:
    preset = PRESETS[st.session_state["tr_preset"]]
    st.session_state["tr_steps"] = preset["max_steps"]
    st.session_state["tr_batch"] = preset["batch_size"]
    st.session_state["tr_eval"] = preset["eval_interval"]


def _init_state() -> None:
    p = PRESETS[DEFAULT_PRESET]
    st.session_state.setdefault("tr_preset", DEFAULT_PRESET)
    st.session_state.setdefault("tr_steps", p["max_steps"])
    st.session_state.setdefault("tr_batch", p["batch_size"])
    st.session_state.setdefault("tr_eval", p["eval_interval"])
    st.session_state.setdefault("tr_lr", 1e-3)
    st.session_state.setdefault("tr_layers", 2)
    st.session_state.setdefault("tr_heads", 4)
    st.session_state.setdefault("tr_embd", 128)
    st.session_state.setdefault("tr_ctx", 128)
    st.session_state.setdefault("tr_dataset", DATASET_SAMPLE)


def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _history_frame(history: dict) -> pd.DataFrame | None:
    if not history["step"]:
        return None
    df = pd.DataFrame({"Training loss": history["train_loss"], "Validation loss": history["val_loss"]}, index=history["step"])
    df.index.name = "step"
    return df


def _monitor(manager) -> None:
    """Live view of the real training job.  Re-runs every second while training is active."""
    was_running = manager.is_running()

    @st.fragment(run_every=1.0 if was_running else None)
    def live() -> None:
        snap = manager.snapshot()
        if was_running and not manager.is_running():
            st.rerun()                     # job finished: refresh the whole page (status, checkpoint...)
            return

        status = snap["status"]
        if status == "idle":
            return

        section("Live training")
        step, max_steps = snap["step"], max(1, snap["max_steps"])
        if status == "running":
            st.progress(min(1.0, step / max_steps), text=f"Step {step:,} of {max_steps:,}")
        elapsed = snap["elapsed"]
        rate = step / elapsed if elapsed > 0 and step > 0 else None
        eta = (max_steps - step) / rate if rate else None
        cards_row(
            [
                metric_card("Step", f"{step:,}/{max_steps:,}", "gradient updates done", "blue"),
                metric_card("Batch loss", fmt_loss(snap["batch_loss"]), "latest mini-batch", "violet"),
                metric_card("Train loss", fmt_loss(snap["train_loss"]), "last evaluation", "blue"),
                metric_card("Validation loss", fmt_loss(snap["val_loss"]), "held-out text", "violet"),
            ]
        )
        st.write("")
        cards_row(
            [
                metric_card("Elapsed", _fmt_time(elapsed), "wall-clock time", "green"),
                metric_card("Speed", f"{rate:.1f}/s" if rate else "—", "steps per second", "green"),
                metric_card("Time left", _fmt_time(eta) if (eta is not None and status == "running") else "—", "estimate from speed so far", "amber"),
                metric_card("Learning rate", f"{snap['lr']:.2e}" if snap["lr"] else "—", "cosine schedule", "blue"),
            ]
        )

        df = _history_frame(snap["history"])
        section("Loss curves")
        if df is None:
            st.info("Waiting for the first evaluation...")
        else:
            st.line_chart(df, color=["#60a5fa", "#a78bfa"])
            st.caption("Points are real measurements taken every evaluation interval. Step 0 is the untrained model.")

        if status == "running":
            if st.button("Stop training", key="stop_btn"):
                manager.stop()
                st.toast("Stop requested - finishing the current step...")
        elif status == "finished":
            r = snap["result"]
            callout(
                "ok",
                f"<b>Training completed.</b> {r['steps_completed']:,} steps in {_fmt_time(r['elapsed_seconds'])}. "
                f"Validation loss went from {r['initial_loss']:.3f} to {r['final_val_loss']:.3f}. "
                f"Checkpoint saved to <code>{esc(r['checkpoint_path'])}</code>.",
            )
        elif status == "stopped":
            r = snap["result"]
            callout(
                "warn",
                f"<b>Training was stopped early</b> after {r['steps_completed']:,} steps. A partially trained "
                "checkpoint was saved and is marked as such.",
            )
        elif status == "error":
            callout("error", f"<b>Training failed:</b> {esc(snap['error'])}")
            with st.expander("Technical details"):
                st.code(snap["traceback"] or "", language="text")

    live()


def render_page() -> None:
    page_header("Model Training", "Train a brand-new, randomly initialised MiniLM. Everything shown here is measured, not simulated.")
    manager = get_manager()
    running = manager.is_running()
    _init_state()

    callout(
        "info",
        "<b>Real training, limited hardware.</b> Each step is a genuine forward pass, backpropagation and AdamW update on "
        "this machine's CPU (a GPU is used only if one is detected). Longer runs give lower loss but take longer - "
        "the app shows the measured speed so you can judge. Training runs inside the server process, so if the "
        "hosting environment restarts or sleeps, the run is lost.",
    )

    section("1 · Training run")
    st.radio("Preset", list(PRESETS), key="tr_preset", horizontal=True, on_change=_apply_preset, disabled=running)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.slider("Training steps", 50, 10000, step=50, key="tr_steps", disabled=running)
        st.select_slider("Batch size", options=[8, 16, 32, 64], key="tr_batch", disabled=running)
    with c2:
        st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], key="tr_lr",
                         format_func=lambda v: f"{v:g}", disabled=running)
        st.select_slider("Evaluate every (steps)", options=[25, 50, 100, 250, 500], key="tr_eval", disabled=running)

    with st.expander("Model architecture (advanced)"):
        a1, a2 = st.columns(2)
        with a1:
            st.slider("Transformer blocks", 1, 6, key="tr_layers", disabled=running)
            st.select_slider("Attention heads", options=[1, 2, 4, 8], key="tr_heads", disabled=running)
        with a2:
            st.select_slider("Embedding dimension", options=[64, 128, 192, 256], key="tr_embd", disabled=running)
            st.select_slider("Context length", options=[32, 64, 128, 256], key="tr_ctx", disabled=running)

    section("2 · Dataset")
    st.radio("Training text", [DATASET_SAMPLE, DATASET_CUSTOM], key="tr_dataset", horizontal=True, disabled=running)
    dataset_path = config.DEFAULT_DATASET
    if st.session_state["tr_dataset"] == DATASET_CUSTOM:
        dataset_path = config.CUSTOM_DATASET
        upload = st.file_uploader("Upload a .txt file (UTF-8)", type=["txt"], disabled=running)
        if upload is not None:
            try:
                raw = upload.getvalue().decode("utf-8")
                if not raw.strip():
                    raise ValueError("the file is empty")
                config.CUSTOM_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
                config.CUSTOM_TEXT_PATH.write_text(raw, encoding="utf-8")
                st.success(f"Saved {len(raw):,} characters as the custom training text.")
            except Exception as exc:
                st.error(f"Could not use that file: {exc}")

    text, text_error = None, None
    try:
        text = load_text(dataset_path)
    except Exception as exc:
        text_error = str(exc)
    if text_error:
        callout("error", f"<b>Dataset problem:</b> {esc(text_error)} Upload a text file or switch to the built-in sample.")
    else:
        ctx = st.session_state["tr_ctx"]
        enough = len(text) >= 11 * (ctx + 2)
        st.caption(f"{len(text):,} characters · {len(set(text))} distinct characters · last 10% is held out for validation")
        if not enough:
            callout("error", f"This text is too short for context length {ctx}. Use at least {11 * (ctx + 2):,} characters or lower the context length.")
            text_error = "too short"

    section("3 · Start")
    existing, _ = get_loaded_model()
    if existing is not None and not running:
        callout("warn", "A saved checkpoint already exists. Starting a new run trains from scratch and <b>replaces it</b> when the run ends.")

    started = False
    if st.button("Start training", type="primary", disabled=running or bool(text_error), key="start_btn"):
        try:
            model_cfg = ModelConfig(
                block_size=st.session_state["tr_ctx"], n_embd=st.session_state["tr_embd"],
                n_head=st.session_state["tr_heads"], n_layer=st.session_state["tr_layers"],
            )
            model_cfg.validate()
            train_cfg = TrainConfig(
                max_steps=st.session_state["tr_steps"], batch_size=st.session_state["tr_batch"],
                learning_rate=st.session_state["tr_lr"], eval_interval=st.session_state["tr_eval"],
                dataset_path=dataset_path,
            )
            train_cfg.validate()
            started = manager.start(model_cfg, train_cfg, device=get_device(), checkpoint_path=config.CHECKPOINT_PATH)
            if not started:
                st.warning("A training job is already running. Wait for it to finish or stop it first.")
        except Exception as exc:
            callout("error", f"<b>Could not start training:</b> {esc(exc)}")
    if started:
        st.rerun()                          # re-render so settings lock and the live monitor appears
    if running:
        st.caption("A training job is running - settings are locked and a second job cannot be started.")
        # Live monitor is rendered below.

    _monitor(manager)

    # Last saved run (shown when no job is active this session).
    if manager.snapshot()["status"] == "idle" and existing is not None:
        h = existing.meta.get("history", {})
        df = _history_frame(h) if h.get("step") else None
        if df is not None:
            section("Last saved run")
            st.caption(
                f"Checkpoint: {existing.meta.get('steps_completed', 0):,} steps · "
                f"{'completed' if existing.meta.get('completed') else 'stopped early'} · saved {existing.meta.get('saved_at', '?')}"
            )
            st.line_chart(df, color=["#60a5fa", "#a78bfa"])
