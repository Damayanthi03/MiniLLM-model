"""Dashboard page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from training.evaluate import perplexity
from ui.theme import badge, callout, cards_row, chip, esc, flow_diagram, metric_card, render, section
from views.common import fmt_int, fmt_loss, get_status, untrained_reference
from config import get_device


def _go(page: str) -> None:
    st.session_state["nav"] = page


def render_page() -> None:
    status = get_status()
    loaded = status["loaded"]

    if loaded is not None:
        cfg = loaded.model.config
        n_params = loaded.model.count_parameters()
        vocab = loaded.tokenizer.vocab_size
        steps = loaded.meta.get("steps_completed", 0)
        val_loss = loaded.meta.get("final_val_loss")
        trained_on = loaded.meta.get("device", "unknown")
    else:
        ref = untrained_reference()
        cfg = ref["config"]
        n_params = ref["model"].count_parameters()
        vocab = ref["tokenizer"].vocab_size
        steps, val_loss, trained_on = 0, None, None

    render(
        f"""
        <div class="hero">
          <h1>MiniLM</h1>
          <p>A lightweight decoder-only Transformer language model, written and trained completely from scratch
          in PyTorch. Random weights, our own training loop, no pretrained models and no external AI APIs.</p>
          <div class="tagline">{badge(status['label'], status['kind'])}
          {chip("trained from scratch")}{chip("character-level", True)}{chip("PyTorch")}{chip("runs on CPU", True)}</div>
        </div>
        """
    )

    if status["error"]:
        callout("error", f"<b>The saved checkpoint could not be loaded.</b> {esc(status['error'])}")
    elif status["kind"] == "idle":
        callout(
            "info",
            "<b>The model has not been trained yet.</b> The numbers below describe the architecture only. "
            "Open <b>Model Training</b> in the sidebar and start a run - it takes real gradient steps on your CPU.",
        )
    elif status["kind"] == "warn":
        callout("warn", "<b>Training was stopped before it finished.</b> This checkpoint is only partially trained.")

    section("Model status")
    cards_row(
        [
            metric_card("Status", status["label"], "from the saved checkpoint" if loaded else "no checkpoint yet",
                        {"ok": "green", "running": "blue", "warn": "amber", "error": "red"}.get(status["kind"], "violet")),
            metric_card("Parameters", f"{n_params:,}", "trainable weights" if loaded else "architecture only (untrained)", "blue"),
            metric_card("Vocabulary", f"{vocab}", "distinct characters", "violet"),
            metric_card("Device", str(get_device()), f"trained on: {trained_on}" if trained_on else "auto-detected", "blue"),
        ]
    )
    st.write("")
    ppl = perplexity(val_loss) if val_loss is not None else None
    cards_row(
        [
            metric_card("Training steps", fmt_int(steps), "completed gradient updates", "green" if steps else "violet"),
            metric_card("Validation loss", fmt_loss(val_loss), "held-out text", "violet"),
            metric_card("Perplexity", "—" if ppl is None else f"{ppl:.2f}", "exp(validation loss)", "violet"),
            metric_card("Dataset", fmt_int(loaded.meta.get("dataset_chars")) if loaded else "—", "characters of text", "blue"),
        ]
    )

    section("Architecture")
    render(
        "".join(
            [
                chip(f"{cfg.n_layer} Transformer blocks"),
                chip(f"{cfg.n_head} attention heads", True),
                chip(f"embedding dim {cfg.n_embd}"),
                chip(f"context length {cfg.block_size}", True),
                chip(f"head size {cfg.n_embd // cfg.n_head}"),
                chip(f"feed-forward width {4 * cfg.n_embd}", True),
                chip(f"dropout {cfg.dropout}"),
            ]
        )
    )
    st.write("")
    flow_diagram(
        [
            ("1", "Characters", "Text is split into single characters"),
            ("2", "Embeddings", "Token + position vectors"),
            ("3", f"{cfg.n_layer} x Block", "Causal attention + feed-forward, with residuals and LayerNorm"),
            ("4", "Output layer", "Scores for every possible next character"),
            ("5", "Sampling", "Temperature / top-k picks the next character"),
        ]
    )

    if loaded is not None and loaded.meta.get("history", {}).get("step"):
        section("Training curve (from the saved checkpoint)")
        h = loaded.meta["history"]
        df = pd.DataFrame({"Training loss": h["train_loss"], "Validation loss": h["val_loss"]}, index=h["step"])
        df.index.name = "step"
        st.line_chart(df, color=["#60a5fa", "#a78bfa"])

    section("How to use this app")
    cols = st.columns(3)
    with cols[0]:
        render('<div class="mini-card"><h4>1 · Train</h4><p>Pick a preset and press Start. Loss values come straight from the running training loop.</p></div>')
        st.button("Go to Model Training", on_click=_go, args=("Model Training",), key="go_train")
    with cols[1]:
        render('<div class="mini-card"><h4>2 · Generate</h4><p>Type a prompt and let the saved model continue it, one character at a time.</p></div>')
        st.button("Go to Text Generation", on_click=_go, args=("Text Generation",), key="go_gen")
    with cols[2]:
        render('<div class="mini-card"><h4>3 · Evaluate</h4><p>Measure validation loss and perplexity on text the model never trained on.</p></div>')
        st.button("Go to Evaluation", on_click=_go, args=("Evaluation",), key="go_eval")

    st.write("")
    callout(
        "warn",
        "<b>Honest note:</b> MiniLM is a tiny educational model trained on a few thousand characters. "
        "It learns spelling, word shapes and sentence rhythm, but it does not understand meaning and it is "
        "nowhere near ChatGPT-level quality.",
    )
