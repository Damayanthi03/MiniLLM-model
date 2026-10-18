"""Evaluation page: held-out loss, perplexity and real sample generations."""
from __future__ import annotations

import json
import math

import streamlit as st

import config
from training.evaluate import run_evaluation
from ui.theme import callout, cards_row, esc, generated_box, metric_card, page_header, render, section
from views.common import get_loaded_model, get_manager


def _load_saved_report(loaded) -> dict | None:
    """Re-use a saved report only if it belongs to exactly this checkpoint."""
    try:
        report = json.loads(config.EVAL_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if report.get("checkpoint_saved_at") == loaded.meta.get("saved_at"):
        return report
    return None


def render_page() -> None:
    page_header("Evaluation", "Measure the saved model on text it never trained on.")

    loaded, error = get_loaded_model()
    if error:
        callout("error", f"<b>The checkpoint could not be loaded.</b> {esc(error)}")
        return
    if loaded is None:
        callout(
            "warn",
            "<b>Nothing to evaluate yet.</b> There is no trained checkpoint. Train the model first, then come back here.",
        )
        return
    if get_manager().is_running():
        callout("info", "A training job is running. This page evaluates the <b>previously saved</b> checkpoint.")
    if not loaded.meta.get("completed"):
        callout("warn", "This checkpoint comes from a run that was stopped early.")

    with st.expander("What is being measured?", expanded=False):
        st.markdown(
            """
**Validation loss** is the average cross-entropy (in nats per character) on the last 10% of the text, which the
model never trained on. Lower is better.

**Perplexity** = exp(validation loss). It is valid here because the loss is an average negative log-likelihood per
character. Read it as "the model is about as unsure as if it were choosing between *N* equally likely characters".
A model that guesses at random has a perplexity equal to the vocabulary size.

This is a **character-level** perplexity, so it cannot be compared with word-level numbers reported for other models.
"""
        )

    report = st.session_state.get("eval_report") or _load_saved_report(loaded)
    if report and report.get("checkpoint_saved_at") != loaded.meta.get("saved_at"):
        report = None   # stale: a newer checkpoint was trained

    if st.button("Run evaluation", type="primary", key="eval_btn"):
        try:
            with st.spinner("Evaluating on held-out text and generating samples..."):
                report = run_evaluation(loaded)
            config.EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            config.EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
            st.session_state["eval_report"] = report
        except Exception as exc:
            report = None
            callout("error", f"<b>Evaluation failed:</b> {esc(exc)}")

    if not report:
        st.caption("Press the button to compute the numbers. Nothing on this page is pre-filled.")
        return

    if report.get("dataset_changed_since_training"):
        callout("warn", "The dataset file has changed since this model was trained, so the held-out split is not identical to the one used then.")

    section("Results on held-out text")
    gap = report["val_loss"] - report["train_loss"]
    cards_row(
        [
            metric_card("Validation loss", f"{report['val_loss']:.3f}", f"{report['val_chars']:,} held-out characters", "violet"),
            metric_card("Perplexity", f"{report['val_perplexity']:.2f}", "exp(validation loss)", "violet"),
            metric_card("Training loss", f"{report['train_loss']:.3f}", f"perplexity {report['train_perplexity']:.2f}", "blue"),
            metric_card("Random-guess baseline", f"{report['baseline_loss']:.3f}", f"perplexity {report['baseline_perplexity']:.0f} (= vocabulary)", "amber"),
        ]
    )
    st.write("")
    improvement = report["baseline_loss"] - report["val_loss"]
    if improvement > 0:
        callout(
            "ok",
            f"<b>Better than guessing:</b> validation loss is {improvement:.2f} nats below the random baseline "
            f"(perplexity {report['val_perplexity']:.1f} versus {report['baseline_perplexity']:.0f}).",
        )
    else:
        callout("warn", "<b>Not better than random guessing</b> on held-out text. The model needs more training or a larger dataset.")
    if gap > 0.4:
        callout(
            "warn",
            f"<b>Overfitting signal:</b> validation loss is {gap:.2f} higher than training loss. With so little text the "
            "model starts memorising it - exactly the limitation expected for low-resource training.",
        )
    else:
        render(f'<div class="callout info">Gap between validation and training loss: <b>{gap:.2f}</b>.</div>')

    section("Sample generations from held-out prompts")
    st.caption(
        f"Prompts are real snippets from the held-out text. Temperature {report['sampling']['temperature']}, "
        f"top-k {report['sampling']['top_k']}. The reference is what the text actually says next."
    )
    for i, s in enumerate(report["samples"], start=1):
        st.markdown(f"**Sample {i}**")
        left, right = st.columns(2, gap="medium")
        with left:
            render('<div class="mono-label">Model continuation</div>')
            generated_box(s["prompt"], s["generated"])
        with right:
            render('<div class="mono-label">What the held-out text says</div>')
            generated_box(s["prompt"], s["actual_text"])
        st.write("")

    callout(
        "warn",
        "<b>Keep expectations realistic.</b> Low-resource training does not produce ChatGPT-level performance. "
        "This model has a few hundred thousand parameters and has seen a few thousand characters of text, whereas "
        "modern assistants have billions of parameters trained on trillions of tokens. "
        f"Report generated {esc(report['generated_at'])} after {report['steps_completed']:,} training steps.",
    )
