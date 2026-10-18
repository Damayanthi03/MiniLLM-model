"""Text Generation page - uses the saved checkpoint, nothing else."""
from __future__ import annotations

import time

import streamlit as st

from model.generate import generate_text
from ui.theme import callout, esc, generated_box, page_header, render, section
from views.common import get_loaded_model, get_manager

EXAMPLES = ["The transformer", "Attention is", "Once upon", "Mina the robot", "Every morning", "In the library"]


def _set_prompt(text: str) -> None:
    st.session_state["gen_prompt"] = text


def render_page() -> None:
    page_header("Text Generation", "Write with the model you trained. Every character below is sampled from the saved checkpoint.")

    loaded, error = get_loaded_model()
    if error:
        callout("error", f"<b>The checkpoint could not be loaded.</b> {esc(error)}")
        return
    if loaded is None:
        callout(
            "warn",
            "<b>No trained checkpoint found.</b> MiniLM has not been trained yet, so there is nothing to generate "
            "from. Open <b>Model Training</b> in the sidebar and train the model first.",
        )
        return

    meta, tokenizer = loaded.meta, loaded.tokenizer
    if not meta.get("completed"):
        callout("warn", "<b>Partially trained model.</b> Training was stopped early, so the output will be rough.")
    if get_manager().is_running():
        callout("info", "A training job is running. Generation still uses the <b>previously saved</b> checkpoint.")

    st.session_state.setdefault("gen_prompt", EXAMPLES[0])

    left, right = st.columns([3, 2], gap="large")
    with left:
        section("Prompt")
        st.text_area("Prompt", key="gen_prompt", height=130, label_visibility="collapsed",
                     help="The model continues this text. Characters it never saw in training are skipped.")
        render('<div class="mono-label">Try an example</div>')
        cols = st.columns(3)
        for i, ex in enumerate(EXAMPLES):
            cols[i % 3].button(ex, key=f"ex_{i}", on_click=_set_prompt, args=(ex,))
    with right:
        section("Sampling controls")
        max_len = st.slider("Maximum output length (characters)", 20, 500, 200, 10)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05,
                                help="Low = safe and repetitive, high = surprising and error-prone.")
        top_k = st.slider("Top-k (0 = off)", 0, tokenizer.vocab_size, min(20, tokenizer.vocab_size), 1,
                          help="Sample only from the k most likely characters.")
        seed_text = st.text_input("Seed (optional, for repeatable output)", value="", placeholder="leave empty for random")

    prompt = st.session_state.get("gen_prompt", "")
    unknown = tokenizer.unknown_chars(prompt)
    if unknown:
        shown = ", ".join(repr(c) for c in sorted(unknown)[:12])
        callout("info", f"These prompt characters are not in the model's vocabulary and will be skipped: {esc(shown)}")

    seed = None
    if seed_text.strip():
        try:
            seed = int(seed_text.strip())
        except ValueError:
            callout("error", "The seed must be a whole number.")
            return

    st.write("")
    if st.button("Generate", type="primary", key="gen_go"):
        try:
            with st.spinner("Sampling from the model..."):
                t0 = time.time()
                continuation = generate_text(
                    loaded.model, tokenizer, prompt, max_new_tokens=max_len,
                    temperature=temperature, top_k=top_k, seed=seed,
                )
                took = time.time() - t0
            st.session_state["gen_result"] = {
                "prompt": prompt, "text": continuation, "took": took,
                "settings": f"max {max_len} · temperature {temperature} · top-k {top_k if top_k else 'off'}"
                            + (f" · seed {seed}" if seed is not None else ""),
                "steps": meta.get("steps_completed"),
            }
        except Exception as exc:
            callout("error", f"<b>Generation failed:</b> {esc(exc)}")

    result = st.session_state.get("gen_result")
    if result:
        section("Output")
        generated_box(result["prompt"], result["text"])
        st.caption(
            f"Generated {len(result['text'])} characters in {result['took']:.2f}s · {result['settings']} · "
            f"checkpoint after {result['steps']} training steps"
        )

    st.write("")
    callout(
        "warn",
        "<b>Expect imperfect text.</b> This is a small character-level model trained on a few thousand characters. "
        "Words may be misspelled or sentences may drift - that is normal and is the honest result of low-resource training.",
    )
