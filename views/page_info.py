"""Model Information page: plain-language explanation, parameter table, attention explorer."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
import torch

from ui.theme import callout, cards_row, chip, esc, flow_diagram, info_card, metric_card, page_header, render, section, table
from views.common import get_loaded_model, untrained_reference

BLOCK_PARTS = {"attn": "Causal self-attention", "ffn": "Feed-forward network", "ln1": "LayerNorm", "ln2": "LayerNorm"}


def parameter_rows(model) -> list[list]:
    """Group the model's real parameters into a readable table."""
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        parts = name.split(".")
        if parts[0] == "blocks":
            key = f"Block {int(parts[1]) + 1} · {BLOCK_PARTS.get(parts[2], parts[2])}"
        elif parts[0] == "tok_emb":
            key = "Token embeddings (also used as the output layer)"
        elif parts[0] == "pos_emb":
            key = "Position embeddings"
        elif parts[0] == "ln_f":
            key = "Final LayerNorm"
        else:
            key = parts[0]
        groups[key] = groups.get(key, 0) + p.numel()
    return [[k, f"{v:,}"] for k, v in groups.items()]


def _attention_explorer(loaded) -> None:
    section("Attention explorer (real weights from your trained model)")
    cfg = loaded.model.config
    default_text = "Mina the robot listened to the birds"
    text = st.text_input("Text to inspect (first 40 characters are used)", value=default_text, key="attn_text")
    ids = loaded.tokenizer.encode(text)[: min(40, cfg.block_size)]
    if len(ids) < 2:
        st.info("Enter at least two characters the model knows.")
        return
    c1, c2 = st.columns(2)
    layer = c1.selectbox("Layer", list(range(1, cfg.n_layer + 1)), key="attn_layer")
    head = c2.selectbox("Head", list(range(1, cfg.n_head + 1)), key="attn_head")

    idx = torch.tensor([ids], dtype=torch.long, device=loaded.device)
    maps = loaded.model.attention_maps(idx)
    weights = maps[layer - 1][0, head - 1].cpu().tolist()       # T x T, row = query position
    chars = [loaded.tokenizer.itos[i] for i in ids]
    labels = [f"{i:02d} {'␣' if ch == ' ' else ch}" for i, ch in enumerate(chars)]
    rows = []
    for qi, row in enumerate(weights):
        for ki, w in enumerate(row):
            rows.append({"query": labels[qi], "key": labels[ki], "weight": float(w)})
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("key:O", sort=labels, title="looks at (earlier characters)"),
            y=alt.Y("query:O", sort=labels, title="current character"),
            color=alt.Color("weight:Q", scale=alt.Scale(scheme="purples"), title="attention"),
            tooltip=["query", "key", alt.Tooltip("weight:Q", format=".3f")],
        )
        .properties(height=420, width="container")
    )
    st.altair_chart(chart)
    st.caption(
        "Each row shows how one character spreads its attention over the characters before it. "
        "The empty upper-right triangle is the causal mask: a character can never look at the future."
    )


def render_page() -> None:
    page_header("Model Information", "How MiniLM works, what it contains, and what it cannot do.")
    loaded, error = get_loaded_model()
    if loaded is not None:
        model, cfg, vocab, trained = loaded.model, loaded.model.config, loaded.tokenizer.vocab_size, True
    else:
        ref = untrained_reference()
        model, cfg, vocab, trained = ref["model"], ref["config"], ref["tokenizer"].vocab_size, False
    if error:
        callout("error", f"<b>The saved checkpoint could not be loaded:</b> {esc(error)}")

    n_params = model.count_parameters()
    cards_row(
        [
            metric_card("Trainable parameters", f"{n_params:,}", "trained weights" if trained else "untrained architecture", "blue"),
            metric_card("Vocabulary size", vocab, "characters", "violet"),
            metric_card("Blocks × heads", f"{cfg.n_layer} × {cfg.n_head}", f"embedding {cfg.n_embd}", "blue"),
            metric_card("Context length", cfg.block_size, "characters seen at once", "violet"),
        ]
    )

    section("The architecture in simple language")
    flow_diagram(
        [
            ("Input", "Characters", "“Mina” becomes M, i, n, a"),
            ("Step 1", "Token ids", "Each character gets a number from the tokenizer"),
            ("Step 2", "Embeddings", "Numbers become learnable vectors; a position vector says where each one sits"),
            ("Step 3", "Blocks", "Attention + feed-forward, repeated, with shortcuts and normalisation"),
            ("Output", "Next character", "A probability for every possible next character"),
        ]
    )
    st.write("")
    c = st.columns(3)
    cards = [
        ("Tokenizer", "Our tokenizer maps every distinct character in the training text to an id and back. Simple, and it can never meet an unknown word."),
        ("Embeddings", "A lookup table turns each id into a vector of numbers. A second table encodes the position, because attention alone does not know order."),
        ("Causal self-attention", "Each character scores all earlier characters and takes a weighted mix of what they know. A mask hides the future, so the model can only use the past."),
        ("Multiple heads", f"{cfg.n_head} heads attend in parallel, each free to learn a different pattern, such as the previous letter or the start of the word."),
        ("Feed-forward network", "A small two-layer network that processes each position on its own after attention has gathered the context."),
        ("Residuals + LayerNorm", "Shortcut connections add each block's input back to its output and normalisation keeps numbers stable, which makes deeper stacks trainable."),
    ]
    for i, (t, d) in enumerate(cards):
        with c[i % 3]:
            render(info_card(t, d))
            st.write("")

    section("How the model learns next-character prediction")
    st.markdown(
        """
1. Cut a window of text, for example `the cat sa`. The **target** is the same window shifted one character: `he cat sat`.
2. At every position the model outputs a probability for each possible next character.
3. **Cross-entropy loss** measures how much probability the model gave to the character that really came next. Guessing uniformly gives a loss of ln(vocabulary size).
4. **Backpropagation** works out how every weight contributed to the error.
5. **AdamW** changes every weight a tiny step in the direction that lowers the loss. Repeat thousands of times.
6. To write, the model predicts one character, appends it, and repeats. **Temperature** and **top-k** control how adventurous the sampling is.
"""
    )

    section("Where the parameters live")
    table(["Component", "Parameters"], parameter_rows(model) + [["Total (trainable)", f"{n_params:,}"]], num_cols=(1,))
    st.caption("The output layer shares its weights with the token embeddings (weight tying), so those weights are counted once.")

    if loaded is not None:
        _attention_explorer(loaded)
    else:
        callout("info", "Train the model to unlock the attention explorer, which visualises real attention weights.")

    section("Limitations - stated honestly")
    callout(
        "warn",
        "<b>This is not ChatGPT, and it is not trying to be.</b><br>"
        "• It is tiny: a few hundred thousand parameters, versus billions in modern assistants.<br>"
        "• It trains on a few thousand characters of text, so its knowledge is limited to that text.<br>"
        "• It is character-level: it must learn spelling from scratch and only sees " f"{cfg.block_size} characters of context.<br>"
        "• It does not understand meaning or answer questions; it continues text in the style it saw.<br>"
        "• With so little data it can memorise (overfit). Compare training and validation loss to see this.<br>"
        "• Generated text may contain misspellings, repeated phrases or nonsense.",
    )
