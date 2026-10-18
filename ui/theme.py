"""Dark-navy AI-research-lab theme: CSS plus small HTML helper components."""
from __future__ import annotations

import html as _html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #070b1a;
  --surface: #0e1530;
  --surface-2: #131c3d;
  --border: rgba(129, 140, 248, 0.22);
  --text: #e6e9f5;
  --muted: #94a3c8;
  --blue: #60a5fa;
  --violet: #a78bfa;
  --green: #34d399;
  --amber: #fbbf24;
  --red: #f87171;
}

html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }

.stApp {
  background:
    radial-gradient(900px 500px at 8% -10%, rgba(96, 165, 250, 0.16), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(139, 92, 246, 0.18), transparent 55%),
    var(--bg);
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1200px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1230 0%, #080d22 100%);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] [role="radiogroup"] { gap: 0.25rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 0.55rem 0.8rem; border-radius: 10px; border: 1px solid transparent;
  transition: all .15s ease; width: 100%;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(139, 92, 246, 0.12); border-color: var(--border);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(90deg, rgba(96,165,250,.20), rgba(139,92,246,.22));
  border-color: rgba(139, 92, 246, 0.55);
}

.brand { display: flex; align-items: center; gap: .7rem; margin: .2rem 0 1.1rem 0; }
.brand-logo {
  width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6); font-weight: 800; color: white; font-size: 1.15rem;
  box-shadow: 0 6px 22px rgba(99, 102, 241, .45);
}
.brand-name { font-weight: 800; font-size: 1.25rem; letter-spacing: -0.02em; }
.brand-sub { color: var(--muted); font-size: .72rem; margin-top: -2px; }

/* ---------- hero ---------- */
.hero {
  border: 1px solid var(--border); border-radius: 22px; padding: 2rem 2.2rem; margin-bottom: 1.4rem;
  background:
    radial-gradient(500px 220px at 90% 0%, rgba(139, 92, 246, .28), transparent 70%),
    radial-gradient(500px 240px at 0% 100%, rgba(59, 130, 246, .22), transparent 70%),
    linear-gradient(180deg, rgba(19, 28, 61, .95), rgba(11, 17, 40, .95));
}
.hero h1 {
  font-size: 3rem; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 .3rem 0; line-height: 1.05;
  background: linear-gradient(90deg, #93c5fd 0%, #c4b5fd 60%, #f0abfc 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: var(--muted); font-size: 1.05rem; margin: 0; max-width: 780px; line-height: 1.55; }
.hero .tagline { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: .5rem; }

.page-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 .2rem 0; }
.page-sub { color: var(--muted); margin: 0 0 1.4rem 0; }
.section-title { font-size: 1.15rem; font-weight: 700; margin: 1.6rem 0 .7rem 0; display: flex; align-items: center; gap: .5rem; }
.section-title::before { content: ""; width: 5px; height: 1.1rem; border-radius: 4px; background: linear-gradient(180deg, #60a5fa, #8b5cf6); }

/* ---------- cards ---------- */
.mini-card {
  border: 1px solid var(--border); border-radius: 16px; padding: 1.05rem 1.2rem; height: 100%;
  background: linear-gradient(180deg, rgba(19, 28, 61, .85), rgba(13, 20, 46, .85));
  transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
}
.mini-card:hover { transform: translateY(-2px); border-color: rgba(167, 139, 250, .55); box-shadow: 0 10px 30px rgba(76, 29, 149, .25); }
.mini-card h4 { margin: 0 0 .35rem 0; font-size: 1rem; font-weight: 700; }
.mini-card p { margin: 0; color: var(--muted); font-size: .92rem; line-height: 1.55; }
.metric .m-label { color: var(--muted); font-size: .74rem; text-transform: uppercase; letter-spacing: .09em; font-weight: 600; }
.metric .m-value { font-size: 1.75rem; font-weight: 800; letter-spacing: -0.02em; margin-top: .2rem; }
.metric .m-sub { color: var(--muted); font-size: .8rem; margin-top: .1rem; min-height: 1.1rem; }
.metric.accent-blue .m-value { color: #93c5fd; }
.metric.accent-violet .m-value { color: #c4b5fd; }
.metric.accent-green .m-value { color: #6ee7b7; }
.metric.accent-amber .m-value { color: #fcd34d; }
.metric.accent-red .m-value { color: #fca5a5; }

/* ---------- badges / chips ---------- */
.badge { display: inline-flex; align-items: center; gap: .4rem; padding: .28rem .8rem; border-radius: 999px; font-size: .8rem; font-weight: 600; border: 1px solid; }
.badge .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.badge.ok { color: var(--green); border-color: rgba(52, 211, 153, .45); background: rgba(52, 211, 153, .10); }
.badge.warn { color: var(--amber); border-color: rgba(251, 191, 36, .45); background: rgba(251, 191, 36, .10); }
.badge.running { color: var(--blue); border-color: rgba(96, 165, 250, .5); background: rgba(96, 165, 250, .12); }
.badge.running .dot { animation: pulse 1.2s ease-in-out infinite; }
.badge.idle { color: var(--muted); border-color: rgba(148, 163, 200, .4); background: rgba(148, 163, 200, .08); }
.badge.error { color: var(--red); border-color: rgba(248, 113, 113, .5); background: rgba(248, 113, 113, .10); }
@keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: .35; transform: scale(1.5); } }

.chip { display: inline-block; padding: .3rem .75rem; margin: 0 .35rem .35rem 0; border-radius: 10px; font-size: .82rem;
  background: rgba(96, 165, 250, .10); border: 1px solid rgba(96, 165, 250, .3); color: #bfdbfe; font-family: 'JetBrains Mono', monospace; }
.chip.violet { background: rgba(139, 92, 246, .12); border-color: rgba(167, 139, 250, .4); color: #ddd6fe; }

/* ---------- callouts ---------- */
.callout { border-radius: 14px; padding: .9rem 1.1rem; margin: .6rem 0 1rem 0; border: 1px solid; font-size: .93rem; line-height: 1.55; }
.callout b { color: inherit; }
.callout.info { background: rgba(96, 165, 250, .08); border-color: rgba(96, 165, 250, .35); color: #cfe3ff; }
.callout.warn { background: rgba(251, 191, 36, .08); border-color: rgba(251, 191, 36, .38); color: #fde9a8; }
.callout.ok { background: rgba(52, 211, 153, .08); border-color: rgba(52, 211, 153, .38); color: #b8f3dc; }
.callout.error { background: rgba(248, 113, 113, .09); border-color: rgba(248, 113, 113, .42); color: #fecaca; }

/* ---------- architecture flow ---------- */
.flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: .5rem; }
.flow .step { flex: 1 1 150px; border: 1px solid var(--border); border-radius: 14px; padding: .75rem .9rem;
  background: linear-gradient(180deg, rgba(19, 28, 61, .9), rgba(12, 18, 42, .9)); }
.flow .step .n { color: var(--violet); font-size: .7rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
.flow .step .t { font-weight: 700; font-size: .95rem; margin: .1rem 0 .15rem 0; }
.flow .step .d { color: var(--muted); font-size: .8rem; line-height: 1.4; }
.flow .arrow { align-self: center; color: var(--violet); font-weight: 700; }

/* ---------- generated text ---------- */
.gen-box { border: 1px solid rgba(139, 92, 246, .45); border-radius: 16px; padding: 1.2rem 1.4rem;
  background: linear-gradient(180deg, rgba(14, 21, 48, .95), rgba(9, 14, 34, .95)); font-family: 'JetBrains Mono', monospace;
  font-size: .95rem; line-height: 1.75; white-space: pre-wrap; word-break: break-word; }
.gen-box .prompt { color: #8ea2d6; }
.gen-box .cont { color: #e9e4ff; background: rgba(139, 92, 246, .16); border-radius: 4px; padding: 0 2px; }
.mono-label { color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; font-weight: 600; margin: .6rem 0 .25rem 0; }

/* ---------- widgets ---------- */
.stButton > button {
  border-radius: 12px; border: 1px solid rgba(139, 92, 246, .5); font-weight: 600; padding: .55rem 1.3rem;
  background: rgba(139, 92, 246, .12); color: var(--text); transition: all .15s ease;
}
.stButton > button:hover { border-color: #a78bfa; background: rgba(139, 92, 246, .25); color: white; }
.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, #3b82f6, #8b5cf6); border: none; color: white;
  box-shadow: 0 8px 24px rgba(99, 102, 241, .35);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.12); box-shadow: 0 10px 30px rgba(99, 102, 241, .55); }
.stButton > button:disabled { opacity: .45; box-shadow: none; }

.stTextArea textarea, .stTextInput input, .stNumberInput input {
  background: #0b1230 !important; border: 1px solid var(--border) !important; border-radius: 12px !important;
  font-family: 'JetBrains Mono', monospace;
}
div[data-baseweb="select"] > div { background: #0b1230; border-color: var(--border); border-radius: 12px; }
[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 14px; background: rgba(14, 21, 48, .6); }
[data-testid="stFileUploader"] section { background: #0b1230; border: 1px dashed rgba(139, 92, 246, .5); border-radius: 14px; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #3b82f6, #8b5cf6); }

table.nice { width: 100%; border-collapse: collapse; font-size: .92rem; }
table.nice th { text-align: left; color: var(--muted); font-weight: 600; font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; padding: .5rem .7rem; border-bottom: 1px solid var(--border); }
table.nice td { padding: .5rem .7rem; border-bottom: 1px solid rgba(129, 140, 248, .10); }
table.nice td.num { text-align: right; font-family: 'JetBrains Mono', monospace; color: #c7d2fe; }

@media (max-width: 760px) {
  .hero { padding: 1.4rem 1.2rem; }
  .hero h1 { font-size: 2.2rem; }
  .flow .arrow { display: none; }
}
</style>
"""


def esc(value) -> str:
    return _html.escape(str(value))


def clean(markup: str) -> str:
    """Strip indentation and blank lines so Markdown never mistakes HTML for a code block."""
    return "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())


def render(markup: str) -> None:
    st.markdown(clean(markup), unsafe_allow_html=True)


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Components - each returns an HTML string (use render() to show it)
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str) -> None:
    render(f'<div class="page-title">{esc(title)}</div><div class="page-sub">{esc(subtitle)}</div>')


def section(title: str) -> None:
    render(f'<div class="section-title">{esc(title)}</div>')


def badge(label: str, kind: str = "idle") -> str:
    return f'<span class="badge {kind}"><span class="dot"></span>{esc(label)}</span>'


def chip(text: str, violet: bool = False) -> str:
    return f'<span class="chip{" violet" if violet else ""}">{esc(text)}</span>'


def metric_card(label: str, value, sub: str = "", accent: str = "blue") -> str:
    return (
        f'<div class="mini-card metric accent-{accent}"><div class="m-label">{esc(label)}</div>'
        f'<div class="m-value">{esc(value)}</div><div class="m-sub">{esc(sub)}</div></div>'
    )


def info_card(title: str, body: str) -> str:
    return f'<div class="mini-card"><h4>{esc(title)}</h4><p>{esc(body)}</p></div>'


def callout(kind: str, text_html: str) -> None:
    """kind: info | warn | ok | error.  text_html may contain <b> tags - escape user data first."""
    render(f'<div class="callout {kind}">{text_html}</div>')


def cards_row(cards: list[str]) -> None:
    cols = st.columns(len(cards))
    for col, markup in zip(cols, cards):
        with col:
            render(markup)


def flow_diagram(steps: list[tuple[str, str, str]]) -> None:
    """steps = [(number, title, description), ...]"""
    parts = []
    for i, (n, title, desc) in enumerate(steps):
        parts.append(
            f'<div class="step"><div class="n">{esc(n)}</div><div class="t">{esc(title)}</div><div class="d">{esc(desc)}</div></div>'
        )
        if i < len(steps) - 1:
            parts.append('<div class="arrow">&rarr;</div>')
    render(f'<div class="flow">{"".join(parts)}</div>')


def table(headers: list[str], rows: list[list], num_cols: tuple[int, ...] = ()) -> None:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(
            f'<td class="{"num" if i in num_cols else ""}">{esc(c)}</td>' for i, c in enumerate(row)
        )
        body += f"<tr>{cells}</tr>"
    render(f'<table class="nice"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


def generated_box(prompt: str, continuation: str) -> None:
    p = esc(prompt).replace("\n", "<br>")
    c = esc(continuation).replace("\n", "<br>")
    render(f'<div class="gen-box"><span class="prompt">{p}</span><span class="cont">{c}</span></div>')
