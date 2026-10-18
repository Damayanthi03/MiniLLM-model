"""MiniLM web app - run with:  streamlit run app.py"""
import streamlit as st

# set_page_config must be the first Streamlit call.
st.set_page_config(
    page_title="MiniLM · Transformer language model from scratch",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.theme import badge, inject_css, render  # noqa: E402
from views import page_dashboard, page_evaluation, page_generate, page_info, page_training  # noqa: E402
from views.common import get_status  # noqa: E402
from config import get_device  # noqa: E402

PAGES = {
    "Dashboard": page_dashboard.render_page,
    "Text Generation": page_generate.render_page,
    "Model Training": page_training.render_page,
    "Model Information": page_info.render_page,
    "Evaluation": page_evaluation.render_page,
}


def main() -> None:
    inject_css()
    st.session_state.setdefault("nav", "Dashboard")

    with st.sidebar:
        render(
            '<div class="brand"><div class="brand-logo">M</div><div>'
            '<div class="brand-name">MiniLM</div><div class="brand-sub">Transformer LM from scratch</div></div></div>'
        )
        st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")
        st.write("")
        status = get_status()
        render(f'<div class="mono-label">Model status</div>{badge(status["label"], status["kind"])}')
        st.caption(f"Device: {get_device()}")
        st.caption("No pretrained weights · no external AI APIs")

    try:
        PAGES[st.session_state["nav"]]()
    except Exception as exc:  # last-resort guard so the demo never shows a raw stack trace
        st.error(f"Something went wrong while rendering this page: {exc}")
        with st.expander("Technical details"):
            st.exception(exc)


main()
