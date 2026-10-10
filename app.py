"""TIX AI shopping assistant floating over a simulated storefront.
Run: streamlit run app.py
Widget placement is controlled by ASSISTANT_SIDE=right|left (default right).
The agent and chat behavior are unchanged; visual styling lives in ui/styles.py.
"""
import logging

import streamlit as st
import streamlit.components.v1 as components

from agent.agent import SalesAgent
from ui import styles
from ui.chat import render_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

st.set_page_config(page_title="TIX", page_icon="🛒", layout="wide")


@st.cache_resource(show_spinner=False)
def get_agent() -> SalesAgent:
    """The agent is stateless; conversation state lives in each browser session."""
    agent = SalesAgent()
    agent.warmup()
    return agent


st.markdown(styles.CSS, unsafe_allow_html=True)
st.markdown(styles.STORE_HTML, unsafe_allow_html=True)
st.markdown(styles.FAB_HTML, unsafe_allow_html=True)
render_chat(get_agent())
components.html(styles.WIDGET_JS, height=0)