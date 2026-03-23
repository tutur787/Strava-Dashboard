"""
ui/styles.py — inject_css() to apply global Streamlit styles.
"""
import streamlit as st


def inject_css() -> None:
    st.markdown("""
<style>
/* Hide Streamlit branding only */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Tighter content padding — top increased so the title isn't clipped */
.block-container {padding-top: 2rem; padding-bottom: 1rem; max-width: 1440px;}

/* Spacing between top-level page elements (title → banner → tabs).
   Covers both selector forms across Streamlit versions. */
.block-container > [data-testid="stVerticalBlock"],
.block-container > div > [data-testid="stVerticalBlock"] {gap: 1.5rem !important;}

/* Title sizing and spacing */
[data-testid="stHeadingWithActionElements"] {padding-bottom: 0.15rem !important; margin-bottom: 0 !important;}
[data-testid="stHeadingWithActionElements"] h1 {
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 14px 18px;
}

/* Tab strip */
.stTabs [data-baseweb="tab-list"] {gap: 2px; border-bottom: 2px solid rgba(255,255,255,0.08);}
.stTabs [data-baseweb="tab"] {border-radius: 8px 8px 0 0; padding: 8px 18px; font-weight: 500;}
.stTabs [aria-selected="true"] {background: rgba(255,255,255,0.07);}

/* Dividers */
hr {border-color: rgba(255,255,255,0.07) !important;}

/* Section headings */
h3 {margin-top: 0.25rem !important;}

/* Cookie-controller (and any other) zero-height iframes inject invisible
   element-containers that still consume the parent flex gap.
   Taking them out of normal flow with position:absolute removes the gap slot
   while keeping the iframe in the DOM so its JS can still run. */
.element-container:has(iframe[height="0"]) {
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    pointer-events: none !important;
}


</style>
""", unsafe_allow_html=True)
