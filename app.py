# ------------
# Nassau Candy Distributor — Product Line Profitability & Margin Performance Dashboard Built with Streamlit + Plotly
# ------------

#-----------------
# IMPORTING LIBRARIES
#-----------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

#-----------------
# IMPORTING DATA
#-----------------
df = pd.read_csv("data/Nassau Candy Distributor.csv")

# -----------------
# PAGE CONFIGURATION
# ----------------- 
st.set_page_config(
    page_title="Nassau Candy Distributor — Product Line Profitability & Margin Performance Dashboard",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------
# THEME AND CUSTOMIZATION
# ----------------- 

PALETTE = {
    "bg_dark":    "#0D1117",
    "bg_card":    "#161B22",
    "bg_sidebar": "#13181F",
    "accent1":    "#F97316",   
    "accent2":    "#A855F7",   
    "accent3":    "#22D3EE",  
    "accent4":    "#4ADE80",   
    "accent5":    "#FB7185",   
    "text_main":  "#E6EDF3",
    "text_muted": "#8B949E",
    "border":     "#30363D",
}
   
DIVISION_COLORS = {
    "Chocolate": "#F97316",
    "Sugar":     "#A855F7",
    "Other":     "#22D3EE",
}

REGION_COLORS = {
    "Atlantic": "#4ADE80",
    "Interior": "#F97316",
    "Pacific":  "#22D3EE",
    "Gulf":     "#FB7185",
}

st.markdown(
    f"""
    <style>
    /* ----Base Styles------- */
    html, body, [class*="css"]  {{
        font-family: 'Arial', sans-serif;
        background-color: {PALETTE['bg_dark']};
        color: {PALETTE['text_main']};
    }}
    .stApp {{
        background-color: {PALETTE['bg_dark']};
    }}

   /* ── Base ── */
  html, body, [class*="css"] {{
      font-family: 'Inter', 'Segoe UI', sans-serif;
      background-color: {PALETTE['bg_dark']};
      color: {PALETTE['text_main']};
  }}
  .stApp {{ background-color: {PALETTE['bg_dark']}; }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, {PALETTE['bg_sidebar']} 0%, #0A0E14 100%);
      border-right: 1px solid {PALETTE['border']};
  }}
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {{
      color: {PALETTE['accent1']};
  }}

  /* ── Metric Cards ── */
  .kpi-card {{
      background: linear-gradient(135deg, {PALETTE['bg_card']} 0%, #1C2433 100%);
      border: 1px solid {PALETTE['border']};
      border-radius: 16px;
      padding: 24px 20px;
      text-align: center;
      box-shadow: 0 4px 24px rgba(0,0,0,0.4);
      transition: transform 0.2s;
      height: 120px;
      display: flex;
      flex-direction: column;
      justify-content: center;
  }}
  .kpi-card:hover {{ transform: translateY(-3px); }}
  .kpi-label {{
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {PALETTE['text_muted']};
      margin-bottom: 8px;
  }}
  .kpi-value {{
      font-size: 1.6rem;
      font-weight: 800;
      line-height: 1;
      background: linear-gradient(90deg, {PALETTE['accent1']}, {PALETTE['accent2']});
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
  }}
  .kpi-delta {{
      font-size: 0.82rem;
      color: {PALETTE['accent4']};
      margin-top: 6px;
  }}

  /* ── Section Headers ── */
  .section-header {{
      font-size: 1.35rem;
      font-weight: 700;
      color: {PALETTE['text_main']};
      border-left: 4px solid {PALETTE['accent1']};
      padding-left: 12px;
      margin: 28px 0 16px 0;
  }}

  /* ── Chart Cards ── */
  .chart-card {{
      background: {PALETTE['bg_card']};
      border: 1px solid {PALETTE['border']};
      border-radius: 14px;
      padding: 8px;
      box-shadow: 0 2px 16px rgba(0,0,0,0.3);
  }}

  /* ── Divider ── */
  hr {{ border-color: {PALETTE['border']}; }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
      background: {PALETTE['bg_card']};
      border-radius: 10px;
      padding: 4px;
      gap: 12px;
      display: flex;
      justify-content: space-between;
  }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 8px;
      color: {PALETTE['text_muted']};
      font-weight: 600;
      font-size: 0.88rem;
      flex: 1;
      text-align: center;
      justify-content: center;
  }}
  .stTabs [aria-selected="true"] {{
      background: linear-gradient(90deg, {PALETTE['accent1']}, {PALETTE['accent2']}) !important;
      color: white !important;
  }}

  /* ── Sliders & Inputs ── */
  .stSlider > div > div > div > div {{ background: {PALETTE['accent1']}; }}
  .stMultiSelect [data-baseweb="tag"] {{
      background: {PALETTE['accent2']} !important;
  }}

  [data-baseweb="select"] [data-baseweb="popover"] li {{
      white-space: normal !important;
      word-break: break-word !important;
      line-height: 1.4 !important;
      padding: 8px 12px !important;
      min-width: 300px !important;
      max-width: 340px !important;
  }}
  [data-baseweb="select"] [data-baseweb="popover"] ul {{
      min-width: 300px !important;
      max-width: 340px !important;
      width: 340px !important;
  }}
  [data-baseweb="select"] > div:first-child {{
      white-space: normal !important;
      word-wrap: break-word !important;
      height: auto !important;
      min-height: 38px !important;
      width: 100% !important;
  }}
  [data-baseweb="select"] input {{
      caret-color: transparent !important;
  }}



  /* ── Scrollbar ── */

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: {PALETTE['bg_dark']}; }}
  ::-webkit-scrollbar-thumb {{ background: {PALETTE['border']}; border-radius: 3px; }}

  /* ── Table ── */
  .dataframe {{ font-size: 0.82rem; }}
  .stDataFrame {{ border-radius: 12px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)

