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