"""
CE Africa Valuation Platform
============================
Main application entry point with redesigned Hub & Spoke navigation.

Sections:
1. Command Center - Dashboard home
2. Setup - Guided configuration wizard
3. AI Assumptions - Derive assumptions from historical data (REQUIRED before Forecast)
4. Forecast - Run and view results
5. Funding - Financing analysis
6. Manufacturing Strategy - Make vs Buy analysis (Sprint 13)
7. AI Trend Analysis - Reporting and insights
8. Compare - Scenario comparison
9. Access Control - User permissions

Version: 2.3 (December 14, 2025)
- Added AI Assumptions Engine (required step before Forecast)
- Probability distribution fitting for MC Simulation
- Manufacturing assumptions derived from AI analysis
- Save Assumptions workflow
- Updated Manufacturing Strategy v2 with part-level analysis

Previous (v2.2):
- Added AI Trend Analysis section (Sprint 14)
- Automated trend detection, anomaly identification, seasonality analysis
- AI-generated insights and recommendations

Previous (v2.1):
- Added Vertical Integration / Manufacturing Strategy section (Sprint 13)
- Removed balloons on snapshot save (replaced with info message)
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import os
import base64

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Crusher Equipment Africa - Valuation Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LINEAR DESIGN SYSTEM THEME (Industrial / Monochrome)
# =============================================================================

LINEAR_THEME_CSS = """
<style>
/* ==========================================================================
   CRUSHER EQUIPMENT AFRICA - INDUSTRIAL DESIGN SYSTEM
   Clean, monochrome, industrial aesthetic
   ========================================================================== */

/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* CSS Variables */
:root {
    /* Backgrounds (near-black base) */
    --bg-base: #09090B;
    --bg-elevated: #0F0F11;
    --bg-surface: #18181B;
    --bg-hover: #1F1F23;
    --bg-active: #27272A;
    
    /* Borders */
    --border-subtle: #27272A;
    --border-default: #3F3F46;
    --border-strong: #52525B;
    
    /* Text */
    --text-primary: #FAFAFA;
    --text-secondary: #A1A1AA;
    --text-tertiary: #71717A;
    
    /* CE Africa Gold (used sparingly) */
    --accent: #D4A537;
    --accent-hover: #E5B84A;
    --accent-muted: rgba(212, 165, 55, 0.15);
    --accent-subtle: rgba(212, 165, 55, 0.08);
    
    /* Status */
    --success: #22C55E;
    --success-muted: rgba(34, 197, 94, 0.15);
    --warning: #F59E0B;
    --warning-muted: rgba(245, 158, 11, 0.15);
    --error: #EF4444;
    --error-muted: rgba(239, 68, 68, 0.15);
    --info: #3B82F6;
    --info-muted: rgba(59, 130, 246, 0.15);
}

/* Global button text color fix - ensure ALL buttons have readable text */
/* This must come BEFORE other button styles to set default */
.stButton > button,
button[class*="stButton"],
div[data-testid="stButton"] > button,
button:not([kind="primary"]) {
    color: var(--text-primary) !important;
}

/* Force light text on all non-primary buttons and their children */
.stButton > button:not([kind="primary"]) p,
.stButton > button:not([kind="primary"]) span,
.stButton > button:not([kind="primary"]) div,
.stButton > button:not([kind="primary"]) label,
button:not([kind="primary"]) p,
button:not([kind="primary"]) span,
button:not([kind="primary"]) div,
button:not([kind="primary"]) label {
    color: var(--text-primary) !important;
}

/* Base styles */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-secondary) !important;
}

.stApp {
    background-color: var(--bg-base) !important;
}

[data-testid="stAppViewContainer"] > .main {
    background-color: var(--bg-base) !important;
}

.block-container {
    padding: 1rem 3rem 2rem 3rem !important;
    max-width: 1400px !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

h1, .stMarkdown h1 {
    font-size: 1.875rem !important;
    font-weight: 700 !important;
}

h2, .stMarkdown h2 {
    font-size: 1.5rem !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding-bottom: 0.5rem !important;
    margin-top: 2rem !important;
}

h3, .stMarkdown h3 {
    font-size: 1.125rem !important;
}

p, .stMarkdown p {
    color: var(--text-secondary) !important;
    line-height: 1.6 !important;
    font-size: 0.9375rem !important;
}

a {
    color: var(--accent) !important;
    text-decoration: none !important;
}

a:hover {
    color: var(--accent-hover) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-elevated) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: var(--bg-elevated) !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    border-bottom: none !important;
    padding-bottom: 0 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
}

/* Buttons - Base styles for ALL buttons */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    border-radius: 4px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    letter-spacing: 0.02em !important;
}

/* CRITICAL: Default button text color - must be light on dark background */
/* Apply to all buttons that are NOT primary (primary has gold bg with black text) */
.stButton > button:not([kind="primary"]),
.stButton > button:not([data-testid="baseButton-primary"]),
button:not([kind="primary"]) {
    color: #FAFAFA !important;
}

/* Force light text on all inner elements of non-primary buttons */
.stButton > button:not([kind="primary"]) *,
.stButton > button:not([data-testid="baseButton-primary"]) *,
button:not([kind="primary"]) * {
    color: #FAFAFA !important;
}

/* Primary button - all possible selectors */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
.stButton > button[data-testid="stBaseButton-primary"],
button[kind="primary"],
button[data-testid="baseButton-primary"],
button[data-testid="stBaseButton-primary"],
.stButton button[type="primary"],
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #D4A537 !important;
    background: #D4A537 !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
}

/* PRIMARY BUTTON TEXT - Target inner elements */
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div,
.stButton > button[data-testid="baseButton-primary"] p,
.stButton > button[data-testid="baseButton-primary"] span,
.stButton > button[data-testid="baseButton-primary"] div,
.stButton > button[data-testid="stBaseButton-primary"] p,
.stButton > button[data-testid="stBaseButton-primary"] span,
.stButton > button[data-testid="stBaseButton-primary"] div,
button[kind="primary"] p,
button[kind="primary"] span,
button[kind="primary"] div {
    color: #000000 !important;
}

/* Even more aggressive - any button with gold background */
.stButton button[style*="background"],
button[style*="rgb(212"],
button[style*="D4A537"] {
    color: #000000 !important;
}

.stButton button[style*="background"] p,
.stButton button[style*="background"] span,
button[style*="rgb(212"] p,
button[style*="rgb(212"] span {
    color: #000000 !important;
}

/* Primary button hover */
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover,
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background-color: #E5B847 !important;
    background: #E5B847 !important;
    color: #000000 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(212, 165, 55, 0.25) !important;
}

.stButton > button[kind="primary"]:hover p,
.stButton > button[kind="primary"]:hover span,
button[kind="primary"]:hover p,
button[kind="primary"]:hover span {
    color: #000000 !important;
}

/* Force black text on any gold/yellow background button */
.stButton > button[style*="background-color: rgb(212, 165, 55)"],
.stButton > button[style*="background: rgb(212, 165, 55)"],
button[style*="background-color: rgb(212, 165, 55)"],
button[style*="background: rgb(212, 165, 55)"] {
    color: #000000 !important;
}

/* Buttons - Secondary */
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"],
.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
    background-color: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-default) !important;
}

/* Secondary button text - ensure all inner elements are light colored */
.stButton > button[kind="secondary"] p,
.stButton > button[kind="secondary"] span,
.stButton > button[kind="secondary"] div,
.stButton > button[data-testid="baseButton-secondary"] p,
.stButton > button[data-testid="baseButton-secondary"] span,
.stButton > button[data-testid="baseButton-secondary"] div,
.stButton > button:not([kind="primary"]) p,
.stButton > button:not([kind="primary"]) span,
.stButton > button:not([kind="primary"]) div {
    color: var(--text-primary) !important;
}

.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
    background-color: var(--bg-hover) !important;
    border-color: var(--border-strong) !important;
    color: var(--text-primary) !important;
}

.stButton > button[kind="secondary"]:hover p,
.stButton > button[kind="secondary"]:hover span,
.stButton > button[kind="secondary"]:hover div,
.stButton > button:not([kind="primary"]):hover p,
.stButton > button:not([kind="primary"]):hover span,
.stButton > button:not([kind="primary"]):hover div {
    color: var(--text-primary) !important;
}

/* Navigation buttons - clean text style */
.nav-button > button {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    border-radius: 0 !important;
    text-align: left !important;
    padding: 0.625rem 1rem !important;
    font-weight: 400 !important;
}

.nav-button > button:hover {
    background-color: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    border-left-color: var(--border-default) !important;
}

.nav-button-active > button {
    background-color: var(--bg-active) !important;
    color: var(--text-primary) !important;
    border-left: 2px solid var(--accent) !important;
    font-weight: 500 !important;
}

/* Form Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9375rem !important;
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    padding: 0.625rem 0.875rem !important;
    transition: all 0.15s ease !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-subtle) !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: var(--text-tertiary) !important;
}

/* Labels */
.stTextInput > label,
.stNumberInput > label,
.stTextArea > label,
.stSelectbox > label,
.stMultiSelect > label,
.stSlider > label,
.stCheckbox > label,
.stRadio > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 0.375rem !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
}

.stSelectbox > div > div:hover {
    border-color: var(--border-default) !important;
}

/* Dropdown menu */
[data-baseweb="menu"] {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 4px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
}

[data-baseweb="menu"] li {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
}

[data-baseweb="menu"] li:hover {
    background-color: var(--bg-hover) !important;
    color: var(--text-primary) !important;
}

[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: var(--accent-muted) !important;
    color: var(--accent) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--text-tertiary) !important;
    background-color: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1.25rem !important;
    margin-bottom: -1px !important;
    transition: all 0.15s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom-color: var(--accent) !important;
    background-color: transparent !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
    padding: 1rem 1.25rem !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.6875rem !important;
    font-weight: 600 !important;
    color: var(--text-tertiary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

/* DataFrames */
.stDataFrame {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}

.stDataFrame [data-testid="stDataFrameResizable"] {
    background-color: var(--bg-elevated) !important;
}

.stDataFrame th {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.6875rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text-tertiary) !important;
    background-color: var(--bg-surface) !important;
    border-bottom: 1px solid var(--border-default) !important;
    padding: 0.75rem 1rem !important;
}

.stDataFrame td {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    color: var(--text-secondary) !important;
    background-color: var(--bg-elevated) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding: 0.625rem 1rem !important;
}

.stDataFrame tr:hover td {
    background-color: var(--bg-hover) !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--text-primary) !important;
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.15s ease !important;
}

.streamlit-expanderHeader:hover {
    background-color: var(--bg-hover) !important;
    border-color: var(--border-default) !important;
}

.streamlit-expanderContent {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
    border-radius: 0 0 4px 4px !important;
    padding: 1rem !important;
}

/* Alerts */
.stAlert {
    font-family: 'Inter', sans-serif !important;
    border-radius: 4px !important;
    padding: 0.875rem 1rem !important;
    border: 1px solid !important;
}

div[data-testid="stInfo"] {
    background-color: var(--info-muted) !important;
    border-color: var(--info) !important;
    color: var(--text-primary) !important;
}

div[data-testid="stSuccess"] {
    background-color: var(--success-muted) !important;
    border-color: var(--success) !important;
    color: var(--text-primary) !important;
}

div[data-testid="stWarning"] {
    background-color: var(--warning-muted) !important;
    border-color: var(--warning) !important;
    color: var(--text-primary) !important;
}

div[data-testid="stError"] {
    background-color: var(--error-muted) !important;
    border-color: var(--error) !important;
    color: var(--text-primary) !important;
}

/* Progress bar */
.stProgress > div > div {
    background-color: var(--border-default) !important;
    border-radius: 0 !important;
    height: 4px !important;
}

.stProgress > div > div > div {
    background-color: var(--accent) !important;
    border-radius: 0 !important;
}

/* Dividers */
hr, .stMarkdown hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 1.5rem 0 !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px !important;
    height: 8px !important;
}

::-webkit-scrollbar-track {
    background: var(--bg-base) !important;
}

::-webkit-scrollbar-thumb {
    background: var(--border-default) !important;
    border-radius: 0 !important;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--border-strong) !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
"""

st.markdown(LINEAR_THEME_CSS, unsafe_allow_html=True)

# =============================================================================
# IMPORTS
# =============================================================================

# Database connector
from db_connector import SupabaseHandler
from supabase_utils import get_user_id

# UI Components
try:
    from components.ui_components import inject_custom_css, section_header
    UI_COMPONENTS_AVAILABLE = True
except ImportError:
    UI_COMPONENTS_AVAILABLE = False
    def inject_custom_css():
        pass
    def section_header(title, subtitle=None):
        st.subheader(title)
        if subtitle:
            st.caption(subtitle)

# New components
try:
    from components.command_center import render_command_center
    COMMAND_CENTER_AVAILABLE = True
except ImportError:
    COMMAND_CENTER_AVAILABLE = False

try:
    from components.setup_wizard import render_setup_wizard
    SETUP_WIZARD_AVAILABLE = True
except ImportError:
    SETUP_WIZARD_AVAILABLE = False

try:
    from components.forecast_section import render_forecast_section
    FORECAST_SECTION_AVAILABLE = True
except ImportError:
    FORECAST_SECTION_AVAILABLE = False

try:
    from components.scenario_comparison import render_scenario_comparison
    SCENARIO_COMPARISON_AVAILABLE = True
except ImportError:
    SCENARIO_COMPARISON_AVAILABLE = False

# AI Assumptions Integration Component
try:
    from components.ai_assumptions_integration import (
        render_ai_assumptions_summary,
        render_ai_assumption_badge,
        get_assumption_with_source
    )
    AI_INTEGRATION_AVAILABLE = True
except ImportError:
    AI_INTEGRATION_AVAILABLE = False

# Enhanced Navigation Component
try:
    from components.enhanced_navigation import (
        render_enhanced_sidebar_nav,
        render_breadcrumb_nav,
        render_tab_navigation,
        render_progress_indicator,
        render_quick_actions
    )
    ENHANCED_NAV_AVAILABLE = True
except ImportError:
    ENHANCED_NAV_AVAILABLE = False

try:
    from components.user_management import render_user_management
    USER_MANAGEMENT_AVAILABLE = True
except ImportError:
    USER_MANAGEMENT_AVAILABLE = False

# Sprint 13: Vertical Integration / Manufacturing Strategy
try:
    from components.vertical_integration import render_vertical_integration_section
    VERTICAL_INTEGRATION_AVAILABLE = True
except ImportError:
    VERTICAL_INTEGRATION_AVAILABLE = False

# Sprint 14: AI Trend Analysis (Reporting)
try:
    from components.ai_trend_analysis import render_trend_analysis_section
    AI_TREND_ANALYSIS_AVAILABLE = True
except ImportError:
    AI_TREND_ANALYSIS_AVAILABLE = False

# Sprint 14 Enhancement: AI Assumptions Engine (Required before Forecast)
try:
    from components.ai_assumptions_engine import (
        render_ai_assumptions_section,
        get_saved_assumptions,
        get_assumption_value,
        get_assumption_distribution,
        get_manufacturing_assumptions,
        sample_from_assumptions
    )
    AI_ASSUMPTIONS_AVAILABLE = True
except ImportError:
    AI_ASSUMPTIONS_AVAILABLE = False
    
    # Stub functions if not available
    def get_saved_assumptions(db, scenario_id):
        return None
    
    def get_assumption_value(assumptions_set, assumption_id, default=0.0):
        return default

# Fallback to legacy components
try:
    from components.forecast_viewer import render_forecast_viewer
    LEGACY_FORECAST_AVAILABLE = True
except ImportError:
    LEGACY_FORECAST_AVAILABLE = False

try:
    from components.dashboard import render_dashboard
    LEGACY_DASHBOARD_AVAILABLE = True
except ImportError:
    LEGACY_DASHBOARD_AVAILABLE = False

# Sprint 11: Funding Engine
try:
    from components.funding_ui import render_funding_section
    FUNDING_AVAILABLE = True
except ImportError:
    FUNDING_AVAILABLE = False

# Sprint 17: What-If Agent
try:
    from components.whatif_agent import render_whatif_agent
    WHATIF_AGENT_AVAILABLE = True
except ImportError:
    WHATIF_AGENT_AVAILABLE = False


# =============================================================================
# BANNER IMAGE HELPER
# =============================================================================

def get_banner_image_base64():
    """
    Returns the base64-encoded banner image.
    Place 'banner.png' in your project root directory.
    """
    banner_path = "banner.png"
    if os.path.exists(banner_path):
        with open(banner_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def render_banner():
    """Render the Crusher Equipment Africa banner."""
    banner_b64 = get_banner_image_base64()
    
    if banner_b64:
        # Use actual image
        st.markdown(f"""
        <div style="margin: -1rem -3rem 1.5rem -3rem; overflow: hidden;">
            <img src="data:image/png;base64,{banner_b64}" 
                 style="width: 100%; height: 120px; object-fit: cover; object-position: center;" 
                 alt="Crusher Equipment Africa">
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback: text-based header matching website style
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
            margin: -1rem -3rem 1.5rem -3rem;
            padding: 2rem 3rem;
            border-bottom: 1px solid #27272A;
            text-align: center;
        ">
            <div style="display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 0.5rem;">
                <div style="flex: 0 0 60px; height: 1px; background: linear-gradient(90deg, transparent, #FAFAFA);"></div>
                <h1 style="
                    color: #FAFAFA;
                    font-size: 1.5rem;
                    font-weight: 700;
                    letter-spacing: 0.15em;
                    margin: 0;
                    font-family: 'Inter', sans-serif;
                ">CRUSHER EQUIPMENT AFRICA</h1>
                <div style="flex: 0 0 60px; height: 1px; background: linear-gradient(90deg, #FAFAFA, transparent);"></div>
            </div>
            <p style="
                color: #A1A1AA;
                font-size: 0.875rem;
                font-style: italic;
                margin: 0;
                letter-spacing: 0.05em;
            ">Empowering Mining Excellence</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'scenario_id' not in st.session_state:
        st.session_state.scenario_id = None
    if 'db_handler' not in st.session_state:
        st.session_state.db_handler = SupabaseHandler()
    if 'current_section' not in st.session_state:
        st.session_state.current_section = 'home'
    if 'setup_step' not in st.session_state:
        st.session_state.setup_step = 'basics'
    if 'navigate_to' not in st.session_state:
        st.session_state.navigate_to = None
    if 'navigate_step' not in st.session_state:
        st.session_state.navigate_step = None
    # NEW: Workflow state management
    if 'workflow_progress' not in st.session_state:
        st.session_state.workflow_progress = {}
    if 'current_workflow_stage' not in st.session_state:
        st.session_state.current_workflow_stage = None


# =============================================================================
# SCENARIO MANAGEMENT
# =============================================================================

# Tables to copy when duplicating imports into a new scenario
IMPORT_CLONE_SCENARIO_TABLES = [
    "historic_financials",
    "historic_customer_revenue",
    "historic_expense_categories",
    "historical_balance_sheet",
    "historical_cashflow",
    "historical_trial_balance",
    "historical_income_statement_line_items",
    "historical_balance_sheet_line_items",
    "historical_cashflow_line_items",
    "expense_assumptions",
]

IMPORT_CLONE_USER_SCENARIO_TABLES = [
    "installed_base",
    "creditors",
    "prospects",
]


def clone_import_data(db, source_scenario_id: str, target_scenario_id: str, user_id: str) -> Dict[str, Any]:
    """Copy imported data from one scenario to another."""
    summary = {"cloned": 0, "tables": {}, "errors": []}

    # Tables keyed only by scenario_id
    for table in IMPORT_CLONE_SCENARIO_TABLES:
        try:
            resp = db.client.table(table).select("*").eq("scenario_id", source_scenario_id).execute()
            rows = resp.data or []
            if not rows:
                continue
            payload = []
            for row in rows:
                new_row = {k: v for k, v in row.items() if k not in ["id", "created_at", "updated_at"]}
                new_row["scenario_id"] = target_scenario_id
                payload.append(new_row)
            if payload:
                db.client.table(table).upsert(payload).execute()
                summary["tables"][table] = len(payload)
                summary["cloned"] += len(payload)
        except Exception as e:
            summary["errors"].append(f"{table}: {e}")

    # Tables keyed by user_id + scenario_id
    for table in IMPORT_CLONE_USER_SCENARIO_TABLES:
        try:
            resp = (
                db.client.table(table)
                .select("*")
                .eq("scenario_id", source_scenario_id)
                .eq("user_id", user_id)
                .execute()
            )
            rows = resp.data or []
            if not rows:
                continue
            payload = []
            for row in rows:
                new_row = {k: v for k, v in row.items() if k not in ["id", "created_at", "updated_at"]}
                new_row["scenario_id"] = target_scenario_id
                new_row["user_id"] = user_id
                payload.append(new_row)
            if payload:
                db.client.table(table).upsert(payload).execute()
                summary["tables"][table] = len(payload)
                summary["cloned"] += len(payload)
        except Exception as e:
            summary["errors"].append(f"{table}: {e}")

    return summary


def render_scenario_selector():
    """Render scenario selector in sidebar."""
    db = st.session_state.db_handler
    user_id = get_user_id()
    
    # Clean sidebar header - no emojis, matches website style
    st.sidebar.markdown("""
    <div style="
        text-align: center; 
        padding: 1.5rem 0.5rem; 
        border-bottom: 1px solid #27272A; 
        margin-bottom: 1.5rem;
    ">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 0.75rem;">
            <div style="flex: 1; height: 1px; background: linear-gradient(90deg, transparent, #52525B);"></div>
            <div style="padding: 0 0.75rem;">
                <span style="color: #FAFAFA; font-size: 0.6rem; letter-spacing: 0.2em;">●</span>
            </div>
            <div style="flex: 1; height: 1px; background: linear-gradient(90deg, #52525B, transparent);"></div>
        </div>
        <h2 style="
            color: #FAFAFA; 
            margin: 0; 
            font-size: 0.8rem; 
            font-weight: 600; 
            letter-spacing: 0.2em; 
            font-family: 'Inter', sans-serif;
        ">VALUATION</h2>
        <h2 style="
            color: #FAFAFA; 
            margin: 0.125rem 0 0 0; 
            font-size: 0.8rem; 
            font-weight: 600; 
            letter-spacing: 0.2em; 
            font-family: 'Inter', sans-serif;
        ">PLATFORM</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### Scenario")
    
    # Load scenarios
    scenarios = db.get_user_scenarios(user_id)
    scenario_opts = {s['name']: s['id'] for s in scenarios}
    
    if not scenario_opts:
        st.sidebar.warning("No scenarios found")
        with st.sidebar.expander("Create Scenario"):
            new_name = st.text_input("Name", placeholder="e.g. Base Case 2025")
            st.caption("First scenario will start with empty imports.")
            if st.button("Create", type="primary", use_container_width=True):
                if new_name:
                    new_scen = db.create_scenario(user_id, new_name)
                    if new_scen:
                        st.session_state.scenario_id = new_scen['id']
                        st.success("Created!")
                        st.rerun()
        return False
    
    # Current selection index
    current_idx = 0
    if st.session_state.scenario_id in scenario_opts.values():
        curr_name = [k for k, v in scenario_opts.items() if v == st.session_state.scenario_id][0]
        current_idx = list(scenario_opts.keys()).index(curr_name) + 1
    
    selected_name = st.sidebar.selectbox(
        "Active Scenario",
        options=["Select..."] + list(scenario_opts.keys()),
        index=current_idx,
        label_visibility="collapsed"
    )
    
    if selected_name != "Select...":
        new_scenario_id = scenario_opts[selected_name]
        # NEW: Load workflow progress when scenario changes
        if new_scenario_id != st.session_state.get('scenario_id'):
            st.session_state.scenario_id = new_scenario_id
            # Load workflow progress for new scenario
            try:
                workflow_progress = load_workflow_progress(db, new_scenario_id)
                st.session_state.workflow_progress = workflow_progress
            except:
                st.session_state.workflow_progress = {}
        else:
            st.session_state.scenario_id = new_scenario_id
    
    # New scenario
    with st.sidebar.expander("New Scenario"):
        new_name = st.text_input("Name", placeholder="e.g. Upside Case", key="new_scen_name")
        import_mode = st.radio(
            "Imports",
            ["Start from scratch", "Copy existing imports"],
            horizontal=True,
            key="new_scen_import_mode",
        )

        source_scenario_id = None
        if import_mode == "Copy existing imports":
            available_sources = list(scenario_opts.keys())
            if not available_sources:
                st.info("No existing scenarios to copy from.")
            else:
                source_name = st.selectbox(
                    "Source scenario",
                    options=["Select source..."] + available_sources,
                    key="new_scen_source",
                )
                if source_name != "Select source...":
                    source_scenario_id = scenario_opts[source_name]
        create_disabled = not new_name or (import_mode == "Copy existing imports" and not source_scenario_id)

        if st.button(
            "Create",
            type="primary",
            use_container_width=True,
            key="create_scen",
            disabled=create_disabled,
        ):
            new_scen = db.create_scenario(user_id, new_name)
            if new_scen:
                if import_mode == "Copy existing imports" and source_scenario_id:
                    clone_summary = clone_import_data(db, source_scenario_id, new_scen["id"], user_id)
                    if clone_summary["errors"]:
                        st.warning("Imported with warnings: " + "; ".join(clone_summary["errors"]))
                    st.info(f"Copied {clone_summary['cloned']} records from imports.")
                st.session_state.scenario_id = new_scen['id']
                st.success("Created!")
                st.rerun()
    
    return st.session_state.scenario_id is not None


# =============================================================================
# NAVIGATION
# =============================================================================

def render_navigation():
    """Render the main navigation in sidebar - aligned with workflow sequence."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    
    # Get workflow progress to show completion status
    db = st.session_state.db_handler
    user_id = get_user_id()
    scenario_id = st.session_state.scenario_id
    
    workflow_progress = {}
    if scenario_id:
        try:
            workflow_progress = calculate_workflow_progress(db, scenario_id, user_id)
        except:
            pass
    
    # Navigation aligned with workflow sequence
    # Start with Command Center, then follow workflow order
    sections = [
        ('home', 'Command Center', None),  # Always available
    ]
    
    # Add workflow stages in order
    for stage in WORKFLOW_STAGES:
        section_id = stage['section']
        stage_id = stage['id']
        stage_name = stage['name']
        
        # Map workflow stages to navigation
        # Check for duplicates to avoid duplicate keys
        if section_id == 'setup':
            if not any(s[0] == 'setup' for s in sections):
                sections.append(('setup', 'Setup', stage_id))
        elif section_id == 'ai_assumptions':
            # Only add once (covers both ai_analysis and assumptions_review)
            if not any(s[0] == 'ai_assumptions' for s in sections):
                sections.append(('ai_assumptions', 'AI Assumptions', stage_id))
        elif section_id == 'forecast':
            # Only add once (covers both forecast and results stages)
            if not any(s[0] == 'forecast' for s in sections):
                sections.append(('forecast', 'Forecast', stage_id))
        elif section_id == 'manufacturing':
            if not any(s[0] == 'manufacturing' for s in sections):
                sections.append(('manufacturing', 'Manufacturing Strategy', stage_id))
        elif section_id == 'funding':
            if not any(s[0] == 'funding' for s in sections):
                sections.append(('funding', 'Funding & Returns', stage_id))
    
    # Add additional sections (not in main workflow)
    sections.extend([
        ('whatif', 'What-If Agent', None),  # Sprint 17
        ('ai_analysis', 'AI Trend Analysis', None),  # Sprint 14
        ('compare', 'Compare', None),
        ('users', 'Access Control', None),
    ])
    
    for section_id, label, workflow_stage_id in sections:
        # Skip AI Assumptions if not available
        if section_id == 'ai_assumptions' and not AI_ASSUMPTIONS_AVAILABLE:
            continue
        
        # Skip manufacturing if not available
        if section_id == 'manufacturing' and not VERTICAL_INTEGRATION_AVAILABLE:
            continue
        
        # Skip AI analysis if not available
        if section_id == 'ai_analysis' and not AI_TREND_ANALYSIS_AVAILABLE:
            continue
        
        # Skip What-If Agent if not available
        if section_id == 'whatif' and not WHATIF_AGENT_AVAILABLE:
            continue
            
        is_active = st.session_state.current_section == section_id
        
        # Check if workflow stage is complete
        is_complete = False
        if workflow_stage_id and workflow_progress:
            stage_status = workflow_progress.get('stages', {}).get(workflow_stage_id, {})
            is_complete = stage_status.get('complete', False)
        
        # Format label with completion indicator
        if is_complete:
            display_label = f"✅ {label}"
            button_type = "secondary"
        elif is_active:
            display_label = f"→ {label}"
            button_type = "primary"
        else:
            display_label = label
            button_type = "secondary"
        
        # Custom styling for active state
        if is_active:
            st.sidebar.markdown(f"""
            <div style="
                background-color: #27272A;
                border-left: 2px solid #D4A537;
                padding: 0.5rem 1rem;
                margin: 0.25rem 0;
                color: #FAFAFA;
                font-weight: 500;
                font-size: 0.875rem;
            ">{display_label}</div>
            """, unsafe_allow_html=True)
        else:
            if st.sidebar.button(
                display_label,
                key=f"nav_{section_id}",
                use_container_width=True,
                type=button_type
            ):
                st.session_state.current_section = section_id
                st.rerun()
    
    # Note about required step
    if AI_ASSUMPTIONS_AVAILABLE:
        st.sidebar.caption("* Required before Forecast")


def handle_navigation():
    """Handle programmatic navigation requests."""
    if st.session_state.navigate_to:
        st.session_state.current_section = st.session_state.navigate_to
        
        if st.session_state.navigate_step:
            st.session_state.setup_step = st.session_state.navigate_step
        
        st.session_state.navigate_to = None
        st.session_state.navigate_step = None
        st.rerun()


# =============================================================================
# ASSUMPTIONS CHECK HELPER
# =============================================================================

def check_assumptions_saved(db, scenario_id: str) -> bool:
    """Check if AI assumptions have been saved for this scenario."""
    if not AI_ASSUMPTIONS_AVAILABLE:
        return True  # Skip check if module not available
    
    assumptions = get_saved_assumptions(db, scenario_id)
    if assumptions is None:
        return False
    
    return getattr(assumptions, 'assumptions_saved', False)


def render_assumptions_required_warning():
    """Render a warning when assumptions are required but not saved."""
    st.warning("""
    ⚠️ **AI Assumptions Required**
    
    Before running the forecast, you must complete the AI Assumptions step:
    1. Go to **AI Assumptions** in the sidebar
    2. Click **Run AI Analysis** to analyze historical data
    3. Review and adjust the proposed assumptions
    4. Click **Save Assumptions** to commit
    
    This ensures your forecast uses data-driven assumptions with proper probability distributions for Monte Carlo simulation.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("→ Go to AI Assumptions", type="primary", use_container_width=True):
            st.session_state.current_section = 'ai_assumptions'
            st.rerun()
    with col2:
        if st.button("Skip (Use Defaults)", use_container_width=True):
            st.session_state['skip_assumptions_check'] = True
            st.rerun()


# =============================================================================
# WORKFLOW STATE MANAGEMENT (NEW - Sprint 16.5)
# =============================================================================

# Define workflow stages in order
WORKFLOW_STAGES = [
    {
        'id': 'setup',
        'name': 'Setup',
        'section': 'setup',
        'description': 'Configure scenario basics, data imports, and assumptions',
        'required': True
    },
    {
        'id': 'ai_analysis',
        'name': 'AI Analysis',
        'section': 'ai_assumptions',
        'description': 'Run AI analysis on historical data',
        'required': True,
        'prerequisites': ['setup']
    },
    {
        'id': 'assumptions_review',
        'name': 'Assumptions Review',
        'section': 'ai_assumptions',
        'description': 'Review and save AI-derived assumptions',
        'required': True,
        'prerequisites': ['ai_analysis']
    },
    {
        'id': 'forecast',
        'name': 'Forecast',
        'section': 'forecast',
        'description': 'Run financial forecast',
        'required': True,
        'prerequisites': ['assumptions_review']
    },
    {
        'id': 'manufacturing',
        'name': 'Manufacturing Strategy',
        'section': 'manufacturing',
        'description': 'Configure make vs buy decisions (optional)',
        'required': False,
        'prerequisites': []  # No prerequisites - can be configured before or after forecast
    },
    {
        'id': 'funding',
        'name': 'Funding & Returns',
        'section': 'funding',
        'description': 'Model financing and returns',
        'required': False,
        'prerequisites': ['forecast']
    },
    {
        'id': 'results',
        'name': 'Results & Analysis',
        'section': 'forecast',
        'description': 'View results and analysis',
        'required': False,
        'prerequisites': ['forecast']
    }
]


def get_workflow_stage_by_id(stage_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow stage definition by ID."""
    for stage in WORKFLOW_STAGES:
        if stage['id'] == stage_id:
            return stage
    return None


def get_workflow_stage_by_section(section: str) -> Optional[Dict[str, Any]]:
    """Get workflow stage definition by section name."""
    for stage in WORKFLOW_STAGES:
        if stage['section'] == section:
            return stage
    return None


def check_workflow_prerequisites(db, scenario_id: str, stage_id: str, user_id: str) -> Tuple[bool, List[str]]:
    """
    Check if prerequisites for a workflow stage are met.
    
    Returns:
        (is_complete, missing_prerequisites)
    """
    stage = get_workflow_stage_by_id(stage_id)
    if not stage or not stage.get('prerequisites'):
        return True, []
    
    missing = []
    for prereq_id in stage['prerequisites']:
        if not is_workflow_stage_complete(db, scenario_id, prereq_id, user_id):
            missing.append(prereq_id)
    
    return len(missing) == 0, missing


def is_workflow_stage_complete(db, scenario_id: str, stage_id: str, user_id: str) -> bool:
    """
    Check if a workflow stage is complete.
    
    Checks actual completion status first, then database as backup.
    This ensures real-time accuracy.
    """
    if not scenario_id:
        return False
    
    stage = get_workflow_stage_by_id(stage_id)
    if not stage:
        return False
    
    # Check actual completion status (most reliable)
    is_actually_complete = False
    
    if stage_id == 'setup':
        # Check if setup is complete (has assumptions, fleet, etc.)
        # First check database workflow_progress as authoritative source
        try:
            if hasattr(db, 'client'):
                result = db.client.table('workflow_progress').select('completed').eq(
                    'scenario_id', scenario_id
                ).eq('stage', 'setup').execute()
                if result.data and len(result.data) > 0:
                    db_complete = result.data[0].get('completed', False)
                    if db_complete:
                        return True  # Database says complete, trust it
        except:
            pass  # Non-critical, continue with actual checks
        
        # Check actual completion status
        try:
            assumptions = db.get_scenario_assumptions(scenario_id, user_id)
            has_assumptions = bool(assumptions and assumptions.get('wacc'))
            
            # Check machines - try both methods
            machines = db.get_machine_instances(user_id, scenario_id)
            has_machines = len(machines) > 0 if machines else False
            
            # Fallback: check installed_base table if machine_instances is empty
            if not has_machines:
                try:
                    installed_base = db.get_installed_base(scenario_id)
                    has_machines = len(installed_base) > 0 if installed_base else False
                except:
                    pass
            
            is_actually_complete = has_assumptions and has_machines
        except Exception as e:
            # If there's an error, check database as fallback
            try:
                if hasattr(db, 'client'):
                    result = db.client.table('workflow_progress').select('completed').eq(
                        'scenario_id', scenario_id
                    ).eq('stage', 'setup').execute()
                    if result.data and len(result.data) > 0:
                        is_actually_complete = result.data[0].get('completed', False)
                    else:
                        is_actually_complete = False
                else:
                    is_actually_complete = False
            except:
                is_actually_complete = False
    
    elif stage_id == 'ai_analysis':
        # Check if AI analysis has been run
        try:
            from components.ai_assumptions_engine import load_assumptions_from_db
            assumptions = load_assumptions_from_db(db, scenario_id)
            is_actually_complete = assumptions is not None and getattr(assumptions, 'analysis_complete', False)
        except:
            is_actually_complete = False
    
    elif stage_id == 'assumptions_review':
        # Check if assumptions have been saved
        is_actually_complete = check_assumptions_saved(db, scenario_id)
    
    elif stage_id == 'forecast':
        # Check if forecast has been run
        # First check database workflow_progress as authoritative source
        try:
            if hasattr(db, 'client'):
                result = db.client.table('workflow_progress').select('completed').eq(
                    'scenario_id', scenario_id
                ).eq('stage', 'forecast').execute()
                if result.data and len(result.data) > 0:
                    db_complete = result.data[0].get('completed', False)
                    if db_complete:
                        return True  # Database says complete, trust it
        except:
            pass  # Non-critical, continue with actual checks
        
        # Check session state for active forecast results
        forecast_key = f"forecast_results_{scenario_id}"
        has_session_results = forecast_key in st.session_state and st.session_state.get(forecast_key) is not None
        
        # Also check for legacy session state key
        if not has_session_results:
            has_session_results = 'forecast_results' in st.session_state and st.session_state.get('forecast_results') is not None
        
        # Check for saved forecast snapshots (forecasts persist as snapshots)
        has_snapshot = False
        try:
            if hasattr(db, 'client'):
                snapshot_result = db.client.table('forecast_snapshots').select('id').eq(
                    'scenario_id', scenario_id
                ).eq('user_id', user_id).order('created_at', desc=True).limit(1).execute()
                has_snapshot = snapshot_result.data and len(snapshot_result.data) > 0
        except:
            pass  # Non-critical
        
        is_actually_complete = has_session_results or has_snapshot
    
    elif stage_id == 'manufacturing':
        # Optional - check if manufacturing strategy exists
        # First check database (authoritative source)
        try:
            if hasattr(db, 'client'):
                result = db.client.table('workflow_progress').select('completed').eq(
                    'scenario_id', scenario_id
                ).eq('stage', 'manufacturing').execute()
                if result.data and len(result.data) > 0:
                    db_complete = result.data[0].get('completed', False)
                    if db_complete:
                        return True  # Database says complete, trust it
        except:
            pass  # Non-critical, continue with actual checks
        
        # Check if manufacturing strategy is saved in database
        try:
            assumptions = db.get_scenario_assumptions(scenario_id, user_id)
            # Check for manufacturing_strategy_saved flag or manufacturing_strategy data
            has_saved_flag = assumptions and assumptions.get('manufacturing_strategy_saved', False)
            has_strategy_data = assumptions and 'manufacturing_strategy' in assumptions and assumptions.get('manufacturing_strategy') is not None
            
            if has_saved_flag or has_strategy_data:
                is_actually_complete = True
            else:
                # Fallback: check session state
                vi_scenario = st.session_state.get('vi_scenario')
                is_actually_complete = vi_scenario is not None
        except:
            # Fallback: check session state only
            try:
                vi_scenario = st.session_state.get('vi_scenario')
                is_actually_complete = vi_scenario is not None
            except:
                is_actually_complete = False
    
    elif stage_id == 'funding':
        # Optional - check if funding has been configured
        forecast_key = f"forecast_results_{scenario_id}"
        is_actually_complete = forecast_key in st.session_state
    
    # If actually complete, ensure it's marked in database
    if is_actually_complete:
        try:
            # Sync to database (don't wait for result)
            mark_workflow_stage_complete(db, scenario_id, stage_id, user_id)
        except:
            pass  # Non-critical
    
    return is_actually_complete


def mark_workflow_stage_complete(db, scenario_id: str, stage_id: str, user_id: str) -> bool:
    """Mark a workflow stage as complete in database."""
    if not scenario_id:
        return False
    
    try:
        if hasattr(db, 'client'):
            # Upsert workflow progress
            data = {
                'scenario_id': scenario_id,
                'stage': stage_id,
                'completed': True,
                'completed_at': datetime.now().isoformat(),
                'user_id': user_id
            }
            
            try:
                db.client.table('workflow_progress').upsert(
                    data, on_conflict='scenario_id,stage'
                ).execute()
            except:
                # Table might not exist, try insert
                try:
                    db.client.table('workflow_progress').insert(data).execute()
                except:
                    pass  # Table doesn't exist yet - will be created by migration
            
            # Update session state
            if 'workflow_progress' not in st.session_state:
                st.session_state.workflow_progress = {}
            st.session_state.workflow_progress[stage_id] = True
            
            return True
    except Exception as e:
        st.error(f"Error saving workflow progress: {str(e)}")
        return False
    
    return False


def load_workflow_progress(db, scenario_id: str) -> Dict[str, bool]:
    """Load workflow progress from database."""
    progress = {}
    
    if not scenario_id:
        return progress
    
    try:
        if hasattr(db, 'client'):
            result = db.client.table('workflow_progress').select('stage, completed').eq(
                'scenario_id', scenario_id
            ).execute()
            
            if result.data:
                for row in result.data:
                    progress[row['stage']] = row.get('completed', False)
    except Exception:
        pass  # Table might not exist yet
    
    return progress


def get_current_workflow_stage(db, scenario_id: str, user_id: str) -> Optional[str]:
    """Get the current (incomplete) workflow stage."""
    if not scenario_id:
        return 'setup'
    
    for stage in WORKFLOW_STAGES:
        if not is_workflow_stage_complete(db, scenario_id, stage['id'], user_id):
            # Check if prerequisites are met
            can_start, _ = check_workflow_prerequisites(db, scenario_id, stage['id'], user_id)
            if can_start:
                return stage['id']
    
    # All stages complete
    return 'results'


def get_next_workflow_stage(db, scenario_id: str, user_id: str, current_stage_id: str = None) -> Optional[Dict[str, Any]]:
    """Get the next workflow stage after the current one."""
    if not current_stage_id:
        current_stage_id = get_current_workflow_stage(db, scenario_id, user_id)
    
    current_idx = None
    for i, stage in enumerate(WORKFLOW_STAGES):
        if stage['id'] == current_stage_id:
            current_idx = i
            break
    
    if current_idx is not None and current_idx < len(WORKFLOW_STAGES) - 1:
        next_stage = WORKFLOW_STAGES[current_idx + 1]
        # Check if prerequisites are met
        can_start, _ = check_workflow_prerequisites(db, scenario_id, next_stage['id'], user_id)
        if can_start:
            return next_stage
    
    return None


def calculate_workflow_progress(db, scenario_id: str, user_id: str) -> Dict[str, Any]:
    """Calculate overall workflow progress."""
    if not scenario_id:
        return {
            'percentage': 0,
            'completed_stages': 0,
            'total_stages': len([s for s in WORKFLOW_STAGES if s.get('required', True)]),
            'current_stage': 'setup',
            'stages': {}
        }
    
    # Check cache first
    cache_key = f"workflow_progress_{scenario_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    completed = 0
    total_required = len([s for s in WORKFLOW_STAGES if s.get('required', True)])
    stages_status = {}
    current_stage_id = get_current_workflow_stage(db, scenario_id, user_id)
    
    for stage in WORKFLOW_STAGES:
        is_complete = is_workflow_stage_complete(db, scenario_id, stage['id'], user_id)
        stages_status[stage['id']] = {
            'complete': is_complete,
            'name': stage['name'],
            'description': stage['description'],
            'is_current': stage['id'] == current_stage_id
        }
        if is_complete and stage.get('required', True):
            completed += 1
    
    percentage = int((completed / total_required) * 100) if total_required > 0 else 0
    
    result = {
        'percentage': percentage,
        'completed_stages': completed,
        'total_stages': total_required,
        'current_stage': current_stage_id,
        'stages': stages_status
    }
    
    # Cache the result
    st.session_state[cache_key] = result
    
    return result


# =============================================================================
# SECTION RENDERERS
# =============================================================================

def render_home_section():
    """Render the Command Center home section."""
    db = st.session_state.db_handler
    user_id = get_user_id()
    
    if COMMAND_CENTER_AVAILABLE:
        render_command_center(
            db=db,
            scenario_id=st.session_state.scenario_id,
            user_id=user_id,
            on_navigate=lambda section, step: (
                setattr(st.session_state, 'navigate_to', section),
                setattr(st.session_state, 'navigate_step', step),
                st.rerun()
            )
        )
    elif LEGACY_DASHBOARD_AVAILABLE:
        render_dashboard(st.session_state.scenario_id, user_id)
    else:
        st.header("Command Center")
        st.info("Command Center component not available. Please ensure `components/command_center.py` exists.")


def render_setup_section():
    """Render the Setup wizard section."""
    db = st.session_state.db_handler
    user_id = get_user_id()
    scenario_id = st.session_state.scenario_id
    
    # NEW: Show contextual help for setup
    try:
        from components.workflow_guidance import render_contextual_help, get_stage_tips
        from app_refactored import get_workflow_stage_by_id
        stage = get_workflow_stage_by_id('setup')
        if stage:
            render_contextual_help(
                'setup',
                stage['name'],
                stage.get('description', ''),
                stage.get('prerequisites', []),
                get_stage_tips('setup')
            )
    except ImportError:
        pass
    
    if SETUP_WIZARD_AVAILABLE:
        render_setup_wizard(
            db=db,
            scenario_id=scenario_id,
            user_id=user_id,
            initial_step=st.session_state.setup_step
        )
        
        # NEW: Mark setup complete when basics are configured
        try:
            assumptions = db.get_scenario_assumptions(scenario_id, user_id)
            has_assumptions = bool(assumptions and assumptions.get('wacc'))
            
            # Check machines - try both methods
            machines = db.get_machine_instances(user_id, scenario_id)
            has_machines = len(machines) > 0 if machines else False
            
            # Fallback: check installed_base table if machine_instances is empty
            if not has_machines:
                try:
                    installed_base = db.get_installed_base(scenario_id)
                    has_machines = len(installed_base) > 0 if installed_base else False
                except:
                    pass
            
            if has_assumptions and has_machines:
                # Mark complete and refresh workflow progress
                mark_workflow_stage_complete(db, scenario_id, 'setup', user_id)
                # Clear cached progress to force refresh
                cache_key = f"workflow_progress_{scenario_id}"
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
        except Exception as e:
            # Log error but don't block
            pass
    else:
        st.header("Setup")
        st.info("Setup Wizard component not available. Please ensure `components/setup_wizard.py` exists.")


def render_ai_assumptions_section_wrapper():
    """Render the AI Assumptions Engine section (Sprint 14 Enhancement)."""
    db = st.session_state.db_handler
    user_id = get_user_id()
    scenario_id = st.session_state.scenario_id
    
    # NEW: Check prerequisites
    can_access, missing = check_workflow_prerequisites(db, scenario_id, 'ai_analysis', user_id)
    if not can_access:
        # Debug info (can be removed in production)
        setup_complete = is_workflow_stage_complete(db, scenario_id, 'setup', user_id)
        
        try:
            from components.workflow_guidance import render_prerequisite_warning
            render_prerequisite_warning(missing, 'ai_assumptions')
            return
        except ImportError:
            # Enhanced error message with diagnostic info
            st.warning("⚠️ **Prerequisites Not Met**")
            st.info(f"""
            Please complete the following steps before continuing:
            - **Setup:** Configure scenario basics, data imports, and assumptions
            
            **Diagnostic Info:**
            - Setup stage complete: {setup_complete}
            - Missing prerequisites: {', '.join(missing) if missing else 'None'}
            
            **To fix:**
            1. Go to **Setup** in the sidebar
            2. Complete all setup steps (Basics, Customers, Fleet, etc.)
            3. Ensure assumptions are saved (WACC configured)
            4. Ensure fleet/machines are added
            """)
            if st.button("→ Go to Setup", type="primary"):
                st.session_state.current_section = 'setup'
                st.rerun()
            return
    
    if AI_ASSUMPTIONS_AVAILABLE:
        # NEW: Show contextual help
        try:
            from components.workflow_guidance import render_contextual_help, get_stage_tips
            from app_refactored import get_workflow_stage_by_id
            stage = get_workflow_stage_by_id('ai_analysis')
            if stage:
                render_contextual_help(
                    'ai_analysis',
                    stage['name'],
                    stage.get('description', ''),
                    stage.get('prerequisites', []),
                    get_stage_tips('ai_analysis')
                )
        except ImportError:
            pass
        
        render_ai_assumptions_section(
            db=db,
            scenario_id=scenario_id,
            user_id=user_id
        )
        
        # NEW: Mark stages complete when assumptions are saved
        if check_assumptions_saved(db, scenario_id):
            # Mark both ai_analysis and assumptions_review as complete
            mark_workflow_stage_complete(db, scenario_id, 'ai_analysis', user_id)
            mark_workflow_stage_complete(db, scenario_id, 'assumptions_review', user_id)
            # Clear cached progress to force refresh
            cache_key = f"workflow_progress_{scenario_id}"
            if cache_key in st.session_state:
                del st.session_state[cache_key]
    else:
        st.header("AI Assumptions Engine")
        st.warning("AI Assumptions Engine module not available.")
        
        st.markdown("""
        **This is a REQUIRED step before running the Forecast.**
        
        **Features:**
        - Analyze historical data to derive assumptions
        - Fit probability distributions (Normal, Lognormal, Triangular, Beta, PERT)
        - Visual distribution preview with adjustment controls
        - Manufacturing make vs buy recommendations
        - Save assumptions for Forecast and MC Simulation
        
        **To enable:**
        1. Download `ai_assumptions_engine.py`
        2. Save to `components/ai_assumptions_engine.py`
        3. Restart the application
        """)


def render_forecast_section_view():
    """Render the Forecast & Analysis section."""
    db = st.session_state.db_handler
    user_id = get_user_id()
    scenario_id = st.session_state.scenario_id
    
    # NEW: Check workflow prerequisites
    can_access, missing = check_workflow_prerequisites(db, scenario_id, 'forecast', user_id)
    if not can_access:
        try:
            from components.workflow_guidance import render_prerequisite_warning
            render_prerequisite_warning(missing, 'forecast')
            return
        except ImportError:
            # Fallback to old check
            pass
    
    # Check if assumptions have been saved (unless skipped)
    skip_check = st.session_state.get('skip_assumptions_check', False)
    
    if AI_ASSUMPTIONS_AVAILABLE and not skip_check:
        if not check_assumptions_saved(db, scenario_id):
            st.header("Forecast & Analysis")
            render_assumptions_required_warning()
            return
    
    # NEW: Show contextual help
    try:
        from components.workflow_guidance import render_contextual_help, get_stage_tips
        from app_refactored import get_workflow_stage_by_id
        stage = get_workflow_stage_by_id('forecast')
        if stage:
            render_contextual_help(
                'forecast',
                stage['name'],
                stage.get('description', ''),
                stage.get('prerequisites', []),
                get_stage_tips('forecast')
            )
    except ImportError:
        pass
    
    if FORECAST_SECTION_AVAILABLE:
        render_forecast_section(
            db=db,
            scenario_id=scenario_id,
            user_id=user_id
        )
        
        # NEW: Mark forecast stage complete when forecast is run
        forecast_key = f"forecast_results_{scenario_id}"
        if forecast_key in st.session_state and st.session_state.get(forecast_key):
            mark_workflow_stage_complete(db, scenario_id, 'forecast', user_id)
            # Clear cached progress to force refresh
            cache_key = f"workflow_progress_{scenario_id}"
            if cache_key in st.session_state:
                del st.session_state[cache_key]
    elif LEGACY_FORECAST_AVAILABLE:
        render_forecast_viewer(scenario_id, user_id)
    else:
        st.header("Forecast & Analysis")
        st.info("Forecast components not available.")


def render_compare_section():
    """Render the Scenario Comparison section."""
    db = st.session_state.db_handler
    user_id = get_user_id()
    
    if SCENARIO_COMPARISON_AVAILABLE:
        render_scenario_comparison(
            db=db,
            scenario_id=st.session_state.scenario_id,
            user_id=user_id
        )
    else:
        st.header("Scenario Comparison")
        st.info("Scenario comparison component not available. Please ensure `components/scenario_comparison.py` exists.")


def render_users_section():
    """Render the User & Access Control section."""
    db = st.session_state.db_handler
    user_id = get_user_id()
    
    if USER_MANAGEMENT_AVAILABLE:
        render_user_management(
            db=db,
            scenario_id=st.session_state.scenario_id,
            user_id=user_id
        )
    else:
        st.header("Access Control")
        st.info("User management component not available. Please ensure `components/user_management.py` exists.")
        
        st.markdown("""
        **Coming Soon:**
        - User role management (Owner, Admin, Analyst, Viewer, Investor)
        - Scenario sharing with team members and external parties
        - Audit logging for compliance tracking
        - Sanitized investor views for due diligence
        
        **To enable:** Run the Sprint 7 migration and install the user_management component.
        """)


def render_funding_section_view():
    """Render the Funding & Returns section."""
    db = st.session_state.db_handler
    user_id = get_user_id()
    scenario_id = st.session_state.scenario_id
    
    # NEW: Check prerequisites (optional stage, but requires forecast)
    can_access, missing = check_workflow_prerequisites(db, scenario_id, 'funding', user_id)
    if not can_access:
        try:
            from components.workflow_guidance import render_prerequisite_warning
            render_prerequisite_warning(missing, 'funding')
            return
        except ImportError:
            st.info("💡 **Optional Stage** - Run a forecast first to enable funding analysis.")
            if st.button("→ Go to Forecast", type="primary"):
                st.session_state.current_section = 'forecast'
                st.rerun()
            return
    
    if FUNDING_AVAILABLE:
        # Get forecast results if available (check multiple sources)
        forecast_key = f"forecast_results_{scenario_id}"
        forecast_results = st.session_state.get(forecast_key)
        
        # Fallback to legacy session state key
        if not forecast_results:
            forecast_results = st.session_state.get('forecast_results')
        
        # If still no results, try loading from snapshot (Sprint 19 - Persistence)
        if not forecast_results:
            try:
                from components.forecast_section import load_snapshots
                import json
                snapshots = load_snapshots(db, scenario_id, limit=1)
                if snapshots and len(snapshots) > 0:
                    latest_snapshot = snapshots[0]
                    # Convert snapshot data to forecast_results format
                    forecast_data_str = latest_snapshot.get('forecast_data', '{}')
                    summary_stats_str = latest_snapshot.get('summary_stats', '{}')
                    
                    if isinstance(forecast_data_str, str):
                        forecast_data = json.loads(forecast_data_str)
                    else:
                        forecast_data = forecast_data_str
                    
                    if isinstance(summary_stats_str, str):
                        summary_stats = json.loads(summary_stats_str)
                    else:
                        summary_stats = summary_stats_str
                    
                    # Reconstruct forecast_results from snapshot
                    forecast_results = {
                        **forecast_data,
                        'summary_stats': summary_stats
                    }
            except Exception as e:
                # Non-critical - forecast_results will remain None
                pass
        
        # NEW: Show contextual help
        try:
            from components.workflow_guidance import render_contextual_help, get_stage_tips
            from app_refactored import get_workflow_stage_by_id
            stage = get_workflow_stage_by_id('funding')
            if stage:
                render_contextual_help(
                    'funding',
                    stage['name'],
                    stage.get('description', ''),
                    stage.get('prerequisites', []),
                    get_stage_tips('funding')
                )
        except ImportError:
            pass
        
        render_funding_section(
            db=db,
            scenario_id=scenario_id,
            user_id=user_id,
            forecast_results=forecast_results
        )
    else:
        st.header("Funding & Returns")
        st.info("Funding module not available. Please ensure `components/funding_ui.py` and `funding_engine.py` are installed.")
        
        st.markdown("""
        **Features (Sprint 11):**
        - Debt instruments: Term loans, mezzanine, convertibles, trade finance
        - Equity financing: Ordinary and preference shares
        - Auto-overdraft facility with auto-repay
        - Equity IRR and Project IRR calculations
        - Goal seek (solve for target returns)
        - Sensitivity analysis
        
        **To enable:** 
        1. Run the Sprint 11 SQL migration
        2. Copy `funding_engine.py` to project root
        3. Copy `funding_ui.py` to components/
        """)


def render_manufacturing_section():
    """Render the Manufacturing Strategy / Vertical Integration section (Sprint 13)."""
    db = st.session_state.db_handler
    user_id = get_user_id()
    scenario_id = st.session_state.scenario_id
    
    # Manufacturing is optional and can be configured before or after forecast
    # No prerequisites required - users can set up manufacturing strategy independently
    
    if VERTICAL_INTEGRATION_AVAILABLE:
        # Check if AI assumptions are available to pre-populate
        if AI_ASSUMPTIONS_AVAILABLE:
            assumptions = get_saved_assumptions(db, scenario_id)
            if assumptions and hasattr(assumptions, 'manufacturing_assumptions'):
                # Store in session state for manufacturing module to use
                st.session_state['ai_mfg_assumptions'] = assumptions.manufacturing_assumptions
        
        # NEW: Show contextual help
        try:
            from components.workflow_guidance import render_contextual_help, get_stage_tips
            from app_refactored import get_workflow_stage_by_id
            stage = get_workflow_stage_by_id('manufacturing')
            if stage:
                render_contextual_help(
                    'manufacturing',
                    stage['name'],
                    stage.get('description', ''),
                    stage.get('prerequisites', []),
                    get_stage_tips('manufacturing')
                )
        except ImportError:
            pass
        
        render_vertical_integration_section(
            db=db,
            scenario_id=scenario_id,
            user_id=user_id
        )
        
        # NEW: Mark manufacturing complete when strategy is saved
        try:
            assumptions = db.get_scenario_assumptions(scenario_id, user_id)
            has_strategy = assumptions and (
                assumptions.get('manufacturing_strategy_saved', False) or
                (assumptions.get('manufacturing_strategy') is not None)
            )
            if has_strategy:
                # Mark complete and refresh workflow progress
                mark_workflow_stage_complete(db, scenario_id, 'manufacturing', user_id)
                # Clear cached progress to force refresh
                cache_key = f"workflow_progress_{scenario_id}"
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
        except:
            pass  # Non-critical
    else:
        st.header("Manufacturing Strategy")
        st.info("Vertical Integration module not available. Please ensure `components/vertical_integration.py` exists.")
        
        st.markdown("""
        **Features (Sprint 13 + Enhancements):**
        - Make vs Buy decision analysis
        - Historical data integration for pricing
        - Two analysis modes: Average or Part-Level
        - Import/Export template for bulk configuration
        - Manufacturing cost modeling (raw materials, labor, overhead)
        - Purchase cost modeling (supplier price, freight, duty)
        - CAPEX requirements and depreciation
        - Capacity constraint planning with ramp-up
        - Break-even volume analysis
        - NPV comparison with sensitivity analysis
        - AI-derived recommendations from AI Assumptions
        
        **To enable:** 
        1. Download `vertical_integration.py` (v2.0)
        2. Save to `components/vertical_integration.py`
        3. Restart the application
        """)


def render_whatif_section():
    """Render the What-If Agent section (Sprint 17)."""
    db = st.session_state.db_handler
    user_id = get_user_id()
    scenario_id = st.session_state.scenario_id
    
    # Check prerequisites
    can_access, missing = check_workflow_prerequisites(db, scenario_id, 'forecast', user_id)
    if not can_access:
        try:
            from components.workflow_guidance import render_prerequisite_warning
            render_prerequisite_warning(missing, 'whatif')
            return
        except ImportError:
            st.info("💡 Run a forecast first to enable What-If analysis.")
            if st.button("→ Go to Forecast", type="primary"):
                st.session_state.current_section = 'forecast'
                st.rerun()
            return
    
    if WHATIF_AGENT_AVAILABLE:
        # Get baseline forecast from session state first
        forecast_key = f"forecast_results_{scenario_id}"
        baseline_forecast = st.session_state.get(forecast_key)
        
        # If not in session state, try loading from latest snapshot
        if not baseline_forecast:
            baseline_forecast = st.session_state.get('forecast_results')
        
        # If still not found, try loading from database snapshots
        if not baseline_forecast:
            try:
                from components.forecast_section import load_snapshots
                
                snapshots = load_snapshots(db, scenario_id, limit=1)
                if snapshots:
                    snapshot = snapshots[0]
                    try:
                        from components.forecast_section import _snapshot_to_forecast_results
                        baseline_forecast = _snapshot_to_forecast_results(snapshot)
                    except Exception as e:
                        baseline_forecast = None
            except Exception:
                baseline_forecast = None
        
        render_whatif_agent(
            db=db,
            scenario_id=scenario_id,
            user_id=user_id,
            baseline_forecast=baseline_forecast
        )
    else:
        st.header("What-If Agent")
        st.info("What-If Agent module not available. Please ensure `components/whatif_agent.py` exists.")
        
        st.markdown("""
        **Features (Sprint 17):**
        - Natural language scenario queries
        - Assumption modification engine
        - Before/after comparison view
        - AI recommendation engine
        
        **To enable:** 
        1. Download `whatif_agent.py`
        2. Save to `components/whatif_agent.py`
        3. Restart the application
        """)


def render_ai_analysis_section():
    """Render the AI Trend Analysis section (Sprint 14 - Reporting)."""
    db = st.session_state.db_handler
    user_id = get_user_id()
    scenario_id = st.session_state.scenario_id
    
    if AI_TREND_ANALYSIS_AVAILABLE:
        render_trend_analysis_section(
            db=db,
            scenario_id=scenario_id,
            user_id=user_id
        )
    else:
        st.header("AI Trend Analysis")
        st.info("AI Trend Analysis module not available. Please ensure `components/ai_trend_analysis.py` exists.")
        
        st.markdown("""
        **Features (Sprint 14):**
        - Automated trend detection (linear regression, CAGR, growth rates)
        - Anomaly identification (Z-score, IQR methods)
        - Seasonality analysis (monthly/quarterly patterns)
        - AI-generated insights and recommendations
        - Interactive visualizations
        - Priority-ranked action items
        - Export analysis reports
        
        **Note:** For assumptions derivation, use the **AI Assumptions** section instead.
        This section is for reporting and insights on historical trends.
        
        **To enable:** 
        1. Download `ai_trend_analysis.py`
        2. Save to `components/ai_trend_analysis.py`
        3. Restart the application
        """)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Inject custom CSS (from ui_components if available)
    if UI_COMPONENTS_AVAILABLE:
        inject_custom_css()
    
    # Scenario selector (includes branded header)
    has_scenario = render_scenario_selector()
    
    if not has_scenario:
        # Render banner
        render_banner()
        
        # Clean welcome message
        st.markdown("""
        <div style="
            text-align: center; 
            padding: 3rem 2rem; 
            background-color: #0F0F11; 
            border: 1px solid #27272A; 
            max-width: 500px;
            margin: 2rem auto;
        ">
            <p style="
                color: #A1A1AA; 
                font-size: 0.9375rem; 
                line-height: 1.6;
                margin: 0;
            ">Select or create a scenario from the sidebar to begin your valuation analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Render banner at top
    render_banner()
    
    # NEW: Render workflow navigator if scenario is selected
    if st.session_state.scenario_id:
        try:
            from components.workflow_navigator import render_workflow_navigator_simple
            workflow_progress = calculate_workflow_progress(
                st.session_state.db_handler,
                st.session_state.scenario_id,
                get_user_id()
            )
            render_workflow_navigator_simple(
                current_stage=workflow_progress.get('current_stage', 'setup'),
                stages_status=workflow_progress.get('stages', {}),
                on_navigate=lambda stage_id, section: (
                    setattr(st.session_state, 'current_section', section),
                    st.rerun()
                ),
                key_prefix="main_workflow"
            )
            st.markdown("---")
        except ImportError:
            pass
    
    # Navigation
    render_navigation()
    
    # Handle programmatic navigation
    handle_navigation()
    
    # NEW: Show "What's Next?" widget in sidebar
    if st.session_state.scenario_id:
        try:
            from components.workflow_guidance import render_whats_next_widget
            workflow_progress = calculate_workflow_progress(
                st.session_state.db_handler,
                st.session_state.scenario_id,
                get_user_id()
            )
            current_stage_id = workflow_progress.get('current_stage', 'setup')
            next_stage = get_next_workflow_stage(
                st.session_state.db_handler,
                st.session_state.scenario_id,
                get_user_id(),
                current_stage_id
            )
            render_whats_next_widget(current_stage_id, next_stage, workflow_progress)
        except ImportError:
            pass
    
    # Module status (no emoji)
    with st.sidebar.expander("Module Status"):
        modules = [
            ("UI Components", UI_COMPONENTS_AVAILABLE),
            ("Command Center", COMMAND_CENTER_AVAILABLE),
            ("Setup Wizard", SETUP_WIZARD_AVAILABLE),
            ("AI Assumptions Engine", AI_ASSUMPTIONS_AVAILABLE),
            ("Forecast Section", FORECAST_SECTION_AVAILABLE),
            ("What-If Agent", WHATIF_AGENT_AVAILABLE),  # NEW - Sprint 17
            ("Funding Engine", FUNDING_AVAILABLE),
            ("Manufacturing Strategy", VERTICAL_INTEGRATION_AVAILABLE),
            ("AI Trend Analysis", AI_TREND_ANALYSIS_AVAILABLE),
            ("Scenario Comparison", SCENARIO_COMPARISON_AVAILABLE),
            ("User Management", USER_MANAGEMENT_AVAILABLE),
        ]
        for name, available in modules:
            status = "Active" if available else "Inactive"
            color = "#22C55E" if available else "#71717A"
            st.markdown(f"<span style='color: {color};'>●</span> {name}: {status}", unsafe_allow_html=True)
    
    # User info
    st.sidebar.markdown("---")
    st.sidebar.caption(f"User: {get_user_id()[:8]}...")
    
    # Version info
    st.sidebar.markdown(
        f"<div style='text-align: center; color: #64748b; font-size: 0.75rem;'>"
        f"v2.3 | {datetime.now().strftime('%Y-%m-%d')}"
        f"</div>",
        unsafe_allow_html=True
    )
    
    # Render current section
    section = st.session_state.current_section
    
    if section == 'home':
        render_home_section()
    elif section == 'setup':
        render_setup_section()
    elif section == 'ai_assumptions':
        render_ai_assumptions_section_wrapper()
    elif section == 'forecast':
        render_forecast_section_view()
    elif section == 'whatif':
        render_whatif_section()
    elif section == 'funding':
        render_funding_section_view()
    elif section == 'manufacturing':
        render_manufacturing_section()
    elif section == 'ai_analysis':
        render_ai_analysis_section()
    elif section == 'compare':
        render_compare_section()
    elif section == 'users':
        render_users_section()
    else:
        render_home_section()


if __name__ == "__main__":
    main()
