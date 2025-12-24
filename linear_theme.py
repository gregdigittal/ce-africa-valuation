"""
CE Africa Valuation Platform - Linear Theme
============================================
Drop-in replacement for ce_africa_branding.py

INSTALLATION:
1. Save this file as 'linear_theme.py' in your project root
2. In app_refactored.py, change ONE LINE:
   
   OLD: from ce_africa_branding import apply_ce_africa_branding
   NEW: from linear_theme import apply_ce_africa_branding
   
   That's it! The function name is the same so no other changes needed.
"""

import streamlit as st
from typing import Any

# =============================================================================
# COLOR SYSTEM
# =============================================================================

COLORS = {
    # Backgrounds (darkest to lightest)
    'bg_base': '#09090B',
    'bg_elevated': '#0F0F11',
    'bg_surface': '#18181B',
    'bg_hover': '#1F1F23',
    'bg_active': '#27272A',
    
    # Borders
    'border_subtle': '#27272A',
    'border_default': '#3F3F46',
    'border_strong': '#52525B',
    
    # Text
    'text_primary': '#FAFAFA',
    'text_secondary': '#A1A1AA',
    'text_tertiary': '#71717A',
    'text_disabled': '#52525B',
    
    # CE Africa Gold (used sparingly)
    'accent_gold': '#D4A537',
    'accent_gold_hover': '#E5B84A',
    'accent_gold_muted': 'rgba(212, 165, 55, 0.15)',
    'accent_gold_subtle': 'rgba(212, 165, 55, 0.08)',
    
    # Status
    'success': '#22C55E',
    'success_muted': 'rgba(34, 197, 94, 0.15)',
    'warning': '#F59E0B',
    'warning_muted': 'rgba(245, 158, 11, 0.15)',
    'error': '#EF4444',
    'error_muted': 'rgba(239, 68, 68, 0.15)',
    'info': '#3B82F6',
    'info_muted': 'rgba(59, 130, 246, 0.15)',
}

# =============================================================================
# MAIN THEME CSS
# =============================================================================

LINEAR_CSS = """
<style>
/* ==========================================================================
   CE AFRICA - LINEAR DESIGN SYSTEM
   ========================================================================== */

/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* CSS Variables */
:root {
    --bg-base: #09090B;
    --bg-elevated: #0F0F11;
    --bg-surface: #18181B;
    --bg-hover: #1F1F23;
    --bg-active: #27272A;
    --border-subtle: #27272A;
    --border-default: #3F3F46;
    --text-primary: #FAFAFA;
    --text-secondary: #A1A1AA;
    --text-tertiary: #71717A;
    --accent: #D4A537;
    --accent-hover: #E5B84A;
    --accent-muted: rgba(212, 165, 55, 0.15);
    --success: #22C55E;
    --warning: #F59E0B;
    --error: #EF4444;
    --info: #3B82F6;
}

/* Base */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-secondary) !important;
}

.stApp {
    background-color: var(--bg-base) !important;
}

.block-container {
    padding: 2rem 3rem !important;
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
}

a {
    color: var(--accent) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-elevated) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: var(--bg-elevated) !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.15s ease !important;
    background-color: var(--accent) !important;
    color: #000000 !important;
    border: none !important;
}

.stButton > button:hover {
    background-color: var(--accent-hover) !important;
    color: #000000 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(212, 165, 55, 0.25) !important;
}

/* Primary buttons - explicit black text */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
.stButton > button[data-testid="stBaseButton-primary"],
button[kind="primary"] {
    background-color: #D4A537 !important;
    color: #000000 !important;
    font-weight: 600 !important;
}

/* PRIMARY BUTTON TEXT - Target inner elements (p, span, div) */
.stButton > button p,
.stButton > button span,
.stButton > button div,
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div,
button[kind="primary"] p,
button[kind="primary"] span,
button[kind="primary"] div {
    color: #000000 !important;
}

.stButton > button:hover p,
.stButton > button:hover span,
.stButton > button:hover div {
    color: #000000 !important;
}

/* Form Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    padding: 0.625rem 0.875rem !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(212, 165, 55, 0.08) !important;
}

/* Labels */
.stTextInput > label,
.stNumberInput > label,
.stTextArea > label,
.stSelectbox > label,
.stMultiSelect > label,
.stSlider > label,
.stCheckbox > label {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
}

/* Dropdown */
[data-baseweb="menu"] {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 6px !important;
}

[data-baseweb="menu"] li {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
}

[data-baseweb="menu"] li:hover {
    background-color: var(--bg-hover) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

.stTabs [data-baseweb="tab"] {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: var(--text-tertiary) !important;
    background-color: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1rem !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.8125rem !important;
    color: var(--text-tertiary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

/* DataFrames */
.stDataFrame {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
}

.stDataFrame th {
    background-color: var(--bg-surface) !important;
    color: var(--text-tertiary) !important;
    font-size: 0.8125rem !important;
    text-transform: uppercase !important;
}

.stDataFrame td {
    background-color: var(--bg-elevated) !important;
    color: var(--text-secondary) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
}

.streamlit-expanderContent {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
}

/* Alerts */
div[data-testid="stInfo"] {
    background-color: rgba(59, 130, 246, 0.15) !important;
    border: 1px solid #3B82F6 !important;
    color: var(--text-primary) !important;
}

div[data-testid="stSuccess"] {
    background-color: rgba(34, 197, 94, 0.15) !important;
    border: 1px solid #22C55E !important;
    color: var(--text-primary) !important;
}

div[data-testid="stWarning"] {
    background-color: rgba(245, 158, 11, 0.15) !important;
    border: 1px solid #F59E0B !important;
    color: var(--text-primary) !important;
}

div[data-testid="stError"] {
    background-color: rgba(239, 68, 68, 0.15) !important;
    border: 1px solid #EF4444 !important;
    color: var(--text-primary) !important;
}

/* Progress */
.stProgress > div > div {
    background-color: var(--border-default) !important;
}

.stProgress > div > div > div {
    background-color: var(--accent) !important;
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
    border-radius: 4px !important;
}

/* Dividers */
hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
"""

# =============================================================================
# DROP-IN REPLACEMENT FUNCTION
# =============================================================================

def apply_ce_africa_branding():
    """
    Apply the Linear-style CE Africa theme.
    
    This is a DROP-IN REPLACEMENT for your existing apply_ce_africa_branding().
    Just change the import and everything else stays the same.
    """
    st.markdown(LINEAR_CSS, unsafe_allow_html=True)


# Alias for new projects
apply_theme = apply_ce_africa_branding


def configure_page(title: str = "CE Africa Valuation Platform"):
    """
    Configure page settings and apply theme in one call.
    """
    st.set_page_config(
        page_title=title,
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_theme()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_currency(value: float, prefix: str = 'R') -> str:
    """Format number as currency."""
    if abs(value) >= 1_000_000_000:
        return f"{prefix} {value / 1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"{prefix} {value / 1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{prefix} {value / 1_000:.1f}K"
    else:
        return f"{prefix} {value:,.0f}"


def format_percent(value: float, decimals: int = 1) -> str:
    """Format number as percentage."""
    return f"{value:.{decimals}f}%"


# =============================================================================
# UI COMPONENT HELPERS (for ui_components.py compatibility)
# =============================================================================

def badge(text: str, variant: str = 'neutral') -> str:
    """Create a badge HTML string."""
    color_map = {
        'success': COLORS['success'],
        'warning': COLORS['warning'],
        'error': COLORS['error'],
        'info': COLORS['info'],
        'neutral': COLORS['text_tertiary'],
    }
    color = color_map.get(variant, COLORS['text_tertiary'])
    return f'<span style="background: {color}22; color: {color}; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 500;">{text}</span>'


def stat_card(title: str, value: Any, subtitle: str = None, trend: str = None) -> str:
    """Create a stat card HTML string."""
    trend_html = f'<div style="color: {COLORS["text_tertiary"]}; font-size: 0.75rem;">{trend}</div>' if trend else ''
    subtitle_html = f'<div style="color: {COLORS["text_tertiary"]}; font-size: 0.875rem;">{subtitle}</div>' if subtitle else ''
    return f'''
    <div style="background: {COLORS["bg_surface"]}; padding: 1rem; border-radius: 8px; border: 1px solid {COLORS["border_subtle"]};">
        <div style="color: {COLORS["text_tertiary"]}; font-size: 0.875rem; margin-bottom: 0.5rem;">{title}</div>
        <div style="color: {COLORS["text_primary"]}; font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem;">{value}</div>
        {subtitle_html}
        {trend_html}
    </div>
    '''


def section_header(title: str, subtitle: str = None) -> None:
    """Render a section header."""
    st.markdown(f'### {title}')
    if subtitle:
        st.caption(subtitle)


def empty_state(title: str, description: str = None, action_label: str = None, action_key: str = None) -> None:
    """Render an empty state component."""
    st.markdown(f'''
    <div style="text-align: center; padding: 3rem 2rem; color: {COLORS["text_tertiary"]};">
        <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS["text_primary"]}; margin-bottom: 0.5rem;">{title}</div>
        {f'<div style="margin-bottom: 1.5rem;">{description}</div>' if description else ''}
    </div>
    ''', unsafe_allow_html=True)
    if action_label and action_key:
        st.button(action_label, key=action_key, type="primary", use_container_width=True)
