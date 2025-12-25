"""
Forecast Section Component - AI ASSUMPTIONS INTEGRATION + 3-STATEMENT MODEL
============================================================================
Date: December 14, 2025
Version: 9.2 - Sprint 16+ (Manufacturing Integration & MC Valuation)

NEW IN v9.2 (Sprint 16+ - Full Manufacturing & Financial Statement Enhancement):
1. ✅ Manufacturing Strategy Integration in forecast
2. ✅ Toggle to include/exclude manufacturing impact
3. ✅ COGS split: Buy vs Make portions in Income Statement
4. ✅ Manufacturing overhead, depreciation, commissioning costs
5. ✅ Monte Carlo Valuation Range (P10/P50/P90)
6. ✅ Valuation distribution histogram
7. ✅ Balance Sheet: PPE split (Other vs Manufacturing)
8. ✅ Balance Sheet: Inventory split (Raw Materials vs Finished Goods)
9. ✅ Cash Flow: Manufacturing investments itemized separately
10. ✅ Historical data shown alongside forecast (Actual vs Forecast color coding)
11. ✅ Working Capital from manufacturing flows to inventory, not commissioning

NEW IN v9.1 (Sprint 16 - 3-Statement Financial Model):
1. ✅ Balance Sheet projection from forecast data
2. ✅ Cash Flow Statement projection
3. ✅ Sub-tabs for Income Statement, Balance Sheet, Cash Flow
4. ✅ Manufacturing CAPEX line items (ready for integration)
5. ✅ Working capital calculations (DSO/DIO/DPO based)

NEW IN v9.0 (Sprint 15 - AI Assumptions Integration):
1. ✅ Forecast pulls from AI Assumptions - Uses saved assumptions from AI engine
2. ✅ Monte Carlo uses fitted distributions - sample_from_assumptions() API
3. ✅ Assumption status indicators - Shows which assumptions are being used
4. ✅ AI distribution toggle - Choose between AI distributions or CV-based
5. ✅ Source tracking - Results show assumption source (AI/Manual/Default)

PREVIOUS VERSIONS:
- v8.4: Monte Carlo Tab Enhancement (Sprint 12.5)
- v8.3: Expense Categories, historic_expense_categories table
- v8.2: Historical Financials column mapping
- v8.1: Pipeline fix, Valuation Engine

ALL PREVIOUS FUNCTIONALITY PRESERVED:
- Income statement formatting with all detail levels
- Monte Carlo simulation with full percentiles
- Snapshot save/load
- All chart types (revenue, profitability, margin)
- Export functionality
- Annual aggregation
- DCF Valuation with sensitivity matrix

INSTALLATION:
1. Ensure ai_assumptions_engine.py is in components/
2. Replace components/forecast_section.py with this file
3. Restart application: streamlit run app_refactored.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Any, Optional, Tuple
import traceback
from dataclasses import dataclass, field
import os
import urllib.request
import urllib.error
from urllib.parse import urljoin


# =============================================================================
# AI ASSUMPTIONS ENGINE INTEGRATION (NEW IN v9.0)
# =============================================================================

try:
    from components.ai_assumptions_engine import (
        get_saved_assumptions,
        get_assumption_value,
        get_assumption_distribution,
        get_manufacturing_assumptions,
        sample_from_assumptions,
        AssumptionsSet,
        DistributionParams,
        generate_distribution_samples,
        aggregate_detailed_line_items_to_summary
    )
    AI_ASSUMPTIONS_AVAILABLE = True
except ImportError:
    AI_ASSUMPTIONS_AVAILABLE = False
    
    # Stub classes/functions if not available
    class AssumptionsSet:
        assumptions_saved = False
        assumptions = {}
    
    class DistributionParams:
        pass
    
    def get_saved_assumptions(db, scenario_id):
        return None
    
    def get_assumption_value(assumptions_set, assumption_id, default=0.0):
        return default
    
    def get_assumption_distribution(assumptions_set, assumption_id):
        return None
    
    def get_manufacturing_assumptions(assumptions_set):
        return {}
    
    def sample_from_assumptions(assumptions_set, n_samples=1000):
        return {}
    
    def generate_distribution_samples(params, n):
        return np.zeros(n)
    
    def aggregate_detailed_line_items_to_summary(db, scenario_id: str, user_id: str = None):
        return pd.DataFrame()


# =============================================================================
# COLOR CONSTANTS - Professional Theme
# =============================================================================
GOLD = "#D4A537"
GOLD_LIGHT = "rgba(212, 165, 55, 0.1)"
GOLD_DARK = "#B8962E"
DARK_BG = "#1E1E1E"
DARKER_BG = "#0E1117"
BORDER_COLOR = "#404040"
TEXT_MUTED = "#888888"
TEXT_WHITE = "#FFFFFF"
GREEN = "#10b981"
RED = "#ef4444"
BLUE = "#3b82f6"
BLUE_LIGHT = "rgba(59, 130, 246, 0.1)"

CHART_COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'confidence_band': 'rgba(37, 99, 235, 0.2)',
    'historical': '#64748b'
}


# =============================================================================
# VALUATION ENGINE (NEW IN 8.1)
# =============================================================================

@dataclass
class ValuationResult:
    """Structure for valuation outputs."""
    enterprise_value: float = 0.0
    equity_value: float = 0.0
    terminal_value_perpetuity: float = 0.0
    terminal_value_multiple: float = 0.0
    implied_share_price: float = 0.0
    pv_discrete_cf: float = 0.0
    pv_terminal_value: float = 0.0
    tv_as_pct_ev: float = 0.0
    implied_ev_ebitda: float = 0.0
    implied_ev_revenue: float = 0.0
    dcf_table: pd.DataFrame = field(default_factory=pd.DataFrame)


class ValuationEngine:
    """DCF Valuation Engine for CE Africa."""
    
    def __init__(self, assumptions: Dict[str, Any]):
        self.assumptions = assumptions
        self.wacc = self._get_rate('wacc', 0.12)
        self.tax_rate = self._get_rate('tax_rate', 0.27)
        self.g_rate = self._get_rate('terminal_growth_rate', 0.03)
        self.exit_multiple = float(assumptions.get('exit_multiple_ebitda', 5.0))
        self.capex_pct = self._get_rate('capex_pct_revenue', 0.05)
        self.depreciation_pct = self._get_rate('depreciation_pct_revenue', 0.03)
        self.nwc_pct = self._get_rate('nwc_pct_revenue', 0.10)
    
    def _get_rate(self, key: str, default: float) -> float:
        value = self.assumptions.get(key, default)
        if value is None:
            return default
        value = float(value)
        return value / 100 if value > 1 else value
    
    def calculate_valuation(self, forecast_df: pd.DataFrame, net_debt: float = 0.0) -> ValuationResult:
        if forecast_df is None or forecast_df.empty:
            return ValuationResult()
        
        df = forecast_df.copy()
        
        # Standardize columns
        if 'total_revenue' in df.columns and 'revenue_total' not in df.columns:
            df['revenue_total'] = df['total_revenue']
        if 'revenue' in df.columns and 'revenue_total' not in df.columns:
            df['revenue_total'] = df['revenue']
        
        if 'revenue_total' not in df.columns or df['revenue_total'].sum() == 0:
            return ValuationResult()
        
        # Calculate EBITDA if not present
        if 'ebitda' not in df.columns:
            if 'total_gross_profit' in df.columns and 'total_opex' in df.columns:
                df['ebitda'] = df['total_gross_profit'] - df['total_opex']
            elif 'gross_profit' in df.columns and 'opex' in df.columns:
                df['ebitda'] = df['gross_profit'] - df['opex']
            else:
                df['ebitda'] = df['revenue_total'] * 0.15
        
        # D&A and CapEx
        if 'depreciation' not in df.columns:
            df['depreciation'] = df['revenue_total'] * self.depreciation_pct
        if 'capex' not in df.columns:
            df['capex'] = df['revenue_total'] * self.capex_pct
        
        # EBIT and NOPAT
        df['ebit'] = df['ebitda'] - df['depreciation']
        # Vectorized: Replace lambda with vectorized operation
        df['tax_expense'] = np.where(df['ebit'] > 0, df['ebit'] * self.tax_rate, 0)
        df['nopat'] = df['ebit'] - df['tax_expense']
        
        # Working Capital
        df['nwc_balance'] = df['revenue_total'] * self.nwc_pct
        df['delta_nwc'] = df['nwc_balance'].diff().fillna(df['nwc_balance'].iloc[0] * 0.1)
        
        # UFCF
        df['ufcf'] = df['nopat'] + df['depreciation'] - df['capex'] - df['delta_nwc']
        
        # Discount factors
        df['period_index'] = np.arange(1, len(df) + 1)
        df['discount_factor'] = 1 / ((1 + self.wacc) ** (df['period_index'] / 12))
        df['pv_ufcf'] = df['ufcf'] * df['discount_factor']
        
        sum_pv_discrete = df['pv_ufcf'].sum()
        
        # Terminal Value
        n_months = min(12, len(df))
        last_period = df.tail(n_months)
        final_year_ufcf = last_period['ufcf'].sum() * (12 / n_months)
        final_year_ebitda = last_period['ebitda'].sum() * (12 / n_months)
        final_discount_factor = df['discount_factor'].iloc[-1]
        
        # Gordon Growth
        if self.wacc <= self.g_rate:
            tv_perpetuity = 0.0
        else:
            tv_perpetuity = (final_year_ufcf * (1 + self.g_rate)) / (self.wacc - self.g_rate)
        
        pv_tv_perpetuity = tv_perpetuity * final_discount_factor
        
        # Exit Multiple
        tv_multiple = final_year_ebitda * self.exit_multiple
        
        # Enterprise & Equity Value
        enterprise_value = sum_pv_discrete + pv_tv_perpetuity
        equity_value = enterprise_value - net_debt
        
        # Metrics
        total_revenue = df['revenue_total'].sum()
        total_ebitda = df['ebitda'].sum()
        tv_pct = (pv_tv_perpetuity / enterprise_value * 100) if enterprise_value > 0 else 0
        ev_ebitda = (enterprise_value / (total_ebitda / (len(df) / 12))) if total_ebitda > 0 else 0
        ev_revenue = (enterprise_value / (total_revenue / (len(df) / 12))) if total_revenue > 0 else 0
        
        return ValuationResult(
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            terminal_value_perpetuity=tv_perpetuity,
            terminal_value_multiple=tv_multiple,
            pv_discrete_cf=sum_pv_discrete,
            pv_terminal_value=pv_tv_perpetuity,
            tv_as_pct_ev=tv_pct,
            implied_ev_ebitda=ev_ebitda,
            implied_ev_revenue=ev_revenue,
            dcf_table=df
        )
    
    def run_sensitivity(self, forecast_df: pd.DataFrame, net_debt: float = 0.0) -> pd.DataFrame:
        wacc_values = [0.08, 0.10, 0.12, 0.14, 0.16]
        growth_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        
        results = np.zeros((len(wacc_values), len(growth_values)))
        
        for i, w in enumerate(wacc_values):
            for j, g in enumerate(growth_values):
                temp = self.assumptions.copy()
                temp['wacc'] = w
                temp['terminal_growth_rate'] = g
                engine = ValuationEngine(temp)
                result = engine.calculate_valuation(forecast_df, net_debt)
                results[i, j] = result.enterprise_value
        
        return pd.DataFrame(
            results,
            index=[f"{w*100:.0f}%" for w in wacc_values],
            columns=[f"{g*100:.0f}%" for g in growth_values]
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def inject_custom_css():
    """Inject custom CSS for the forecast section."""
    st.markdown(f"""
    <style>
    .forecast-metric {{
        background: linear-gradient(135deg, {DARK_BG}, {DARKER_BG});
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }}
    .forecast-metric-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {GOLD};
    }}
    .forecast-metric-label {{
        font-size: 0.85rem;
        color: {TEXT_MUTED};
    }}
    .section-header {{
        color: {GOLD};
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {BORDER_COLOR};
    }}
    </style>
    """, unsafe_allow_html=True)


def format_currency(value: float, decimals: int = 0) -> str:
    """Format value as South African Rand currency."""
    if pd.isna(value) or value is None:
        return "-"
    if abs(value) >= 1_000_000_000:
        return f"R {value/1_000_000_000:,.{decimals}f}B"
    elif abs(value) >= 1_000_000:
        return f"R {value/1_000_000:,.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"R {value/1_000:,.{decimals}f}K"
    else:
        return f"R {value:,.{decimals}f}"


# Helper functions for replacing lambdas (Sprint 17.5)
def format_period_view_label(value: str) -> str:
    """Format period view label for radio buttons."""
    return "Monthly" if value == "monthly" else "Annual"

def format_annual_view_label(value: str) -> str:
    """Format annual view label for radio buttons."""
    return "Annual" if value == "annual" else "Monthly"

def get_impact_range_key(impact_dict: Dict[str, Any]) -> float:
    """Get range key for sorting impacts."""
    return impact_dict.get('range', 0)


# =============================================================================
# TABLE NAME CONSTANTS (Sprint 17.5 - Standardized Naming)
# =============================================================================

# Database table names - standardized for consistency
TABLE_ASSUMPTIONS = 'assumptions'
TABLE_HISTORIC_FINANCIALS = 'historic_financials'
TABLE_HISTORICAL_FINANCIALS = 'historical_financials'  # Fallback/alternative name
TABLE_HISTORIC_EXPENSE_CATEGORIES = 'historic_expense_categories'
TABLE_MACHINE_INSTANCES = 'machine_instances'
TABLE_INSTALLED_BASE = 'installed_base'
TABLE_WEAR_PROFILES = 'wear_profiles'
TABLE_EXPENSE_ASSUMPTIONS = 'expense_assumptions'
TABLE_FORECAST_SNAPSHOTS = 'forecast_snapshots'
TABLE_PROSPECTS = 'prospects'


# =============================================================================
# ERROR HANDLING HELPERS (Sprint 17.5 - Standardized Error Handling)
# =============================================================================

def handle_database_error(operation: str, error: Exception, show_error: bool = True) -> None:
    """
    Standardized database error handling.
    
    Args:
        operation: Description of the operation that failed
        error: The exception that occurred
        show_error: Whether to display error to user
    """
    error_msg = f"Database error during {operation}: {str(error)}"
    if show_error:
        st.error(error_msg)
    # Log error (could be extended with proper logging framework)
    print(f"ERROR: {error_msg}")

def handle_data_loading_error(operation: str, error: Exception, return_value=None, show_warning: bool = True):
    """
    Standardized data loading error handling.
    
    Args:
        operation: Description of the operation that failed
        error: The exception that occurred
        return_value: Value to return on error (default: None or empty DataFrame)
        show_warning: Whether to show warning to user
    
    Returns:
        return_value or appropriate default
    """
    if show_warning:
        st.warning(f"Could not load {operation}: {str(error)}")
    print(f"WARNING: Failed to load {operation}: {str(error)}")
    return return_value if return_value is not None else (pd.DataFrame() if 'DataFrame' in str(type(return_value)) else None)


def format_value(value: float, is_negative_expense: bool = False) -> Tuple[str, str]:
    """
    Format a value for display in financial statements.
    Returns (formatted_string, css_class)
    """
    if pd.isna(value) or value is None:
        return "-", "muted"
    
    # For expenses shown as negative
    display_value = -abs(value) if is_negative_expense else value
    
    if abs(value) >= 1_000_000:
        formatted = f"R {display_value/1_000_000:,.1f}M"
    elif abs(value) >= 1_000:
        formatted = f"R {display_value/1_000:,.0f}K"
    else:
        formatted = f"R {display_value:,.0f}"
    
    # Determine CSS class
    if display_value < 0:
        css_class = "negative"
        formatted = f"({formatted.replace('-', '').replace('R ', 'R')})"
    else:
        css_class = ""
    
    return formatted, css_class


def metric_card(label: str, value: str, delta: str = None):
    """Render a metric card."""
    delta_html = f'<div style="color: {TEXT_MUTED}; font-size: 0.75rem;">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="forecast-metric">
        <div class="forecast-metric-value">{value}</div>
        <div class="forecast-metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = None):
    """Render a section header."""
    sub = f'<p style="color: {TEXT_MUTED}; margin: 0; font-size: 0.85rem;">{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div style="margin: 1.5rem 0 1rem 0;">
        <h3 style="color: {GOLD}; margin: 0; font-size: 1.2rem;">{title}</h3>
        {sub}
    </div>
    """, unsafe_allow_html=True)


def alert_box(message: str, alert_type: str = "info"):
    """Render a styled alert box."""
    colors = {
        "info": (BLUE, BLUE_LIGHT),
        "warning": ("#f59e0b", "rgba(245, 158, 11, 0.1)"),
        "error": (RED, "rgba(239, 68, 68, 0.1)"),
        "success": (GREEN, "rgba(16, 185, 129, 0.1)")
    }
    color, bg = colors.get(alert_type, colors["info"])
    st.markdown(f"""
    <div style="
        background: {bg};
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    ">
        <span style="color: {color};">{message}</span>
    </div>
    """, unsafe_allow_html=True)


def empty_state(title: str, message: str, icon: str = "📊", button_label: str = None, button_key: str = None):
    """Render an empty state placeholder."""
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 3rem;
        background: {GOLD_LIGHT};
        border: 1px dashed {BORDER_COLOR};
        border-radius: 8px;
        margin: 2rem 0;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="color: {GOLD}; margin: 0 0 0.5rem 0;">{title}</h3>
        <p style="color: {TEXT_MUTED};">{message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if button_label and button_key:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            return st.button(button_label, key=button_key, use_container_width=True)
    return False


# =============================================================================
# DATA LOADING - FIXED WITH FALLBACK
# =============================================================================

def load_assumptions(db, scenario_id: str, force_refresh: bool = False) -> Optional[Dict]:
    """
    Load scenario assumptions from database with caching.
    
    IMPORTANT: Assumptions are ALWAYS saved to database for persistence across app restarts.
    Session state cache is only for performance optimization within a session.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        force_refresh: If True, bypass cache and load fresh from database
    
    Returns:
        Assumptions dictionary from database
    """
    cache_key = f'assumptions_{scenario_id}'
    
    # Always load from database first on app start (when cache doesn't exist)
    # or when force_refresh is True
    if force_refresh or cache_key not in st.session_state:
        try:
            result = db.client.table(TABLE_ASSUMPTIONS).select('data').eq('scenario_id', scenario_id).execute()
            if result.data:
                assumptions = result.data[0].get('data', {})
                # Cache the result for performance
                st.session_state[cache_key] = assumptions
                return assumptions
            # No assumptions found - clear cache if it exists
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            return None
        except Exception as e:
            st.error(f"Error loading assumptions from database: {e}")
            # If DB load fails, try cache as fallback
            if cache_key in st.session_state:
                return st.session_state[cache_key]
            return None
    
    # Use cache if available (performance optimization)
    return st.session_state.get(cache_key)


def load_historical_expense_categories(db, scenario_id: str) -> pd.DataFrame:
    """
    Load historical expense category breakdown with caching (Sprint 17.5 optimization).
    
    Table: historic_expense_categories
    Columns: month, personnel, logistics, professional, facilities, depreciation, etc.
    """
    # Check cache first
    cache_key = f'historical_expense_categories_{scenario_id}'
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    try:
        result = db.client.table(TABLE_HISTORIC_EXPENSE_CATEGORIES).select('*').eq(
            'scenario_id', scenario_id
        ).order('month').execute()
        
        if result.data:
            df = pd.DataFrame(result.data)
            if 'month' in df.columns:
                df['month'] = pd.to_datetime(df['month'])
            # Cache the result
            st.session_state[cache_key] = df
            return df
            
    except Exception as e:
        handle_data_loading_error("historical expense categories", e, return_value=pd.DataFrame(), show_warning=False)
    
    df = pd.DataFrame()
    st.session_state[cache_key] = df
    return df


def load_historical_financials(db, scenario_id: str) -> pd.DataFrame:
    """
    Load historical financial data using detailed line items (income statement).
    Aggregates detailed line items to summary format, upsamples annual/YTD to monthly
    using monthly sales pattern when available, and caches the result.
    """
    cache_key = f'historical_financials_{scenario_id}'
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # Aggregate from detailed line items (may be annual/YTD)
    aggregated_df = aggregate_detailed_line_items_to_summary(db, scenario_id)
    # Fallback: if no revenue/COGS found in detailed, try legacy summary table (historic_financials)
    try:
        needs_summary_fallback = aggregated_df.empty
        if not aggregated_df.empty:
            rev_cols = [c for c in aggregated_df.columns if 'revenue' in c.lower()]
            cogs_cols = [c for c in aggregated_df.columns if 'cogs' in c.lower()]
            rev_sum = aggregated_df[rev_cols].sum().sum() if rev_cols else 0
            cogs_sum = aggregated_df[cogs_cols].sum().sum() if cogs_cols else 0
            needs_summary_fallback = (rev_sum == 0 and cogs_sum == 0)
        if needs_summary_fallback and hasattr(db, 'client'):
            resp = db.client.table('historic_financials').select('*').eq('scenario_id', scenario_id).order('month').execute()
            if resp.data:
                summary_df = pd.DataFrame(resp.data)
                if 'month' in summary_df.columns:
                    summary_df['period_date'] = pd.to_datetime(summary_df['month'], errors='coerce')
                rename_map = {
                    'revenue': 'total_revenue',
                    'cogs': 'total_cogs',
                    'gross_profit': 'total_gross_profit',
                    'opex': 'total_opex'
                }
                for old, new in rename_map.items():
                    if old in summary_df.columns and new not in summary_df.columns:
                        summary_df[new] = summary_df[old]
                aggregated_df = summary_df
    except Exception:
        pass
    expenses_df = pd.DataFrame()  # compatibility placeholder
    
    # DIAGNOSTIC: Store historical data info for debugging
    hist_diag = {
        'total_rows': len(aggregated_df) if not aggregated_df.empty else 0,
        'date_range': '',
        'has_expense_detail': False,
        'expense_columns': [],
        'years_present': [],
        'expense_source_columns': [],
        'upsampled': False,
        'upsample_strategy': '',
        'upsample_rows_created': 0,
        'upsample_years': [],
        'used_sales_pattern': False,
        'revenue_cols_patterned': [],
        'opex_cols_prorata': [],
    }
    if not aggregated_df.empty and 'period_date' in aggregated_df.columns:
        aggregated_df['period_date'] = pd.to_datetime(aggregated_df['period_date'])
        min_date = aggregated_df['period_date'].min()
        max_date = aggregated_df['period_date'].max()
        hist_diag['date_range'] = f"{min_date.strftime('%Y-%m') if pd.notna(min_date) else 'N/A'} to {max_date.strftime('%Y-%m') if pd.notna(max_date) else 'N/A'}"
        hist_diag['years_present'] = sorted(aggregated_df['period_date'].dt.year.dropna().unique().tolist())
        
        expense_cols = [c for c in aggregated_df.columns if c.startswith('opex_')]
        hist_diag['expense_columns'] = expense_cols
        hist_diag['has_expense_detail'] = len(expense_cols) > 0 and aggregated_df[expense_cols].sum().sum() > 0

    # If annual/YTD (<=1 row per year), upsample to monthly using sales pattern
    upsampled_df = aggregated_df
    if not aggregated_df.empty:
        sales_pattern = _get_monthly_sales_pattern(db, scenario_id)
        # Load optional YTD overrides saved during import (months covered + months to fill)
        ytd_overrides = {}
        fy_end_month = 12
        try:
            assumptions = {}
            if hasattr(db, "get_scenario_assumptions"):
                assumptions = db.get_scenario_assumptions(scenario_id, None) or {}
            else:
                assumptions = {}
            ip = (assumptions or {}).get("import_period_settings", {})
            ytd_cfg = None
            for k in [
                "historical_income_statement_line_items",
                "historical_cashflow_line_items",
                "historical_balance_sheet_line_items",
            ]:
                cfg_block = (ip.get(k) or {})
                fy_end_month = int(cfg_block.get("fiscal_year_end_month") or fy_end_month)
                cfg = cfg_block.get("ytd")
                if cfg:
                    ytd_cfg = cfg
                    break
            if ytd_cfg and ytd_cfg.get("year"):
                ytd_overrides[int(ytd_cfg["year"])] = {
                    "year_end_month": int(ytd_cfg.get("year_end_month") or fy_end_month),
                    "ytd_end_calendar_month": int(ytd_cfg.get("ytd_end_calendar_month") or 0) or None,
                    "fill_fy_months": [int(m) for m in (ytd_cfg.get("fill_fy_months") or [])],
                }
        except Exception:
            ytd_overrides = {}

        upsampled_df, upsample_diag = _upsample_aggregate_financials(
            aggregated_df,
            sales_pattern=sales_pattern,
            ytd_overrides=ytd_overrides,
            fiscal_year_end_month=fy_end_month,
        )
        hist_diag['upsampled'] = upsample_diag.get('upsampled', False)
        hist_diag['upsample_strategy'] = upsample_diag.get('strategy', '')
        hist_diag['upsample_rows_created'] = upsample_diag.get('rows_created', 0)
        hist_diag['upsample_years'] = upsample_diag.get('years', [])
        hist_diag['used_sales_pattern'] = upsample_diag.get('used_sales_pattern', False)
        hist_diag['revenue_cols_patterned'] = upsample_diag.get('revenue_cols_patterned', [])
        hist_diag['opex_cols_prorata'] = upsample_diag.get('opex_cols_prorata', [])
    
    st.session_state['historical_data_diagnostic'] = hist_diag
    st.session_state[cache_key] = upsampled_df
    return upsampled_df


def validate_historical_import_coverage(
    db,
    scenario_id: str,
    user_id: Optional[str] = None,
    assumptions: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """
    Hard gate before running forecasts:
    - If import period selections exist, verify those months exist in DB for IS/BS/CF line items.
    - For Income Statement, verify revenue totals are non-zero for at least one expected month.
    Returns (ok, messages). Messages are human-readable errors/warnings.
    """
    msgs: List[str] = []
    assumptions = assumptions or {}
    ip = (assumptions or {}).get("import_period_settings") or {}
    if not isinstance(ip, dict) or not ip:
        return True, msgs  # no explicit expectations; don't block

    def _expected_periods(key: str) -> List[str]:
        block = ip.get(key) or {}
        sel = block.get("selected_periods")
        if isinstance(sel, list) and sel:
            # stored as "YYYY-MM"
            return [str(x) for x in sel if isinstance(x, (str, int, float)) and str(x).strip()]
        return []

    def _month_starts(labels: List[str]) -> List[str]:
        out = []
        for lbl in labels or []:
            s = str(lbl).strip()
            if re.match(r"^\d{4}-\d{2}$", s):
                out.append(f"{s}-01")
        return sorted(set(out))

    def _fetch_periods(table: str, month_starts: List[str]) -> List[str]:
        if not month_starts:
            return []
        q = db.client.table(table).select("period_date").eq("scenario_id", scenario_id).in_("period_date", month_starts)
        # prefer user_id where possible, but don't fail if schema/RLS differs
        try:
            if user_id:
                q = q.eq("user_id", user_id)
        except Exception:
            pass
        try:
            resp = q.execute()
        except Exception:
            # fallback without user filter
            resp = db.client.table(table).select("period_date").eq("scenario_id", scenario_id).in_("period_date", month_starts).execute()
        rows = resp.data or []
        if not rows:
            return []
        df = pd.DataFrame(rows)
        if "period_date" not in df.columns:
            return []
        return (
            pd.to_datetime(df["period_date"], errors="coerce")
            .dt.to_period("M")
            .astype(str)
            .dropna()
            .tolist()
        )

    def _revenue_nonzero(table: str, month_starts: List[str]) -> bool:
        if not month_starts:
            return True
        q = (
            db.client.table(table)
            .select("period_date,category,line_item_name,amount")
            .eq("scenario_id", scenario_id)
            .in_("period_date", month_starts)
        )
        try:
            if user_id:
                q = q.eq("user_id", user_id)
        except Exception:
            pass
        try:
            resp = q.execute()
        except Exception:
            resp = (
                db.client.table(table)
                .select("period_date,category,line_item_name,amount")
                .eq("scenario_id", scenario_id)
                .in_("period_date", month_starts)
                .execute()
            )
        rows = resp.data or []
        if not rows:
            return False
        df = pd.DataFrame(rows)
        if df.empty:
            return False
        df["category"] = df.get("category", "").fillna("").astype(str).str.lower()
        df["line_item_name"] = df.get("line_item_name", "").fillna("").astype(str).str.lower()
        df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce").fillna(0.0)
        is_rev = df["category"].str.contains("revenue|sales|income|turnover", na=False) | df["line_item_name"].str.contains("revenue|sales|income|turnover", na=False)
        return float(df.loc[is_rev, "amount"].sum()) != 0.0

    checks = [
        ("historical_income_statement_line_items", "historical_income_statement_line_items", "Income Statement"),
        ("historical_balance_sheet_line_items", "historical_balance_sheet_line_items", "Balance Sheet"),
        ("historical_cashflow_line_items", "historical_cashflow_line_items", "Cash Flow"),
    ]

    ok = True
    for ip_key, table, label in checks:
        expected = _expected_periods(ip_key)
        if not expected:
            continue
        ms = _month_starts(expected)
        found = set(_fetch_periods(table, ms))
        missing = sorted(set(expected) - found)
        if missing:
            ok = False
            msgs.append(f"{label}: missing imported periods in DB: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")

        if table == "historical_income_statement_line_items":
            if not _revenue_nonzero(table, ms):
                ok = False
                msgs.append("Income Statement: revenue totals appear to be zero across expected months (check category/line item labels and amount parsing).")

    return ok, msgs


def _upsample_aggregate_financials(
    df: pd.DataFrame, 
    sales_pattern: Optional[np.ndarray] = None,
    ytd_overrides: Optional[Dict[int, Dict[str, Any]]] = None,
    fiscal_year_end_month: int = 12
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    If data is annual/YTD only (<=1 row per year), upsample to monthly.
    
    Uses sales pattern for revenue-related items if available,
    otherwise falls back to pro-rata (even distribution).
    If the row is YTD (month < 12), annualize to 12 months before splitting,
    so YTD totals (e.g., 10 months) are grossed up to 12 months.
    
    Args:
        df: Financial data DataFrame
        sales_pattern: Optional 12-element array with monthly weights (sums to 1.0)
        
    Returns:
        (upsampled_df, diagnostic_info)
    """
    diag = {
        'upsampled': False, 
        'strategy': '', 
        'rows_created': 0, 
        'years': [],
        'used_sales_pattern': False,
        'revenue_cols_patterned': [],
        'opex_cols_prorata': []
    }
    
    if df.empty or 'period_date' not in df.columns:
        return df, diag
    
    df = df.copy()
    df['period_date'] = pd.to_datetime(df['period_date'], errors='coerce')
    if df['period_date'].isna().all():
        return df, diag
    
    # Check if <=1 row per year (needs upsampling)
    counts = df['period_date'].dt.year.value_counts()
    if counts.max() > 1:
        return df, diag  # already monthly or multi-row
    
    # Identify column types for differentiated upsampling
    non_numeric_cols = [
        'period_date', 'month', 'period_label', 'data_type', 'is_actual', 'is_annual_total',
        'id', 'scenario_id', 'created_at', 'user_id'
    ]
    all_cols = df.columns.tolist()
    numeric_cols = [c for c in all_cols if c not in non_numeric_cols]
    
    # Revenue-related columns (use sales pattern if available)
    revenue_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in 
        ['revenue', 'sales', 'income', 'turnover', 'gross_profit', 'cogs', 'cost_of_sales'])]
    
    # OPEX columns (always pro-rata - more stable month-on-month)
    opex_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in 
        ['opex', 'overhead', 'expense', 'salary', 'personnel', 'admin', 'rent', 'depreciation'])]
    
    # Other numeric columns (use sales pattern for balance sheet items tied to revenue)
    other_cols = [c for c in numeric_cols if c not in revenue_cols and c not in opex_cols]
    
    diag['revenue_cols_patterned'] = revenue_cols
    diag['opex_cols_prorata'] = opex_cols
    
    # Validate or create sales pattern
    if sales_pattern is not None and len(sales_pattern) == 12:
        # Normalize to sum to 1.0
        pattern_sum = np.sum(sales_pattern)
        if pattern_sum > 0:
            sales_pattern = sales_pattern / pattern_sum
            diag['used_sales_pattern'] = True
        else:
            sales_pattern = None
    else:
        sales_pattern = None
    
    rows = []
    ytd_overrides = ytd_overrides or {}

    fiscal_year_end_month = int(fiscal_year_end_month or 12)

    def _fy_end_year(dt: pd.Timestamp) -> int:
        return int(dt.year) if int(dt.month) <= fiscal_year_end_month else int(dt.year) + 1

    for _, row in df.iterrows():
        dt = row['period_date']
        if pd.isna(dt):
            continue

        dt = pd.to_datetime(dt, errors='coerce')
        if pd.isna(dt):
            continue
        
        # Build the 12-month fiscal-year window (month starts) ending at fiscal_year_end_month
        fy_end = pd.Timestamp(_fy_end_year(dt), fiscal_year_end_month, 1)
        fy_start = (fy_end - pd.DateOffset(months=11)).to_period('M').to_timestamp()
        fy_months = pd.date_range(start=fy_start, periods=12, freq='MS')

        # Determine months covered (YTD) based on dt within the fiscal year window
        dt_m = dt.to_period('M').to_timestamp()
        months = int(sum(m <= dt_m for m in fy_months))
        months = max(1, min(months, 12))

        # Optional override: treat FY end year as key and use explicit fill_fy_months and/or ytd_end_calendar_month
        override = ytd_overrides.get(int(_fy_end_year(dt)))
        fill_fy_months = set()
        if override:
            if override.get("ytd_end_calendar_month"):
                # derive months_covered from calendar month within FY
                cal_m = int(override["ytd_end_calendar_month"])
                # guess year for this month within FY span
                candidate = pd.Timestamp(fy_end.year if cal_m <= fiscal_year_end_month else fy_end.year - 1, cal_m, 1)
                if candidate in set(fy_months):
                    months = int(list(fy_months).index(candidate)) + 1
            if override.get("fill_fy_months"):
                fill_fy_months = set(int(x) for x in (override.get("fill_fy_months") or []))

        # If YTD (<12), annualize the numeric totals to 12 months so gross-up occurs
        annualize_factor = 12 / months if months < 12 else 1.0
        
        for idx_m, month_start in enumerate(fy_months, start=1):  # 1..12 fiscal months
            new_row = {}
            
            # Copy non-numeric columns
            for c in non_numeric_cols:
                if c in row.index:
                    new_row[c] = row[c]
            
            # Set period date
            new_row['period_date'] = pd.to_datetime(month_start)
            # Mark actual vs filled forecast months if override provided
            if fill_fy_months:
                new_row['is_actual'] = (idx_m not in fill_fy_months)
                new_row['data_type'] = 'Actual' if new_row['is_actual'] else 'Forecast (YTD fill)'
            
            # Calculate weights
            if sales_pattern is not None:
                # Use full-year pattern; for YTD, still weight across 12 months
                revenue_weight = sales_pattern[int(month_start.month) - 1]
            else:
                # No pattern - use equal weighting
                revenue_weight = 1 / 12
            
            opex_weight = 1 / 12  # OPEX always pro-rata (annualized)
            
            # Apply weights to columns
            for c in numeric_cols:
                val = row[c] if c in row.index else 0
                if pd.isna(val):
                    new_row[c] = 0
                elif c in revenue_cols:
                    new_row[c] = val * annualize_factor * revenue_weight
                elif c in opex_cols:
                    new_row[c] = val * annualize_factor * opex_weight
                else:
                    # Other columns - use sales pattern for balance sheet, pro-rata for rest
                    if 'stock' in c.lower() or 'inventory' in c.lower() or 'receivable' in c.lower():
                        new_row[c] = val * annualize_factor * revenue_weight
                    else:
                        new_row[c] = val * annualize_factor * opex_weight
            
            rows.append(new_row)
        
        diag['years'].append(dt.year)
        diag['rows_created'] += 12
    
    if rows:
        diag['upsampled'] = True
        diag['strategy'] = 'sales_pattern_revenue_prorata_opex' if diag['used_sales_pattern'] else 'pro_rata_monthly'
        upsampled = pd.DataFrame(rows)
        upsampled['period_date'] = pd.to_datetime(upsampled['period_date'])
        return upsampled.sort_values('period_date'), diag
    
    return df, diag


def _get_monthly_sales_pattern(db, scenario_id: str) -> Optional[np.ndarray]:
    """
    Extract monthly sales pattern from available data.
    
    Looks for:
    1. Monthly sales data from sales_orders or similar table
    2. Monthly revenue from historical_income_statement_line_items
    3. Any monthly financial data with revenue columns
    
    Returns:
        12-element array with monthly weights, or None if no pattern found
    """
    try:
        # Prefer: granular sales history (invoice-level). This table is user-scoped (not scenario-scoped).
        if hasattr(db, 'client'):
            try:
                scen = db.client.table('scenarios').select('user_id').eq('id', scenario_id).limit(1).execute()
                scen_user_id = None
                if scen.data and scen.data[0].get('user_id'):
                    scen_user_id = scen.data[0]['user_id']

                if scen_user_id:
                    resp = db.client.table('granular_sales_history').select(
                        'date, quantity, unit_price_sold'
                    ).eq('user_id', scen_user_id).execute()

                    if resp.data and len(resp.data) >= 12:
                        sdf = pd.DataFrame(resp.data)
                        sdf['date'] = pd.to_datetime(sdf['date'], errors='coerce')
                        sdf = sdf.dropna(subset=['date'])
                        if not sdf.empty:
                            sdf['month'] = sdf['date'].dt.month
                            sdf['quantity'] = pd.to_numeric(sdf.get('quantity'), errors='coerce').fillna(0.0)
                            sdf['unit_price_sold'] = pd.to_numeric(sdf.get('unit_price_sold'), errors='coerce').fillna(0.0)
                            sdf['sales'] = sdf['quantity'] * sdf['unit_price_sold']

                            monthly = sdf.groupby('month')['sales'].sum()
                            # Need at least 6 months with activity to avoid noisy patterns
                            active_months = int((monthly > 0).sum())
                            if active_months >= 6:
                                pattern = np.zeros(12)
                                for m in range(1, 13):
                                    pattern[m - 1] = float(monthly.get(m, 0.0))
                                if pattern.sum() > 0:
                                    return pattern / pattern.sum()
            except Exception:
                pass

        # Try 1: Look for monthly sales data in dedicated table
        if hasattr(db, 'client'):
            try:
                response = db.client.table('sales_orders').select(
                    'order_date, total_amount'
                ).eq('scenario_id', scenario_id).execute()
                
                if response.data and len(response.data) > 12:
                    df = pd.DataFrame(response.data)
                    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
                    df['month'] = df['order_date'].dt.month
                    
                    # Aggregate by month
                    monthly = df.groupby('month')['total_amount'].sum()
                    
                    if len(monthly) >= 6:  # Need at least 6 months of data
                        pattern = np.zeros(12)
                        for m in range(1, 13):
                            pattern[m - 1] = monthly.get(m, 0)
                        
                        if pattern.sum() > 0:
                            return pattern / pattern.sum()
            except Exception:
                pass
        
        # Try 2: Look for monthly revenue in line items
        if hasattr(db, 'client'):
            try:
                response = db.client.table('historical_income_statement_line_items').select(
                    'period_date, amount, category'
                ).eq('scenario_id', scenario_id).execute()
                
                if response.data and len(response.data) > 12:
                    df = pd.DataFrame(response.data)
                    df['period_date'] = pd.to_datetime(df['period_date'], errors='coerce')
                    
                    # Filter for revenue
                    revenue_df = df[df['category'].str.lower().str.contains('revenue|sales|income', na=False)]
                    
                    if not revenue_df.empty:
                        revenue_df['month'] = revenue_df['period_date'].dt.month
                        monthly = revenue_df.groupby('month')['amount'].sum()
                        
                        if len(monthly) >= 6:
                            pattern = np.zeros(12)
                            for m in range(1, 13):
                                pattern[m - 1] = monthly.get(m, 0)
                            
                            if pattern.sum() > 0:
                                return pattern / pattern.sum()
            except Exception:
                pass
        
        # Try 3: Use typical mining services seasonality (if no data)
        # Mining services typically have Q4 peak due to project completions
        # and Q1 dip due to holiday slowdowns
        # This is a fallback - user should verify
        # Return None to use pro-rata instead
        
    except Exception:
        pass
    
    return None



# =============================================================================
# AI ASSUMPTIONS HELPER FUNCTIONS (NEW IN v9.0)
# =============================================================================

def get_effective_assumption(ai_assumptions: Optional[Any], 
                             manual_assumptions: Dict,
                             ai_key: str,
                             manual_key: str,
                             default: float) -> Tuple[float, str]:
    """
    Get effective assumption value, preferring AI-derived if available.
    Returns (value, source) tuple where source is 'AI', 'Manual', or 'Default'.
    """
    # First try AI assumptions
    if AI_ASSUMPTIONS_AVAILABLE and ai_assumptions:
        try:
            if hasattr(ai_assumptions, 'assumptions_saved') and ai_assumptions.assumptions_saved:
                if hasattr(ai_assumptions, 'assumptions') and ai_key in ai_assumptions.assumptions:
                    assumption = ai_assumptions.assumptions[ai_key]
                    if hasattr(assumption, 'final_static_value'):
                        value = assumption.final_static_value
                        if value != 0:
                            return (float(value), "AI")
        except Exception:
            pass
    
    # Fall back to manual assumptions
    if manual_assumptions and manual_key in manual_assumptions:
        value = manual_assumptions.get(manual_key)
        if value is not None and isinstance(value, (int, float)):
            return (float(value), "Manual")
    
    return (default, "Default")


def merge_assumptions_with_ai(manual_assumptions: Dict, 
                               ai_assumptions: Optional[Any]) -> Dict[str, Any]:
    """
    Merge manual assumptions with AI-derived assumptions.
    AI assumptions are tracked with '_source' suffix keys.
    Returns merged dict with source tracking.
    """
    merged = manual_assumptions.copy() if manual_assumptions else {}
    
    if not AI_ASSUMPTIONS_AVAILABLE or not ai_assumptions:
        return merged
    
    try:
        if not hasattr(ai_assumptions, 'assumptions_saved') or not ai_assumptions.assumptions_saved:
            return merged
        
        # Map AI assumption IDs to manual assumption keys
        mapping = {
            'revenue_growth_pct': 'revenue_growth_rate',
            'gross_margin_pct': 'gross_margin_liner',
            'gross_margin_liner': 'gross_margin_liner',
            'gross_margin_refurb': 'gross_margin_refurb',
            'opex_pct_revenue': 'opex_as_pct_revenue',
        }
        
        for ai_key, manual_key in mapping.items():
            if hasattr(ai_assumptions, 'assumptions') and ai_key in ai_assumptions.assumptions:
                assumption = ai_assumptions.assumptions[ai_key]
                if hasattr(assumption, 'final_static_value'):
                    value = assumption.final_static_value
                    if value != 0:
                        # Convert percentage if needed
                        if 'pct' in ai_key or 'margin' in ai_key:
                            if value > 1:
                                value = value / 100
                        merged[manual_key] = value
                        merged[f'{manual_key}_source'] = 'AI'
    except Exception:
        pass
    
    return merged


def check_ai_assumptions_status(db, scenario_id: str) -> Tuple[bool, Optional[Any]]:
    """
    Check if AI assumptions are available and saved for this scenario with caching (Sprint 17.5).
    Returns (is_available, assumptions_set).
    """
    if not AI_ASSUMPTIONS_AVAILABLE:
        return False, None
    
    # Check cache first
    cache_key = f'ai_assumptions_status_{scenario_id}'
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    try:
        ai_assumptions = get_saved_assumptions(db, scenario_id)
        if ai_assumptions and hasattr(ai_assumptions, 'assumptions_saved'):
            result = (ai_assumptions.assumptions_saved, ai_assumptions)
            # Cache the result
            st.session_state[cache_key] = result
            return result
        result = (False, ai_assumptions)
        st.session_state[cache_key] = result
        return result
    except Exception:
        result = (False, None)
        st.session_state[cache_key] = result
        return result

def load_forecast_data(db, scenario_id: str, user_id: str = None, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Load all data needed for forecasting with caching (Sprint 17.5 optimization).
    FIXED in 8.1: Removed broken customers join from prospects query.
    NEW in 9.0: Also loads AI-derived assumptions.
    """
    # Check cache first (cache entire data structure)
    cache_key = f'forecast_data_{scenario_id}'
    if not force_refresh and cache_key in st.session_state:
        cached = st.session_state[cache_key]
        # If cached has no machines, fall through to reload
        if cached.get('machines'):
            return cached
    
    data = {
        'machines': [],
        'prospects': [],
        'expenses': [],
        'wear_profiles': {},
        'assumptions': None,
        'ai_assumptions': None,  # NEW in v9.0
        'historical_financials': pd.DataFrame(),
        'data_source': 'machine_instances'
    }
    
    try:
        # Load assumptions (uses its own cache)
        data['assumptions'] = load_assumptions(db, scenario_id)
        
        # NEW in v9.0: Load AI-derived assumptions (cache separately)
        if AI_ASSUMPTIONS_AVAILABLE:
            ai_cache_key = f'ai_assumptions_{scenario_id}'
            if ai_cache_key in st.session_state:
                data['ai_assumptions'] = st.session_state[ai_cache_key]
            else:
                try:
                    data['ai_assumptions'] = get_saved_assumptions(db, scenario_id)
                    st.session_state[ai_cache_key] = data['ai_assumptions']
                except Exception:
                    data['ai_assumptions'] = None
        
        # Load historical financials (uses its own cache)
        data['historical_financials'] = load_historical_financials(db, scenario_id)
        
        # Try machine_instances first (new schema from Sprint 2)
        try:
            machines = db.client.table(TABLE_MACHINE_INSTANCES).select(
                '*, sites(site_name, ore_type_id, customers(customer_name)), '
                'wear_profiles_v2(profile_name, liner_life_months, avg_consumable_revenue, '
                'refurb_interval_months, avg_refurb_revenue, ore_type_id, gross_margin_liner, gross_margin_refurb)'
            ).eq('scenario_id', scenario_id).eq('status', 'Active').execute()
            
            if machines.data and len(machines.data) > 0:
                data['machines'] = machines.data
                data['data_source'] = 'machine_instances'
            else:
                raise Exception("machine_instances empty, trying fallback")
        except:
            # FALLBACK: Query installed_base table (legacy schema)
            try:
                installed_base = db.client.table(TABLE_INSTALLED_BASE).select('*').eq('scenario_id', scenario_id).execute()
                
                if installed_base.data and len(installed_base.data) > 0:
                    # Load wear profiles
                    wear_profiles_result = db.client.table(TABLE_WEAR_PROFILES).select('*').execute()
                    wear_profile_dict = {p.get('machine_model', ''): p for p in (wear_profiles_result.data or [])}
                    
                    # Transform installed_base format to expected format
                    transformed_machines = []
                    for machine in installed_base.data:
                        model = machine.get('machine_model', '') or machine.get('Model', '')
                        profile = wear_profile_dict.get(model, {})
                        
                        transformed_machines.append({
                            'id': machine.get('id'),
                            'machine_id': machine.get('machine_id'),
                            'customer_name': machine.get('customer_name', machine.get('Customer', '')),
                            'site_name': machine.get('site_name', machine.get('Site', '')),
                            'machine_model': model,
                            'commission_date': machine.get('commission_date', machine.get('Commission Date')),
                            'status': machine.get('status', 'Active'),
                            'wear_profiles_v2': {
                                'liner_life_months': profile.get('liner_life_months', 6),
                                'avg_consumable_revenue': profile.get('avg_consumable_revenue', 50000),
                                'refurb_interval_months': profile.get('refurb_interval_months', 36),
                                'avg_refurb_revenue': profile.get('avg_refurb_revenue', 150000),
                                'gross_margin_liner': profile.get('gross_margin_liner', 0.38),
                                'gross_margin_refurb': profile.get('gross_margin_refurb', 0.32)
                            },
                            '_source': 'installed_base'
                        })
                    
                    data['machines'] = transformed_machines
                    data['data_source'] = 'installed_base'
            except Exception as fallback_error:
                pass  # Will be handled below
        
        # =================================================================
        # FIXED in 8.1: Load prospects WITHOUT customer join
        # The old query failed because there's no FK between prospects and customers
        # =================================================================
        try:
            prospects = db.client.table(TABLE_PROSPECTS).select('*').eq('scenario_id', scenario_id).execute()
            data['prospects'] = prospects.data or []
        except Exception as e:
            data['prospects'] = []
        
        # Load expense assumptions
        try:
            expenses = db.client.table(TABLE_EXPENSE_ASSUMPTIONS).select('*').eq('scenario_id', scenario_id).eq('is_active', True).execute()
            data['expenses'] = expenses.data or []
        except:
            data['expenses'] = []
        
    except Exception as e:
        handle_database_error("loading forecast data", e, show_error=True)
    
    # Cache the entire data structure
    st.session_state[cache_key] = data
    return data



def load_snapshots(db, scenario_id: str, limit: int = 10, user_id: Optional[str] = None) -> List[Dict]:
    """Load saved forecast snapshots."""
    try:
        # Prefer API snapshot list if configured and user_id available (thin-client mode)
        api_base = (os.getenv("FORECAST_API_URL") or "").strip().rstrip("/")
        effective_user_id = user_id or st.session_state.get("user_id")
        if api_base and effective_user_id:
            try:
                url = urljoin(api_base + "/", f"v1/scenarios/{scenario_id}/snapshots?user_id={effective_user_id}&limit={int(limit)}")
                _headers = {"Content-Type": "application/json"}
                try:
                    jwt_token = os.getenv("FORECAST_API_JWT", "").strip()
                    if jwt_token:
                        _headers["Authorization"] = f"Bearer {jwt_token}"
                except Exception:
                    pass
                req = urllib.request.Request(url, headers=_headers, method="GET")
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw) if raw else []
                    if isinstance(data, list) and data:
                        return data
            except Exception:
                pass

        result = db.client.table(TABLE_FORECAST_SNAPSHOTS).select('*').eq(
            'scenario_id', scenario_id
        ).order('created_at', desc=True).limit(limit).execute()
        return result.data or []
    except:
        return []


# =============================================================================
# FORECAST ENGINE - FIXED PIPELINE CALCULATION
# =============================================================================

def run_forecast(db, scenario_id: str, user_id: str, progress_callback=None, 
                 manufacturing_scenario=None) -> Dict[str, Any]:
    """
    Run the complete forecast calculation.
    FIXED in 8.1: Pipeline now uses annual_liner_value + refurb_value (both fields).
    NEW in 9.0: Uses AI-derived assumptions when available.
    NEW in 9.2: Manufacturing strategy integration.
    NEW in Sprint 21: Uses extracted ForecastEngine for calculation.
    
    Args:
        db: Database handler
        scenario_id: Current scenario ID
        user_id: Current user ID
        progress_callback: Optional progress callback function
        manufacturing_scenario: Optional IntegrationScenario for manufacturing integration
    """
    # Sprint 21: Use extracted forecast engine
    try:
        from forecast_engine import ForecastEngine
        engine = ForecastEngine()
        
        # Load forecast data (UI function - handles caching)
        if progress_callback:
            progress_callback(0.1, "Loading data...")
        
        data = load_forecast_data(db, scenario_id, user_id)
        
        # Add trend forecast configuration if enabled
        # Get value from session state (set by checkbox widget)
        use_trend_forecast = st.session_state.get('use_trend_forecast', False)
        if use_trend_forecast:
            assumptions_data = db.get_scenario_assumptions(scenario_id, user_id) if hasattr(db, 'get_scenario_assumptions') else {}
            
            # Check for forecast_configs (new format from Trend Forecast tab)
            forecast_configs = assumptions_data.get('forecast_configs', {})
            # Also check legacy trend_forecasts for backward compatibility
            trend_forecasts = assumptions_data.get('trend_forecasts', {})
            
            if forecast_configs or trend_forecasts:
                data['assumptions']['use_trend_forecast'] = True
                # Use forecast_configs if available, otherwise fall back to trend_forecasts
                if forecast_configs:
                    data['assumptions']['forecast_configs'] = forecast_configs
                if trend_forecasts:
                    data['assumptions']['trend_forecasts'] = trend_forecasts
                
                # Add historical data for trend calculation
                # Use the same data loading logic as AI Assumptions (includes aggregation from line items)
                try:
                    from components.ai_assumptions_engine import load_historical_data
                    hist_df = load_historical_data(db, scenario_id, user_id)
                    
                    if not hist_df.empty:
                        # Ensure column names match what forecast engine expects
                        # The forecast engine looks for: revenue, cogs, opex, gross_profit, etc.
                        # Map common column name variations
                        hist_df_mapped = hist_df.copy()
                        
                        # Map total_* columns to base names if base names don't exist
                        column_mapping = {
                            'total_revenue': 'revenue',
                            'total_cogs': 'cogs',
                            'total_opex': 'opex',
                            'total_gross_profit': 'gross_profit'
                        }
                        
                        for old_name, new_name in column_mapping.items():
                            if old_name in hist_df_mapped.columns and new_name not in hist_df_mapped.columns:
                                hist_df_mapped[new_name] = hist_df_mapped[old_name]
                        
                        data['historic_financials'] = hist_df_mapped
                        
                        # Add historical revenue series for legacy support
                        revenue_col = None
                        if 'revenue' in hist_df_mapped.columns:
                            revenue_col = 'revenue'
                        elif 'total_revenue' in hist_df_mapped.columns:
                            revenue_col = 'total_revenue'
                        
                        if revenue_col:
                            if 'month' in hist_df_mapped.columns:
                                data['historical_revenue'] = hist_df_mapped.set_index('month')[revenue_col].sort_index()
                            elif 'period_date' in hist_df_mapped.columns:
                                data['historical_revenue'] = hist_df_mapped.set_index('period_date')[revenue_col].sort_index()
                            else:
                                data['historical_revenue'] = hist_df_mapped[revenue_col]
                except Exception as e:
                    # Fallback to old method if new method fails
                    try:
                        hist_financials = db.get_historic_financials(scenario_id)
                        if hist_financials:
                            hist_df = pd.DataFrame(hist_financials)
                            data['historic_financials'] = hist_df
                            if 'revenue' in hist_df.columns:
                                if 'month' in hist_df.columns:
                                    data['historical_revenue'] = hist_df.set_index('month')['revenue'].sort_index()
                                else:
                                    data['historical_revenue'] = hist_df['revenue']
                    except Exception:
                        pass
        
        # Verify forecast_configs are loaded (silent check - errors will be shown if forecast fails)
        
        # Run forecast using engine
        results = engine.run_forecast(data, manufacturing_scenario, progress_callback)
        
        # Add data_source to results (from load_forecast_data)
        results['data_source'] = data.get('data_source', 'unknown')
        
        # Add forecast method to results
        if st.session_state.get('use_trend_forecast', False):
            results['forecast_method'] = 'trend_based'
        else:
            results['forecast_method'] = 'pipeline_based'
        
        # NEW: Generate detailed line item forecasts
        try:
            from components.detailed_line_item_forecast import (
                generate_detailed_forecast,
                save_forecast_line_items
            )
            
            if progress_callback:
                progress_callback(0.9, "Generating detailed line item forecasts...")
            
            # Get forecast period count and start date
            forecast_periods = len(results.get('timeline', []))
            if forecast_periods > 0:
                # Determine start date (next month after last historical or current date)
                from dateutil.relativedelta import relativedelta
                start_date = datetime.now().replace(day=1) + relativedelta(months=1)
                
                # Get assumptions for line item configurations
                assumptions_data = db.get_scenario_assumptions(scenario_id, user_id) if hasattr(db, 'get_scenario_assumptions') else {}
                line_item_configs = assumptions_data.get('line_item_configs', {})
                
                # Also get AI-generated line item assumptions
                line_item_assumptions = {}
                if 'line_item_assumptions' in assumptions_data:
                    line_item_assumptions = assumptions_data['line_item_assumptions']
                
                # Generate detailed forecasts for each statement type
                for statement_type in ['income_statement', 'balance_sheet', 'cash_flow']:
                    try:
                        # Build source forecast DataFrame for correlations
                        source_df = None
                        if statement_type == 'income_statement' and 'timeline' in results:
                            # Create source forecast with revenue for correlations
                            rev_arr = results.get('revenue', {}).get('total') or []
                            # Align lengths defensively (avoid ValueError: All arrays must be same length)
                            min_len = min(
                                forecast_periods,
                                len(rev_arr) if hasattr(rev_arr, '__len__') else 0
                            )
                            if min_len <= 0:
                                min_len = forecast_periods
                                rev_arr = [0] * forecast_periods
                            else:
                                rev_arr = list(rev_arr)[:min_len]
                            
                            source_df = pd.DataFrame({
                                'period_date': pd.date_range(start=start_date, periods=min_len, freq='MS'),
                                'revenue': rev_arr,
                                'total_revenue': rev_arr
                            })
                        
                            detailed_forecast = generate_detailed_forecast(
                            db=db,
                            scenario_id=scenario_id,
                            user_id=user_id,
                            statement_type=statement_type,
                            forecast_periods=len(source_df) if source_df is not None else forecast_periods,
                            start_date=start_date,
                            source_forecast=source_df,
                            assumptions={
                                'line_item_configs': line_item_configs,
                                'line_item_assumptions': line_item_assumptions
                            }
                        )
                        
                        if not detailed_forecast.empty:
                            # Save to database (without snapshot_id initially, will be set when snapshot is saved)
                            save_forecast_line_items(
                                db=db,
                                scenario_id=scenario_id,
                                user_id=user_id,
                                snapshot_id=None,  # Will be updated when snapshot is saved
                                statement_type=statement_type,
                                forecast_df=detailed_forecast
                            )
                            
                            # Store in results for immediate use
                            if 'detailed_line_items' not in results:
                                results['detailed_line_items'] = {}
                            results['detailed_line_items'][statement_type] = detailed_forecast
                    except Exception as e:
                        st.warning(f"Could not generate detailed {statement_type} forecasts: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        except ImportError:
            # Detailed line item forecasting not available
            pass
        except Exception as e:
            st.warning(f"Detailed line item forecasting failed: {e}")
            import traceback
            st.code(traceback.format_exc())
        
        return results
    except ImportError:
        # Fallback to original implementation if engine not available
        pass
    except Exception as e:
        # If engine fails, fallback to original
        st.warning(f"Forecast engine error, using fallback: {e}")
    
    # Original implementation (fallback)
    results = {
        'success': False,
        'error': None,
        'timeline': [],
        'timeline_dates': [],
        'revenue': {
            'consumables': [],
            'refurb': [],
            'pipeline': [],
            'total': []
        },
        'costs': {
            'cogs': [],
            'cogs_buy': [],              # NEW: Buy portion of COGS
            'cogs_make': [],             # NEW: Manufacturing direct COGS (materials + labor)
            'mfg_variable_overhead': [], # NEW v3.2: Variable manufacturing overhead
            'mfg_fixed_overhead': [],    # NEW v3.2: Fixed manufacturing overhead  
            'mfg_overhead': [],          # Combined overhead (for backward compatibility)
            'mfg_depreciation': [],      # NEW: Manufacturing depreciation
            'commissioning': [],         # NEW: Commissioning costs
            'opex': [],
            'total': []
        },
        'profit': {
            'gross': [],
            'ebit': []
        },
        'summary': {},
        'by_customer': {},
        'data_source': None,
        'historical_included': False,
        'assumptions_source': 'Manual',  # NEW in v9.0
        'ai_assumptions_used': [],  # NEW in v9.0
        'manufacturing_included': manufacturing_scenario is not None,  # NEW in v9.2
        'manufacturing_strategy': manufacturing_scenario.strategy if manufacturing_scenario else None  # NEW in v9.2
    }
    
    if progress_callback:
        progress_callback(0.1, "Loading data...")
    
    # Load all data (with fallback)
    data = load_forecast_data(db, scenario_id, user_id, force_refresh=True)
    results['data_source'] = data.get('data_source', 'unknown')
    
    if not data['assumptions']:
        results['error'] = "No assumptions configured. Please complete setup first."
        return results
    
    if not data['machines']:
        results['error'] = f"No machines found in fleet. Please import fleet data first. (Checked: {data.get('data_source', 'both tables')})"
        return results
    
    assumptions = data['assumptions']
    ai_assumptions = data.get('ai_assumptions')  # NEW in v9.0
    forecast_months = assumptions.get('forecast_duration_months', 60)
    
    # NEW in v9.0: Get margins using AI assumptions with fallback
    margin_consumable, margin_source_c = get_effective_assumption(
        ai_assumptions, assumptions, 
        'gross_margin_liner', 'margin_consumable_pct', 0.38
    )
    if margin_source_c == 'Default':
        # Try alternate key
        margin_consumable_raw = assumptions.get('gross_margin_liner') or 38
        margin_consumable = margin_consumable_raw / 100 if margin_consumable_raw > 1 else margin_consumable_raw
        margin_source_c = 'Manual'
    elif margin_consumable > 1:
        margin_consumable = margin_consumable / 100
    
    margin_refurb, margin_source_r = get_effective_assumption(
        ai_assumptions, assumptions,
        'gross_margin_refurb', 'margin_refurb_pct', 0.32
    )
    if margin_source_r == 'Default':
        margin_refurb_raw = assumptions.get('gross_margin_refurb') or 32
        margin_refurb = margin_refurb_raw / 100 if margin_refurb_raw > 1 else margin_refurb_raw
        margin_source_r = 'Manual'
    elif margin_refurb > 1:
        margin_refurb = margin_refurb / 100
    
    # Track AI assumptions used
    if margin_source_c == 'AI':
        results['ai_assumptions_used'].append('gross_margin_liner')
    if margin_source_r == 'AI':
        results['ai_assumptions_used'].append('gross_margin_refurb')
    
    if results['ai_assumptions_used']:
        results['assumptions_source'] = 'AI + Manual'
    
    inflation_raw = assumptions.get('inflation_rate', 5) or 5
    inflation = inflation_raw / 100 if inflation_raw > 1 else inflation_raw
    
    if progress_callback:
        progress_callback(0.2, "Generating timeline...")
    
    # Generate timeline
    start_date = datetime.now().replace(day=1)
    timeline = pd.date_range(start=start_date, periods=forecast_months, freq='MS')
    results['timeline'] = timeline.strftime('%Y-%m').tolist()
    results['timeline_dates'] = timeline.strftime('%Y-%m-%d').tolist()
    
    # Initialize arrays
    n_months = len(timeline)
    consumable_rev = np.zeros(n_months)
    refurb_rev = np.zeros(n_months)
    pipeline_rev = np.zeros(n_months)
    
    if progress_callback:
        progress_callback(0.3, f"Calculating fleet revenue ({len(data['machines'])} machines)...")
    
    # Calculate fleet revenue
    # FIXED in v9.1: Use smooth monthly revenue instead of event-based spikes
    # This matches how historical data is recorded (continuous monthly)
    for machine in data['machines']:
        profile = machine.get('wear_profiles_v2', {})
        if not profile:
            # Use industry defaults
            profile = {
                'liner_life_months': 6,
                'avg_consumable_revenue': 50000,
                'refurb_interval_months': 36,
                'avg_refurb_revenue': 150000,
                'gross_margin_liner': 0.38,
                'gross_margin_refurb': 0.32
            }
        
        liner_life = max(1, profile.get('liner_life_months', 6) or 6)
        consumable_rev_per = profile.get('avg_consumable_revenue', 50000) or 50000
        refurb_interval = max(1, profile.get('refurb_interval_months', 36) or 36)
        refurb_rev_per = profile.get('avg_refurb_revenue', 150000) or 150000
        
        # Calculate MONTHLY revenue (smoothed over replacement cycle)
        # Events per year = 12 / liner_life_months
        # Annual revenue = events_per_year * revenue_per_event
        # Monthly revenue = annual_revenue / 12
        consumable_events_per_year = 12 / liner_life
        annual_consumable = consumable_events_per_year * consumable_rev_per
        monthly_consumable = annual_consumable / 12
        
        refurb_events_per_year = 12 / refurb_interval
        annual_refurb = refurb_events_per_year * refurb_rev_per
        monthly_refurb = annual_refurb / 12
        
        # Apply smooth monthly revenue with inflation
        for month_idx in range(n_months):
            inflation_factor = (1 + inflation) ** (month_idx / 12)
            consumable_rev[month_idx] += monthly_consumable * inflation_factor
            refurb_rev[month_idx] += monthly_refurb * inflation_factor
    
    if progress_callback:
        progress_callback(0.5, f"Adding pipeline revenue ({len(data['prospects'])} prospects)...")
    
    # =================================================================
    # FIXED in 8.1: Pipeline revenue calculation
    # Now uses BOTH annual_liner_value AND refurb_value from prospects table
    # =================================================================
    for prospect in data['prospects']:
        # Get confidence (handle both % and decimal formats)
        confidence_raw = prospect.get('confidence_pct', 0) or 0
        confidence = confidence_raw / 100 if confidence_raw > 1 else confidence_raw
        
        # Get revenue - FIXED: Use BOTH annual_liner_value AND refurb_value
        annual_liner = prospect.get('annual_liner_value', 0) or 0
        refurb_value = prospect.get('refurb_value', 0) or 0
        
        # Fallback to old field names if new ones are empty
        if annual_liner == 0 and refurb_value == 0:
            annual_liner = prospect.get('expected_annual_revenue', 0) or 0
        
        # Total annual revenue from this prospect
        total_annual_rev = annual_liner + refurb_value
        monthly_rev = (total_annual_rev / 12) * confidence
        
        # Determine start month based on close date
        close_date = prospect.get('expected_close_date')
        start_idx = 0
        if close_date:
            try:
                close_dt = datetime.fromisoformat(str(close_date).replace('Z', '+00:00'))
                start_idx = max(0, (close_dt.year - start_date.year) * 12 + close_dt.month - start_date.month)
            except:
                start_idx = 0
        
        # Add revenue starting from close date
        for month_idx in range(start_idx, n_months):
            inflation_factor = (1 + inflation) ** (month_idx / 12)
            pipeline_rev[month_idx] += monthly_rev * inflation_factor
    
    if progress_callback:
        progress_callback(0.6, "Calculating costs...")
    
    # Calculate COGS with proper margins
    total_rev = consumable_rev + refurb_rev + pipeline_rev
    
    # COGS = Revenue * (1 - Margin)
    cogs_consumable = consumable_rev * (1 - margin_consumable)
    cogs_refurb = refurb_rev * (1 - margin_refurb)
    blended_margin = (margin_consumable + margin_refurb) / 2
    cogs_pipeline = pipeline_rev * (1 - blended_margin)
    base_cogs = cogs_consumable + cogs_refurb + cogs_pipeline
    
    # NEW in v9.2: Apply manufacturing strategy impact
    cogs = np.zeros(n_months)
    cogs_buy = np.zeros(n_months)
    cogs_make = np.zeros(n_months)
    mfg_variable_overhead = np.zeros(n_months)
    mfg_fixed_overhead = np.zeros(n_months)
    mfg_overhead = np.zeros(n_months)  # Combined for backward compatibility
    mfg_depreciation = np.zeros(n_months)
    commissioning_costs = np.zeros(n_months)
    
    if manufacturing_scenario and manufacturing_scenario.strategy != 'buy':
        # Import manufacturing function
        try:
            from components.vertical_integration import get_manufacturing_impact_for_forecast
        except ImportError:
            try:
                from vertical_integration import get_manufacturing_impact_for_forecast
            except ImportError:
                get_manufacturing_impact_for_forecast = None
        
        if get_manufacturing_impact_for_forecast:
            for month_idx in range(n_months):
                mfg_impact = get_manufacturing_impact_for_forecast(
                    manufacturing_scenario, 
                    month_idx, 
                    base_cogs[month_idx]
                )
                cogs[month_idx] = mfg_impact['total_cogs']
                cogs_buy[month_idx] = mfg_impact['buy_cogs']
                cogs_make[month_idx] = mfg_impact['mfg_cogs']
                mfg_variable_overhead[month_idx] = mfg_impact.get('mfg_variable_overhead', 0)
                mfg_fixed_overhead[month_idx] = mfg_impact.get('mfg_fixed_overhead', 0)
                mfg_overhead[month_idx] = mfg_impact.get('mfg_overhead', 0)  # Combined
                mfg_depreciation[month_idx] = mfg_impact['mfg_depreciation']
                commissioning_costs[month_idx] = mfg_impact['commissioning_cost']
        else:
            # Fallback if import fails
            cogs = base_cogs.copy()
            cogs_buy = base_cogs.copy()
    else:
        # No manufacturing - all COGS is buy
        cogs = base_cogs.copy()
        cogs_buy = base_cogs.copy()
    
    gross_profit = total_rev - cogs
    
    # Calculate OPEX
    opex = np.zeros(n_months)
    
    if data['expenses']:
        for expense in data['expenses']:
            if not expense.get('is_active', True):
                continue
            
            func_type = expense.get('function_type', 'fixed')
            
            for month_idx in range(n_months):
                escalation = expense.get('escalation_rate', 0) or 0
                inflation_factor = (1 + escalation) ** (month_idx / 12)
                
                if func_type == 'fixed':
                    monthly = expense.get('fixed_monthly', 0) or 0
                    opex[month_idx] += monthly * inflation_factor
                
                elif func_type == 'variable':
                    base = expense.get('fixed_monthly', 0) or 0
                    rate = expense.get('variable_rate', 0) or 0
                    opex[month_idx] += base * inflation_factor + (total_rev[month_idx] * rate)
                
                elif func_type == 'power':
                    coef = expense.get('power_coefficient', 1) or 1
                    rate = expense.get('variable_rate', 0.01) or 0.01
                    if total_rev[month_idx] > 0:
                        opex[month_idx] += rate * (total_rev[month_idx] ** coef)
                
                elif func_type == 'step':
                    base = expense.get('fixed_monthly', 0) or 0
                    threshold = expense.get('step_threshold', 1000000) or 1000000
                    step_amount = expense.get('step_amount', 50000) or 50000
                    steps = int(total_rev[month_idx] / threshold) if threshold > 0 else 0
                    opex[month_idx] += (base + steps * step_amount) * inflation_factor
                
                elif func_type == 'linked':
                    linked_to = expense.get('linked_to', '')
                    linked_rate = expense.get('linked_rate', 0) or 0
                    # Simple implementation - link to total COGS
                    if linked_to.lower() in ['cogs', 'cost_of_goods']:
                        opex[month_idx] += cogs[month_idx] * linked_rate
                    else:
                        opex[month_idx] += (expense.get('fixed_monthly', 0) or 0) * inflation_factor
                
                elif func_type == 'budget':
                    rate = expense.get('variable_rate', 0) or 0
                    budget = expense.get('annual_budget', 0) or 0
                    calculated = total_rev[month_idx] * rate
                    capped = min(calculated, budget / 12) if budget > 0 else calculated
                    opex[month_idx] += capped
    else:
        # Default OPEX if no expense assumptions (~27% of revenue based on historical)
        default_opex_rate = 0.27
        for month_idx in range(n_months):
            opex[month_idx] = total_rev[month_idx] * default_opex_rate
    
    if progress_callback:
        progress_callback(0.8, "Finalizing results...")
    
    ebit = gross_profit - opex
    
    # Store results (convert to lists for JSON serialization)
    results['revenue']['consumables'] = consumable_rev.tolist()
    results['revenue']['refurb'] = refurb_rev.tolist()
    results['revenue']['pipeline'] = pipeline_rev.tolist()
    results['revenue']['total'] = total_rev.tolist()
    
    results['costs']['cogs'] = cogs.tolist()
    results['costs']['cogs_buy'] = cogs_buy.tolist()
    results['costs']['cogs_make'] = cogs_make.tolist()
    results['costs']['mfg_variable_overhead'] = mfg_variable_overhead.tolist()
    results['costs']['mfg_fixed_overhead'] = mfg_fixed_overhead.tolist()
    results['costs']['mfg_overhead'] = mfg_overhead.tolist()  # Combined for backward compatibility
    results['costs']['mfg_depreciation'] = mfg_depreciation.tolist()
    results['costs']['commissioning'] = commissioning_costs.tolist()
    results['costs']['opex'] = opex.tolist()
    results['costs']['total'] = (cogs + opex).tolist()
    
    results['profit']['gross'] = gross_profit.tolist()
    results['profit']['ebit'] = ebit.tolist()
    
    # Calculate summary stats
    total_revenue_sum = float(np.sum(total_rev))
    total_gp_sum = float(np.sum(gross_profit))
    total_ebit_sum = float(np.sum(ebit))
    
    results['summary'] = {
        'total_revenue': total_revenue_sum,
        'total_consumables': float(np.sum(consumable_rev)),
        'total_refurb': float(np.sum(refurb_rev)),
        'total_pipeline': float(np.sum(pipeline_rev)),
        'total_cogs': float(np.sum(cogs)),
        'total_cogs_buy': float(np.sum(cogs_buy)),
        'total_cogs_make': float(np.sum(cogs_make)),
        'total_mfg_variable_overhead': float(np.sum(mfg_variable_overhead)),
        'total_mfg_fixed_overhead': float(np.sum(mfg_fixed_overhead)),
        'total_mfg_overhead': float(np.sum(mfg_overhead)),
        'total_mfg_depreciation': float(np.sum(mfg_depreciation)),
        'total_commissioning': float(np.sum(commissioning_costs)),
        'total_opex': float(np.sum(opex)),
        'total_gross_profit': total_gp_sum,
        'total_ebit': total_ebit_sum,
        'avg_gross_margin': float(total_gp_sum / max(total_revenue_sum, 1)),
        'avg_ebit_margin': float(total_ebit_sum / max(total_revenue_sum, 1)),
        'forecast_months': forecast_months,
        'machine_count': len(data['machines']),
        'prospect_count': len(data['prospects']),
        'data_source': data.get('data_source', 'unknown'),
        'margin_consumable_used': margin_consumable,
        'margin_refurb_used': margin_refurb,
        'assumptions_source': results['assumptions_source'],  # NEW in v9.0
        'manufacturing_included': manufacturing_scenario is not None,  # NEW in v9.2
        'manufacturing_strategy': manufacturing_scenario.strategy if manufacturing_scenario else None  # NEW in v9.2
    }
    
    results['success'] = True
    results['assumptions'] = assumptions
    
    if progress_callback:
        progress_callback(1.0, "Complete!")
    
    return results
    # End of original implementation (fallback)



# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

def run_monte_carlo(base_results: Dict, config: Dict, progress_callback=None) -> Dict[str, Any]:
    """Run Monte Carlo simulation on forecast results."""
    mc_results = {
        'success': False,
        'iterations': config.get('iterations', 1000),
        'percentiles': {
            'revenue': {},
            'gross_profit': {},
            'ebit': {}
        },
        'distributions': {}
    }
    
    try:
        n_iterations = config.get('iterations', 1000)
        fleet_cv = config.get('fleet_cv', 0.10)
        prospect_cv = config.get('prospect_cv', 0.30)
        cost_cv = config.get('cost_cv', 0.10)
        seed = config.get('seed', 42)
        
        np.random.seed(seed)
        
        base_consumables = np.array(base_results['revenue']['consumables'])
        base_refurb = np.array(base_results['revenue']['refurb'])
        base_pipeline = np.array(base_results['revenue']['pipeline'])
        base_cogs = np.array(base_results['costs']['cogs'])
        base_opex = np.array(base_results['costs']['opex'])
        
        n_months = len(base_consumables)
        
        # Store all iterations
        all_revenue = np.zeros((n_iterations, n_months))
        all_gp = np.zeros((n_iterations, n_months))
        all_ebit = np.zeros((n_iterations, n_months))
        
        for i in range(n_iterations):
            if progress_callback and i % 100 == 0:
                progress_callback(0.5 + 0.4 * (i / n_iterations), f"Monte Carlo iteration {i}/{n_iterations}")
            
            # Apply random variations
            consumables_sim = base_consumables * np.random.lognormal(0, fleet_cv, n_months)
            refurb_sim = base_refurb * np.random.lognormal(0, fleet_cv, n_months)
            pipeline_sim = base_pipeline * np.random.lognormal(0, prospect_cv, n_months)
            
            revenue_sim = consumables_sim + refurb_sim + pipeline_sim
            
            cogs_sim = base_cogs * np.random.lognormal(0, cost_cv, n_months)
            opex_sim = base_opex * np.random.lognormal(0, cost_cv, n_months)
            
            gp_sim = revenue_sim - cogs_sim
            ebit_sim = gp_sim - opex_sim
            
            all_revenue[i] = revenue_sim
            all_gp[i] = gp_sim
            all_ebit[i] = ebit_sim
        
        # Calculate percentiles
        mc_results['percentiles']['revenue'] = {
            'p10': np.percentile(all_revenue, 10, axis=0).tolist(),
            'p25': np.percentile(all_revenue, 25, axis=0).tolist(),
            'p50': np.percentile(all_revenue, 50, axis=0).tolist(),
            'p75': np.percentile(all_revenue, 75, axis=0).tolist(),
            'p90': np.percentile(all_revenue, 90, axis=0).tolist(),
            'mean': np.mean(all_revenue, axis=0).tolist(),
            'std': np.std(all_revenue, axis=0).tolist()
        }
        
        mc_results['percentiles']['gross_profit'] = {
            'p10': np.percentile(all_gp, 10, axis=0).tolist(),
            'p50': np.percentile(all_gp, 50, axis=0).tolist(),
            'p90': np.percentile(all_gp, 90, axis=0).tolist()
        }
        
        mc_results['percentiles']['ebit'] = {
            'p10': np.percentile(all_ebit, 10, axis=0).tolist(),
            'p50': np.percentile(all_ebit, 50, axis=0).tolist(),
            'p90': np.percentile(all_ebit, 90, axis=0).tolist()
        }
        
        # Total distributions
        mc_results['distributions'] = {
            'total_revenue': np.sum(all_revenue, axis=1).tolist(),
            'total_gp': np.sum(all_gp, axis=1).tolist(),
            'total_ebit': np.sum(all_ebit, axis=1).tolist()
        }
        
        mc_results['success'] = True
        
    except Exception as e:
        mc_results['error'] = str(e)
    
    return mc_results


# =============================================================================
# SNAPSHOT FUNCTIONS - FIXED JSON SERIALIZATION
# =============================================================================

def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        val = float(obj)
        return val if np.isfinite(val) else None
    # Handle numpy scalar float types (for NumPy 2.0+ compatibility, np.float_ removed)
    elif isinstance(obj, np.floating):
        val = float(obj)
        return val if np.isfinite(val) else None
    elif isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (pd.Timestamp, datetime, date)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif pd.isna(obj):
        return None
    return obj


def save_snapshot(
    db,
    scenario_id: str,
    user_id: str,
    forecast_results: Dict,
    mc_results: Optional[Dict] = None,
    snapshot_name: str = None,
    snapshot_type: str = 'base',
    notes: str = ''
) -> bool:
    """
    Save forecast results as a snapshot.
    FIXED: Properly convert numpy arrays to lists for JSON serialization.
    """
    try:
        if not snapshot_name:
            snapshot_name = f"Forecast {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Convert all numpy types to native Python types
        serializable_results = convert_to_serializable(forecast_results)
        serializable_mc = convert_to_serializable(mc_results) if mc_results else None
        
        # Build the insert data
        # IMPORTANT:
        # - forecast_snapshots columns are JSONB in Supabase; store dicts, not json.dumps() strings.
        # - Store the FULL forecast payload so downstream features (What-If) can re-use assumptions.
        # - Still keep summary_stats as its own JSONB for quick listing.
        forecast_blob = serializable_results or {}

        insert_data = {
            'scenario_id': scenario_id,
            'user_id': user_id,
            'snapshot_name': snapshot_name,
            'snapshot_type': snapshot_type,
            'snapshot_date': datetime.now().date().isoformat(),
            'forecast_data': forecast_blob,
            'summary_stats': serializable_results.get('summary', {}) or {},
            'total_revenue_forecast': float(serializable_results.get('summary', {}).get('total_revenue', 0) or 0),
            'total_gross_profit_forecast': float(serializable_results.get('summary', {}).get('total_gross_profit', 0) or 0),
            'enterprise_value': 0.0,
            'notes': notes or '',
            'is_locked': False
        }
        
        # Add optional fields
        if serializable_results.get('assumptions'):
            insert_data['assumptions_data'] = serializable_results['assumptions']
        
        if serializable_mc:
            insert_data['monte_carlo_data'] = serializable_mc
        
        # Insert into database
        result = db.client.table(TABLE_FORECAST_SNAPSHOTS).insert(insert_data).execute()
        
        if result.data:
            snapshot_id = result.data[0].get('id')
            
            # NEW: Update detailed line item forecasts with snapshot_id
            try:
                from components.detailed_line_item_forecast import save_forecast_line_items
                
                detailed_line_items = forecast_results.get('detailed_line_items', {})
                for statement_type, forecast_df in detailed_line_items.items():
                    if not forecast_df.empty:
                        save_forecast_line_items(
                            db=db,
                            scenario_id=scenario_id,
                            user_id=user_id,
                            snapshot_id=snapshot_id,
                            statement_type=statement_type,
                            forecast_df=forecast_df
                        )
            except Exception as e:
                st.warning(f"Could not save detailed line items to snapshot: {e}")
            
            return True
        return False
        
    except Exception as e:
        st.error(f"Error saving snapshot: {e}")
        traceback.print_exc()
        return False


def delete_snapshot(db, snapshot_id: str, user_id: str) -> bool:
    """Delete a forecast snapshot (if not locked)."""
    try:
        # Check if locked
        snapshot = db.client.table(TABLE_FORECAST_SNAPSHOTS).select('is_locked').eq('id', snapshot_id).execute()
        if snapshot.data and snapshot.data[0].get('is_locked'):
            st.warning("Cannot delete locked snapshot")
            return False
        
        db.client.table(TABLE_FORECAST_SNAPSHOTS).delete().eq('id', snapshot_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting snapshot: {e}")
        return False


def _maybe_json(value, default):
    """
    Snapshots historically stored some JSONB fields as JSON strings.
    This helper supports both legacy (string) and correct (dict) shapes.
    """
    if value is None:
        return default
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else default
        except Exception:
            return default
    return default


def _snapshot_to_forecast_results(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize snapshot row -> forecast_results dict.
    Supports:
    - legacy string-encoded JSON blobs
    - legacy "forecast_data" that only contains timeline/revenue/costs/profit
    - new snapshots where forecast_data is the full forecast results dict
    """
    try:
        forecast_data = _maybe_json(snapshot.get("forecast_data"), {})
        summary = _maybe_json(snapshot.get("summary_stats"), {})
        mc = _maybe_json(snapshot.get("monte_carlo_data"), None)
        valuation = _maybe_json(snapshot.get("valuation_data"), None)

        # New snapshots: forecast_data is already full results payload
        if isinstance(forecast_data, dict) and ("revenue" in forecast_data or "timeline" in forecast_data):
            out = dict(forecast_data)
            if "success" not in out:
                out["success"] = True
            if "summary" not in out:
                out["summary"] = summary or {}
            return out

        # Fallback: old minimal blob
        out = {
            "success": True,
            "timeline": (forecast_data or {}).get("timeline", []),
            "revenue": (forecast_data or {}).get("revenue", {}),
            "costs": (forecast_data or {}).get("costs", {}),
            "profit": (forecast_data or {}).get("profit", {}),
            "summary": summary or {},
        }
        return out
    except Exception:
        return None


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_revenue_chart(results: Dict, mc_results: Optional[Dict] = None, historical_df: pd.DataFrame = None) -> go.Figure:
    """Create revenue forecast chart with optional Monte Carlo bands and historical data."""
    fig = go.Figure()
    
    timeline = results.get('timeline', [])
    
    # Add historical data if available
    if historical_df is not None and not historical_df.empty and 'total_revenue' in historical_df.columns:
        hist_dates = historical_df['period_date'].dt.strftime('%Y-%m').tolist() if 'period_date' in historical_df.columns else []
        hist_revenue = historical_df['total_revenue'].tolist()
        
        if hist_dates and hist_revenue:
            fig.add_trace(go.Scatter(
                x=hist_dates,
                y=hist_revenue,
                name='Historical Actuals',
                mode='lines+markers',
                line=dict(color=CHART_COLORS['historical'], width=2, dash='dot'),
                marker=dict(size=6)
            ))
    
    # Add Monte Carlo confidence band if available
    if mc_results and mc_results.get('success'):
        p10 = mc_results['percentiles']['revenue']['p10']
        p90 = mc_results['percentiles']['revenue']['p90']
        
        fig.add_trace(go.Scatter(
            x=timeline + timeline[::-1],
            y=p90 + p10[::-1],
            fill='toself',
            fillcolor=CHART_COLORS['confidence_band'],
            line=dict(color='rgba(0,0,0,0)'),
            name='80% Confidence',
            showlegend=True
        ))
    
    # Revenue components
    fig.add_trace(go.Scatter(
        x=timeline,
        y=results['revenue']['consumables'],
        name='Consumables',
        stackgroup='revenue',
        line=dict(color=CHART_COLORS['primary'])
    ))
    
    fig.add_trace(go.Scatter(
        x=timeline,
        y=results['revenue']['refurb'],
        name='Refurbishment',
        stackgroup='revenue',
        line=dict(color=CHART_COLORS['secondary'])
    ))
    
    fig.add_trace(go.Scatter(
        x=timeline,
        y=results['revenue']['pipeline'],
        name='Pipeline',
        stackgroup='revenue',
        line=dict(color=CHART_COLORS['success'])
    ))
    
    fig.update_layout(
        title='Revenue Forecast',
        xaxis_title='Period',
        yaxis_title='Revenue (R)',
        height=400,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    fig.update_yaxes(tickformat=',.0f', tickprefix='R ')
    
    return fig


def create_profitability_chart(results: Dict) -> go.Figure:
    """Create profitability waterfall chart."""
    fig = go.Figure()
    
    timeline = results.get('timeline', [])
    
    # Gross profit
    fig.add_trace(go.Scatter(
        x=timeline,
        y=results['profit']['gross'],
        name='Gross Profit',
        line=dict(color=CHART_COLORS['success'], width=2)
    ))
    
    # EBIT
    fig.add_trace(go.Scatter(
        x=timeline,
        y=results['profit']['ebit'],
        name='EBIT',
        line=dict(color=CHART_COLORS['primary'], width=2)
    ))
    
    # OPEX (as area)
    fig.add_trace(go.Scatter(
        x=timeline,
        y=results['costs']['opex'],
        name='OPEX',
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.2)',
        line=dict(color=CHART_COLORS['danger'], width=1)
    ))
    
    fig.update_layout(
        title='Profitability Analysis',
        xaxis_title='Period',
        yaxis_title='Amount (R)',
        height=400,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    fig.update_yaxes(tickformat=',.0f', tickprefix='R ')
    
    return fig


def create_margin_chart(results: Dict) -> go.Figure:
    """Create margin trend chart."""
    fig = go.Figure()
    
    timeline = results.get('timeline', [])
    total_rev = np.array(results['revenue']['total'])
    gross_profit = np.array(results['profit']['gross'])
    ebit = np.array(results['profit']['ebit'])
    
    # Calculate margins (avoid division by zero)
    gp_margin = np.where(total_rev > 0, (gross_profit / total_rev) * 100, 0)
    ebit_margin = np.where(total_rev > 0, (ebit / total_rev) * 100, 0)
    
    fig.add_trace(go.Scatter(
        x=timeline,
        y=gp_margin,
        name='Gross Margin %',
        line=dict(color=CHART_COLORS['success'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=timeline,
        y=ebit_margin,
        name='EBIT Margin %',
        line=dict(color=CHART_COLORS['primary'], width=2)
    ))
    
    fig.update_layout(
        title='Margin Trends',
        xaxis_title='Period',
        yaxis_title='Margin %',
        height=350,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    fig.update_yaxes(ticksuffix='%')
    
    return fig


# =============================================================================
# FINANCIAL STATEMENT BUILDERS
# =============================================================================

def build_monthly_financials(
    forecast_result: Dict[str, Any],
    assumptions: Dict[str, Any],
    historical_data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Build monthly financial data from forecast results.
    Returns DataFrame with all monthly periods and line items.
    """
    all_rows = []
    existing_periods: set[tuple[int, int]] = set()
    
    # Add historical data first if available
    if historical_data is not None and not historical_data.empty:
        for _, row in historical_data.iterrows():
            period_date = row.get('period_date')
            if period_date is not None:
                if isinstance(period_date, str):
                    period_date = pd.to_datetime(period_date)

                # Prefer existing flags from upstream upsampling (YTD fill months)
                try:
                    _is_actual = row.get("is_actual")
                    is_actual = bool(_is_actual) if _is_actual is not None else True
                except Exception:
                    is_actual = True
                data_type = row.get("data_type") if isinstance(row.get("data_type"), str) else ("Actual" if is_actual else "Forecast (YTD fill)")
                
                # Get total opex for fallback calculations
                total_opex = row.get('total_opex', row.get('opex', 0)) or 0
                
                # Get actual expense breakdowns if available (from historic_expense_categories)
                # If not available, estimate based on typical ratios
                opex_personnel = row.get('opex_personnel', 0) or 0
                opex_facilities = row.get('opex_facilities', 0) or 0
                opex_logistics = row.get('opex_logistics', 0) or 0
                opex_professional = row.get('opex_professional', 0) or 0
                depreciation = row.get('depreciation', 0) or 0
                interest_expense = row.get('interest_expense', 0) or 0
                
                # Calculate opex_other as remainder
                known_opex = opex_personnel + opex_facilities + opex_logistics + opex_professional
                opex_other = max(total_opex - known_opex - depreciation, 0) if known_opex > 0 else total_opex
                
                # Use opex_logistics as opex_admin proxy, opex_professional as opex_sales proxy
                opex_admin = opex_logistics
                opex_sales = opex_professional
                
                # Get P&L items
                ebit = row.get('ebit', 0) or 0
                ebt = row.get('ebt', ebit - interest_expense) or 0
                tax_expense = row.get('tax', row.get('tax_expense', 0)) or 0
                net_income = row.get('net_income', ebt - tax_expense) or 0
                
                all_rows.append({
                    'period_year': period_date.year,
                    'period_month': period_date.month,
                    'period_date': period_date.strftime('%Y-%m-%d'),
                    'period_label': period_date.strftime('%b %Y'),
                    'is_actual': is_actual,
                    'data_type': data_type,
                    'is_annual_total': False,
                    'total_revenue': row.get('total_revenue', row.get('revenue', 0)) or 0,
                    'total_cogs': row.get('total_cogs', row.get('cogs', 0)) or 0,
                    # Manufacturing cost breakdown (zeros for historical)
                    'cogs_buy': row.get('total_cogs', row.get('cogs', 0)) or 0,  # All COGS is "buy" in historical
                    'cogs_make': 0,
                    'mfg_variable_overhead': 0,
                    'mfg_fixed_overhead': 0,
                    'mfg_overhead': 0,
                    'mfg_depreciation': 0,
                    'total_gross_profit': row.get('total_gross_profit', row.get('gross_profit', 0)) or 0,
                    'total_opex': total_opex,
                    'ebitda': row.get('ebitda', ebit + depreciation) or 0,
                    'ebit': ebit,
                    'net_income': net_income,
                    # Revenue detail (historical doesn't have this breakdown)
                    'rev_wear_existing': 0,
                    'rev_service_existing': 0,
                    'rev_wear_prospect': 0,
                    'rev_service_prospect': 0,
                    'revenue_existing': row.get('total_revenue', row.get('revenue', 0)) or 0,
                    'revenue_prospect': 0,
                    # Expense detail from historic_expense_categories (v8.3 fix)
                    'opex_personnel': opex_personnel,
                    'opex_facilities': opex_facilities,
                    'opex_admin': opex_admin,
                    'opex_sales': opex_sales,
                    'opex_other': opex_other,
                    'depreciation': depreciation,
                    'interest_expense': interest_expense,
                    'ebt': ebt,
                    'tax_expense': tax_expense
                })
                try:
                    existing_periods.add((int(period_date.year), int(period_date.month)))
                except Exception:
                    pass
    
    # Add forecast data
    timeline = forecast_result.get('timeline', [])
    
    if not timeline:
        return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    
    # Use the latest historical date (if available) to anchor YTD; fallback to now.
    # IMPORTANT: Don't override this later with datetime.now() — YTD should reflect the
    # latest imported actual month for the scenario (especially right after import).
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    if historical_data is not None and not historical_data.empty and 'period_date' in historical_data.columns:
        latest_date = pd.to_datetime(historical_data['period_date'], errors='coerce').max()
    else:
        latest_date = None
    current_date = latest_date if pd.notna(latest_date) else datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    
    # Extract margins from assumptions
    gross_margin_wear = assumptions.get('gross_margin_liner') or assumptions.get('margin_consumable_pct', 0.38)
    if gross_margin_wear > 1:
        gross_margin_wear = gross_margin_wear / 100
    
    gross_margin_service = assumptions.get('gross_margin_refurb') or assumptions.get('margin_refurb_pct', 0.32)
    if gross_margin_service > 1:
        gross_margin_service = gross_margin_service / 100
    
    tax_rate = assumptions.get('tax_rate', 0.28)
    if tax_rate > 1:
        tax_rate = tax_rate / 100
    
    depreciation_pct = assumptions.get('depreciation_pct_revenue', 0.03)
    
    # Interest calculation parameters (NEW - Best Practice)
    prime_lending_rate = assumptions.get('prime_lending_rate', 0.10)  # Default 10% annual
    if prime_lending_rate > 1:
        prime_lending_rate = prime_lending_rate / 100  # Convert percentage to decimal
    call_rate = assumptions.get('call_rate', 0.05)  # Default 5% annual for cash deposits
    if call_rate > 1:
        call_rate = call_rate / 100
    
    # Track cash and debt balances for interest calculation
    initial_cash = assumptions.get('initial_cash', 5_000_000)
    initial_debt = assumptions.get('initial_debt', 0)
    current_cash = initial_cash
    current_debt = initial_debt
    
    # Fallback for backward compatibility (if rates not set, use old method)
    use_calculated_interest = assumptions.get('prime_lending_rate') is not None or assumptions.get('call_rate') is not None
    interest_annual_fallback = assumptions.get('interest_expense', 0)
    
    consumables = forecast_result.get('revenue', {}).get('consumables', [])
    refurb = forecast_result.get('revenue', {}).get('refurb', [])
    pipeline = forecast_result.get('revenue', {}).get('pipeline', [])
    total_rev_arr = forecast_result.get('revenue', {}).get('total', [])
    cogs_arr = forecast_result.get('costs', {}).get('cogs', [])
    opex_arr = forecast_result.get('costs', {}).get('opex', [])
    # NOTE: keep using current_year/current_month anchored above (latest historical period)
    # so we don't mistakenly treat an already-imported month as "missing".
    
    # If historical data already includes flags (e.g. "Forecast (YTD fill)"),
    # do not run the legacy YTD pre-load / gross-up logic (it causes duplicates).
    historical_has_flags = bool(
        historical_data is not None
        and not historical_data.empty
        and any(c in historical_data.columns for c in ["is_actual", "data_type"])
    )

    # Check if we have historical data for current month or any YTD months
    has_current_month_actual = False
    has_ytd_actuals = False
    ytd_actuals_data = []
    
    if historical_data is not None and not historical_data.empty:
        # Ensure period_date is datetime
        if 'period_date' in historical_data.columns:
            historical_data['period_date'] = pd.to_datetime(historical_data['period_date'], errors='coerce')
        
        for _, hist_row in historical_data.iterrows():
            hist_date = hist_row.get('period_date')
            if hist_date and pd.notna(hist_date):
                if isinstance(hist_date, str):
                    hist_date = pd.to_datetime(hist_date)
                
                # Check if this is current month
                if hist_date.year == current_year and hist_date.month == current_month:
                    # If upsampled history includes is_actual flags, honor them.
                    try:
                        if not historical_has_flags:
                            has_current_month_actual = True
                        else:
                            _ia = hist_row.get("is_actual")
                            has_current_month_actual = bool(_ia) if _ia is not None else True
                    except Exception:
                        has_current_month_actual = True
                
                # Check if this is any YTD month (current year, up to current month)
                if hist_date.year == current_year and hist_date.month <= current_month:
                    if not historical_has_flags:
                        has_ytd_actuals = True
                        ytd_actuals_data.append(hist_row.to_dict())
    
    # NEW: Pre-load YTD actuals into all_rows before forecast loop
    # DIAGNOSTIC: Log YTD detection for debugging
    missing_months = []
    if historical_data is not None and not historical_data.empty:
        hist_dates = historical_data.get('period_date')
        if hist_dates is not None:
            hist_dates = pd.to_datetime(hist_dates, errors='coerce')
            hist_months = {(d.year, d.month) for d in hist_dates.dropna()}
            for m in range(1, current_month + 1):
                if (current_year, m) not in hist_months:
                    missing_months.append(m)

    ytd_debug_info = {
        'current_year': current_year,
        'current_month': current_month,
        'has_ytd_actuals': has_ytd_actuals,
        'ytd_months_found': len(ytd_actuals_data) if ytd_actuals_data else 0,
        'ytd_periods': [],
        'ytd_revenue_total': 0,
        'missing_months': missing_months,
    }
    
    if (not historical_has_flags) and has_ytd_actuals and ytd_actuals_data:
        ytd_rows_added = []
        for ytd_row in ytd_actuals_data:
            hist_date = ytd_row.get('period_date')
            if hist_date and pd.notna(hist_date):
                if isinstance(hist_date, str):
                    hist_date = pd.to_datetime(hist_date)
                
                # Only include YTD months (up to current month)
                if hist_date.year == current_year and hist_date.month <= current_month:
                    # Convert historical data row to monthly financials format
                    ytd_period_row = _convert_historical_to_monthly_row(ytd_row, assumptions)
                    if ytd_period_row:
                        all_rows.append(ytd_period_row)
                        ytd_rows_added.append(ytd_period_row)
                        # Track for diagnostics
                        ytd_debug_info['ytd_periods'].append(hist_date.strftime('%Y-%m'))
                        ytd_debug_info['ytd_revenue_total'] += ytd_period_row.get('total_revenue', 0)

        # If YTD stops before December, gross-up the remaining months using the YTD average
        if ytd_rows_added:
            ytd_df = pd.DataFrame(ytd_rows_added)
            ytd_df = ytd_df[ytd_df['period_year'] == current_year]
            if not ytd_df.empty:
                max_month = int(ytd_df['period_month'].max())
                if max_month < 12:
                    # Fill only months that are actually missing.
                    # This prevents cases where a month exists in imported actuals
                    # but is still being treated as a forecast fill month.
                    existing_periods = set()
                    for r in all_rows:
                        try:
                            y = r.get('period_year')
                            m = r.get('period_month')
                            if y is None or m is None:
                                continue
                            existing_periods.add((int(y), int(m)))
                        except Exception:
                            continue

                    months_to_fill = [
                        m for m in range(max_month + 1, 13)
                        if (int(current_year), int(m)) not in existing_periods
                    ]
                    numeric_means = ytd_df.select_dtypes(include=[np.number]).mean().fillna(0)
                    for m in months_to_fill:
                        filler_date = datetime(current_year, m, 1)
                        # Build a gross-up row using YTD averages
                        row = {
                            'period_year': current_year,
                            'period_month': m,
                            'period_date': filler_date.strftime('%Y-%m-%d'),
                            'period_label': filler_date.strftime('%b %Y'),
                            'is_actual': False,
                            'data_type': 'Forecast (YTD avg)',
                            # Revenue
                            'total_revenue': numeric_means.get('total_revenue', 0),
                            'revenue_existing': numeric_means.get('revenue_existing', 0),
                            'revenue_prospect': numeric_means.get('revenue_prospect', 0),
                            'rev_wear_existing': numeric_means.get('rev_wear_existing', 0),
                            'rev_service_existing': numeric_means.get('rev_service_existing', 0),
                            'rev_wear_prospect': numeric_means.get('rev_wear_prospect', 0),
                            'rev_service_prospect': numeric_means.get('rev_service_prospect', 0),
                            # COGS
                            'total_cogs': numeric_means.get('total_cogs', 0),
                            'cogs_wear_existing': numeric_means.get('cogs_wear_existing', 0),
                            'cogs_service_existing': numeric_means.get('cogs_service_existing', 0),
                            'cogs_wear_prospect': numeric_means.get('cogs_wear_prospect', 0),
                            'cogs_service_prospect': numeric_means.get('cogs_service_prospect', 0),
                            # OPEX
                            'total_opex': numeric_means.get('total_opex', 0),
                            'opex_personnel': numeric_means.get('opex_personnel', 0),
                            'opex_facilities': numeric_means.get('opex_facilities', 0),
                            'opex_admin': numeric_means.get('opex_admin', 0),
                            'opex_sales': numeric_means.get('opex_sales', 0),
                            'opex_other': numeric_means.get('opex_other', 0),
                            # Profitability
                            'total_gross_profit': numeric_means.get('total_gross_profit', 0),
                            'ebitda': numeric_means.get('ebitda', 0),
                            'depreciation': numeric_means.get('depreciation', 0),
                            'ebit': numeric_means.get('ebit', 0),
                            'interest_expense': numeric_means.get('interest_expense', 0),
                            'ebt': numeric_means.get('ebt', 0),
                            'tax_expense': numeric_means.get('tax_expense', 0),
                            'net_income': numeric_means.get('net_income', 0),
                        }
                        all_rows.append(row)
                        ytd_debug_info['ytd_periods'].append(filler_date.strftime('%Y-%m'))
    
    # Store diagnostics in session state for UI display
    if historical_has_flags:
        # Build a simpler diagnostic from the already-loaded monthly rows
        try:
            cy = int(current_year)
            actual_months = sorted(
                {
                    int(r.get("period_month"))
                    for r in all_rows
                    if int(r.get("period_year", -1)) == cy and bool(r.get("is_actual", True))
                }
            )
            ytd_debug_info["has_ytd_actuals"] = len(actual_months) > 0
            ytd_debug_info["ytd_months_found"] = len(actual_months)
            ytd_debug_info["ytd_periods"] = [f"{cy}-{m:02d}" for m in actual_months]
        except Exception:
            pass
    st.session_state['ytd_diagnostic'] = ytd_debug_info
    
    # Rebuild existing_periods after any YTD filler rows were added
    try:
        existing_periods = {
            (int(r.get("period_year")), int(r.get("period_month")))
            for r in all_rows
            if r.get("period_year") is not None and r.get("period_month") is not None
        }
    except Exception:
        existing_periods = set()

    for i, period in enumerate(timeline):
        try:
            period_date = datetime.strptime(period, '%Y-%m')
        except:
            # If we have actual for current month, start forecast from next month
            # Otherwise, start from current month
            if has_current_month_actual:
                period_date = datetime.now().replace(day=1) + relativedelta(months=i+1)
            else:
                period_date = datetime.now().replace(day=1) + relativedelta(months=i)
        
        # Skip this period if we already have data (actual or YTD-fill) for it
        try:
            period_year = int(period_date.year)
            period_month = int(period_date.month)
            if (period_year, period_month) in existing_periods:
                continue
        except Exception:
            pass
        
        # Revenue breakdown
        rev_wear_existing = consumables[i] if i < len(consumables) else 0
        rev_service_existing = refurb[i] if i < len(refurb) else 0
        rev_prospect = pipeline[i] if i < len(pipeline) else 0
        
        # Split prospect revenue 70/30 wear/service
        rev_wear_prospect = rev_prospect * 0.7
        rev_service_prospect = rev_prospect * 0.3
        
        revenue_existing = rev_wear_existing + rev_service_existing
        revenue_prospect = rev_wear_prospect + rev_service_prospect
        total_revenue = total_rev_arr[i] if i < len(total_rev_arr) else (revenue_existing + revenue_prospect)
        
        # COGS by segment
        cogs_wear_existing = rev_wear_existing * (1 - gross_margin_wear)
        cogs_service_existing = rev_service_existing * (1 - gross_margin_service)
        cogs_wear_prospect = rev_wear_prospect * (1 - gross_margin_wear)
        cogs_service_prospect = rev_service_prospect * (1 - gross_margin_service)
        total_cogs = cogs_arr[i] if i < len(cogs_arr) else (cogs_wear_existing + cogs_service_existing + cogs_wear_prospect + cogs_service_prospect)
        
        # Manufacturing COGS breakdown (NEW in v9.2)
        cogs_buy_arr = forecast_result.get('costs', {}).get('cogs_buy', [])
        cogs_make_arr = forecast_result.get('costs', {}).get('cogs_make', [])
        mfg_overhead_arr = forecast_result.get('costs', {}).get('mfg_overhead', [])
        mfg_depreciation_arr = forecast_result.get('costs', {}).get('mfg_depreciation', [])
        mfg_variable_overhead_arr = forecast_result.get('costs', {}).get('mfg_variable_overhead', [])
        mfg_fixed_overhead_arr = forecast_result.get('costs', {}).get('mfg_fixed_overhead', [])
        
        cogs_buy = cogs_buy_arr[i] if cogs_buy_arr and i < len(cogs_buy_arr) else total_cogs
        cogs_make = cogs_make_arr[i] if cogs_make_arr and i < len(cogs_make_arr) else 0
        mfg_overhead = mfg_overhead_arr[i] if mfg_overhead_arr and i < len(mfg_overhead_arr) else 0
        mfg_variable_overhead = mfg_variable_overhead_arr[i] if mfg_variable_overhead_arr and i < len(mfg_variable_overhead_arr) else 0
        mfg_fixed_overhead = mfg_fixed_overhead_arr[i] if mfg_fixed_overhead_arr and i < len(mfg_fixed_overhead_arr) else 0
        mfg_depreciation = mfg_depreciation_arr[i] if mfg_depreciation_arr and i < len(mfg_depreciation_arr) else 0
        
        # NEW: Calculate revenue split (bought vs manufactured) based on COGS ratio
        # This represents which portion of revenue corresponds to bought vs manufactured products
        if total_cogs > 0:
            buy_pct = cogs_buy / total_cogs
            make_pct = cogs_make / total_cogs if cogs_make > 0 else 0
        else:
            buy_pct = 1.0
            make_pct = 0.0
        
        revenue_bought = total_revenue * buy_pct
        revenue_manufactured = total_revenue * make_pct
        
        # Gross profit
        total_gross_profit = total_revenue - total_cogs
        
        # Operating expenses breakdown
        total_opex = opex_arr[i] if i < len(opex_arr) else (total_revenue * 0.27)
        opex_personnel = total_opex * 0.45
        opex_facilities = total_opex * 0.20
        opex_admin = total_opex * 0.15
        opex_sales = total_opex * 0.12
        opex_other = total_opex * 0.08
        
        # Operating metrics
        ebitda = total_gross_profit - total_opex
        depreciation = total_revenue * depreciation_pct
        ebit = ebitda - depreciation
        
        # Calculate interest based on balance sheet (NEW - Best Practice)
        if use_calculated_interest:
            # Calculate interest based on beginning-of-period balances
            interest = calculate_interest_expense(
                cash_balance=current_cash,
                debt_balance=current_debt,
                prime_lending_rate=prime_lending_rate,
                call_rate=call_rate,
                period_months=1
            )
        else:
            # Fallback to old method (backward compatibility)
            interest = interest_annual_fallback / 12
        
        ebt = ebit - interest
        tax = max(ebt * tax_rate, 0)
        net_income = ebt - tax
        
        # Update cash and debt balances for next period (simplified - will be refined by balance sheet builder)
        # This is a rough estimate: cash increases by net income, decreases by capex
        capex_pct = assumptions.get('capex_pct_revenue', 0.05)
        capex = total_revenue * capex_pct
        current_cash = current_cash + net_income - capex
        # If cash goes negative, it becomes debt
        if current_cash < 0:
            current_debt = abs(current_cash)
            current_cash = 0
        else:
            # Debt is paid down if cash is positive (simplified)
            if current_debt > 0 and current_cash > 0:
                debt_payment = min(current_cash * 0.1, current_debt)  # Pay down 10% of cash or all debt
                current_debt = max(0, current_debt - debt_payment)
                current_cash = current_cash - debt_payment
        
        all_rows.append({
            'period_year': period_date.year,
            'period_month': period_date.month,
            'period_date': period_date.strftime('%Y-%m-%d'),
            'period_label': period_date.strftime('%b %Y'),
            'is_actual': False,
            'data_type': 'Forecast',
            'is_annual_total': False,
            
            # Revenue detail - with bought/manufactured split (NEW)
            'rev_wear_existing': rev_wear_existing,
            'rev_service_existing': rev_service_existing,
            'rev_wear_prospect': rev_wear_prospect,
            'rev_service_prospect': rev_service_prospect,
            'revenue_existing': revenue_existing,
            'revenue_prospect': revenue_prospect,
            'revenue_bought': revenue_bought,  # NEW: Revenue from bought products
            'revenue_manufactured': revenue_manufactured,  # NEW: Revenue from manufactured products
            'total_revenue': total_revenue,
            
            # COGS - with manufacturing breakdown (NEW in v9.2, UPDATED v3.2)
            'cogs_buy': cogs_buy,
            'cogs_make': cogs_make,
            'mfg_variable_overhead': mfg_variable_overhead,
            'mfg_fixed_overhead': mfg_fixed_overhead,
            'mfg_overhead': mfg_overhead,  # Combined for backward compatibility
            'mfg_depreciation': mfg_depreciation,
            'total_cogs': total_cogs,
            
            # Gross Profit
            'total_gross_profit': total_gross_profit,
            
            # OpEx detail
            'opex_personnel': opex_personnel,
            'opex_facilities': opex_facilities,
            'opex_admin': opex_admin,
            'opex_sales': opex_sales,
            'opex_other': opex_other,
            'total_opex': total_opex,
            
            # P&L
            'ebitda': ebitda,
            'depreciation': depreciation,
            'ebit': ebit,
            'interest_expense': interest,
            'ebt': ebt,
            'tax_expense': tax,
            'net_income': net_income,
        })
    
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(['period_year', 'period_month']).reset_index(drop=True)
    
    return df


def aggregate_to_annual(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly data to annual with proper totals.
    
    For the current year, combines YTD actuals with forecast for remaining months.
    """
    if monthly_df.empty:
        return pd.DataFrame()
    
    from datetime import datetime
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    numeric_cols = [col for col in monthly_df.columns if col not in 
                   ['period_date', 'period_year', 'period_month', 'period_label', 
                    'is_actual', 'data_type', 'is_annual_total']]
    
    # Group by year, preserving is_actual flag
    annual_df = monthly_df.groupby('period_year').agg({
        **{col: 'sum' for col in numeric_cols},
        'is_actual': 'any'  # If any month is actual, the year is actual
    }).reset_index()
    
    # For current year, check if it has both actual and forecast data
    current_year_data = monthly_df[monthly_df['period_year'] == current_year]
    if not current_year_data.empty:
        has_actuals = current_year_data['is_actual'].any()
        has_forecast = (~current_year_data['is_actual']).any()
        
        if has_actuals and has_forecast:
            # Current year has both actuals (YTD) and forecast (remaining months)
            # Mark as "Actual + Forecast" for clarity
            annual_df.loc[annual_df['period_year'] == current_year, 'data_type'] = 'Actual + Forecast Annual'
            # Don't mark as "A" since it's partially forecast
            annual_df.loc[annual_df['period_year'] == current_year, 'period_label'] = f"FY{current_year}"
        elif has_actuals:
            # Only actuals (shouldn't happen if forecast is running, but handle it)
            annual_df.loc[annual_df['period_year'] == current_year, 'data_type'] = 'Actual Annual'
            annual_df.loc[annual_df['period_year'] == current_year, 'period_label'] = f"FY{current_year} A"
        elif has_forecast:
            # Only forecast (full year forecast)
            annual_df.loc[annual_df['period_year'] == current_year, 'data_type'] = 'Forecast Annual'
            annual_df.loc[annual_df['period_year'] == current_year, 'period_label'] = f"FY{current_year}"
    
    # For other years, use existing logic
    for idx, row in annual_df.iterrows():
        if row['period_year'] != current_year:
            if row.get('is_actual', False):
                annual_df.at[idx, 'period_label'] = f"FY{row['period_year']} A"
                annual_df.at[idx, 'data_type'] = 'Actual Annual'
            else:
                annual_df.at[idx, 'period_label'] = f"FY{row['period_year']}"
                annual_df.at[idx, 'data_type'] = 'Forecast Annual'
    
    annual_df['is_annual_total'] = True
    annual_df['period_month'] = 12
    
    return annual_df


def _convert_historical_to_monthly_row(hist_row: dict, assumptions: dict) -> Optional[dict]:
    """
    Convert a historical data row to monthly financials format.
    
    Args:
        hist_row: Dictionary with historical financial data
        assumptions: Assumptions dictionary for defaults
    
    Returns:
        Dictionary in monthly financials format, or None if conversion fails
    """
    try:
        period_date = hist_row.get('period_date')
        if not period_date:
            return None
        
        if isinstance(period_date, str):
            period_date = pd.to_datetime(period_date)
        
        if pd.isna(period_date):
            return None
        
        # Extract values with defaults
        revenue = float(hist_row.get('revenue', hist_row.get('total_revenue', 0)) or 0)
        cogs = float(hist_row.get('cogs', hist_row.get('total_cogs', 0)) or 0)
        opex = float(hist_row.get('opex', hist_row.get('total_opex', 0)) or 0)
        gross_profit = float(hist_row.get('gross_profit', hist_row.get('total_gross_profit', revenue - cogs)) or 0)
        depreciation = float(hist_row.get('depreciation', 0) or 0)
        interest = float(hist_row.get('interest_expense', hist_row.get('interest', 0)) or 0)
        tax = float(hist_row.get('tax', hist_row.get('tax_expense', 0)) or 0)
        
        # Extract OPEX detail columns if available (FIX: was always setting to 0)
        opex_personnel = float(hist_row.get('opex_personnel', 0) or 0)
        opex_facilities = float(hist_row.get('opex_facilities', 0) or 0)
        opex_admin = float(hist_row.get('opex_admin', hist_row.get('opex_logistics', 0)) or 0)
        opex_sales = float(hist_row.get('opex_sales', hist_row.get('opex_professional', hist_row.get('opex_marketing', 0))) or 0)
        
        # Calculate opex_other as remainder
        known_opex = opex_personnel + opex_facilities + opex_admin + opex_sales
        opex_other = max(opex - known_opex - depreciation, 0) if known_opex > 0 else opex
        
        # Calculate derived values
        ebit = gross_profit - opex - depreciation
        ebt = ebit - interest
        net_income = ebt - tax
        
        return {
            'period_year': period_date.year,
            'period_month': period_date.month,
            'period_date': period_date.strftime('%Y-%m-%d'),
            'period_label': period_date.strftime('%b %Y'),
            'is_actual': True,
            'data_type': 'Actual',
            
            # Revenue
            'total_revenue': revenue,
            'revenue_existing': revenue * 0.7,  # Estimate split if not available
            'revenue_prospect': revenue * 0.3,
            'rev_wear_existing': revenue * 0.5,
            'rev_service_existing': revenue * 0.2,
            'rev_wear_prospect': revenue * 0.2,
            'rev_service_prospect': revenue * 0.1,
            
            # COGS
            'total_cogs': cogs,
            'cogs_wear_existing': cogs * 0.5,
            'cogs_service_existing': cogs * 0.2,
            'cogs_wear_prospect': cogs * 0.2,
            'cogs_service_prospect': cogs * 0.1,
            
            # Gross Profit
            'total_gross_profit': gross_profit,
            
            # OPEX with detail (FIX: now extracts from hist_row if available)
            'total_opex': opex,
            'opex_personnel': opex_personnel,
            'opex_facilities': opex_facilities,
            'opex_admin': opex_admin,
            'opex_sales': opex_sales,
            'opex_other': opex_other,
            
            # Operating Results
            'ebitda': ebit + depreciation,
            'depreciation': depreciation,
            'ebit': ebit,
            'interest_expense': interest,
            'ebt': ebt,
            'tax_expense': tax,
            'net_income': net_income,
        }
    except Exception as e:
        return None


# =============================================================================
# INTEREST CALCULATION (NEW - Best Practice Financial Modeling)
# =============================================================================

def calculate_interest_expense(
    cash_balance: float,
    debt_balance: float,
    prime_lending_rate: float,
    call_rate: float,
    period_months: int = 1
) -> float:
    """
    Calculate interest expense/income based on cash and debt balances.
    
    Best Practice Financial Modeling:
    - If cash balance is negative (overdraft/debt), calculate interest expense at prime lending rate
    - If cash balance is positive, calculate interest income at call rate
    - Net interest = Interest Expense (on debt) - Interest Income (on cash)
    
    Args:
        cash_balance: Current cash balance (can be negative for overdraft)
        debt_balance: Total debt balance (long-term + short-term)
        prime_lending_rate: Annual prime lending rate (as decimal, e.g., 0.10 for 10%)
        call_rate: Annual call/money market rate for cash deposits (as decimal, e.g., 0.05 for 5%)
        period_months: Number of months in the period (default 1 for monthly)
    
    Returns:
        Net interest expense (positive = expense, negative = income) for the period
    """
    # Convert annual rates to period rates
    period_rate_debt = prime_lending_rate * (period_months / 12)
    period_rate_cash = call_rate * (period_months / 12)
    
    # Calculate interest on debt (if debt exists or cash is negative)
    if debt_balance > 0:
        interest_expense_on_debt = debt_balance * period_rate_debt
    elif cash_balance < 0:
        # Negative cash = overdraft, treat as debt
        interest_expense_on_debt = abs(cash_balance) * period_rate_debt
    else:
        interest_expense_on_debt = 0
    
    # Calculate interest income on positive cash balance
    if cash_balance > 0:
        interest_income_on_cash = cash_balance * period_rate_cash
    else:
        interest_income_on_cash = 0
    
    # Net interest expense (positive = expense, negative = income)
    net_interest = interest_expense_on_debt - interest_income_on_cash
    
    return net_interest


# =============================================================================
# BALANCE SHEET & CASH FLOW BUILDERS (Sprint 16)
# =============================================================================

def build_balance_sheet(
    forecast_result: Dict[str, Any],
    assumptions: Dict[str, Any],
    historical_data: pd.DataFrame = None,
    manufacturing_strategy: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Build balance sheet projection from forecast results.
    
    NEW in v9.2:
    - Includes historical data for comparison (if provided)
    - Separates manufacturing PPE from other PPE
    - Splits inventory: Raw Materials (MFG) vs Finished Goods (Trading)
    - Working capital from manufacturing flows into inventory
    
    Working Capital Assumptions:
    - DSO (Days Sales Outstanding): From assumptions or default 45 days
    - DIO (Days Inventory Outstanding): From assumptions or default 60 days  
    - DPO (Days Payable Outstanding): From assumptions or default 30 days
    """
    all_rows = []
    
    # Get assumptions
    dso = assumptions.get('days_sales_outstanding', 45)
    dio = assumptions.get('days_inventory_outstanding', 60)
    dpo = assumptions.get('days_payable_outstanding', 30)
    initial_cash = assumptions.get('initial_cash', 5_000_000)
    initial_debt = assumptions.get('initial_debt', 0)
    initial_equity = assumptions.get('initial_equity', 10_000_000)
    depreciation_pct = assumptions.get('depreciation_pct_revenue', 0.03)
    capex_pct = assumptions.get('capex_pct_revenue', 0.05)
    tax_rate = assumptions.get('tax_rate', 0.28)
    if tax_rate > 1:
        tax_rate = tax_rate / 100
    
    # Manufacturing details
    mfg_equipment = 0
    mfg_facility = 0
    mfg_tooling = 0
    mfg_wc = 0  # Manufacturing working capital (raw material stock)
    mfg_start_month = 0
    mfg_active = False
    if manufacturing_strategy:
        mfg_equipment = manufacturing_strategy.get('equipment_cost', 0)
        mfg_facility = manufacturing_strategy.get('facility_cost', 0)
        mfg_tooling = manufacturing_strategy.get('tooling_cost', 0)
        mfg_wc = manufacturing_strategy.get('working_capital', 0)
        mfg_start_month = manufacturing_strategy.get('commissioning_completion_month', 0)
    
    total_mfg_ppe = mfg_equipment + mfg_facility + mfg_tooling
    
    # ==========================================================================
    # SECTION 1: HISTORICAL DATA (if provided)
    # ==========================================================================
    if historical_data is not None and not historical_data.empty:
        # Try to get historical balance sheet items
        hist_df = historical_data.copy()
        
        # Standardize column names
        if 'month' in hist_df.columns and 'period_date' not in hist_df.columns:
            hist_df['period_date'] = pd.to_datetime(hist_df['month'])
        elif 'period_date' in hist_df.columns:
            hist_df['period_date'] = pd.to_datetime(hist_df['period_date'])
        
        if 'period_date' in hist_df.columns:
            hist_df = hist_df.sort_values('period_date')
            
            for _, row in hist_df.iterrows():
                period_date = row['period_date']
                if isinstance(period_date, str):
                    try:
                        period_date = datetime.strptime(period_date, '%Y-%m-%d')
                    except:
                        continue
                
                # Get historical P&L values
                revenue = row.get('revenue', row.get('total_revenue', 0)) or 0
                cogs = row.get('cogs', row.get('total_cogs', 0)) or 0
                opex = row.get('opex', row.get('total_opex', 0)) or 0
                ebit = revenue - cogs - opex
                
                # Estimate balance sheet items from P&L
                accounts_receivable = revenue * dso / 30
                inventory_finished_goods = cogs * dio / 30
                accounts_payable = cogs * dpo / 30
                
                all_rows.append({
                    'period_year': period_date.year,
                    'period_month': period_date.month,
                    'period_date': period_date.strftime('%Y-%m-%d'),
                    'period_label': period_date.strftime('%b %Y'),
                    'is_actual': True,
                    'data_type': 'Actual',
                    
                    # Assets
                    'cash_and_equivalents': initial_cash,
                    'accounts_receivable': accounts_receivable,
                    'inventory_raw_materials': 0,  # No manufacturing in historics
                    'inventory_finished_goods': inventory_finished_goods,
                    'inventory_total': inventory_finished_goods,
                    'prepaid_expenses': revenue * 0.02,
                    'total_current_assets': initial_cash + accounts_receivable + inventory_finished_goods + revenue * 0.02,
                    
                    'ppe_other': assumptions.get('initial_ppe', 15_000_000),
                    'ppe_manufacturing': 0,  # No mfg PPE in historics
                    'property_plant_equipment': assumptions.get('initial_ppe', 15_000_000),
                    'accumulated_depreciation': 0,
                    'net_ppe': assumptions.get('initial_ppe', 15_000_000),
                    'intangible_assets': 0,
                    'total_non_current_assets': assumptions.get('initial_ppe', 15_000_000),
                    
                    # Liabilities
                    'accounts_payable': accounts_payable,
                    'accrued_expenses': opex * 0.3,
                    'short_term_debt': 0,
                    'total_current_liabilities': accounts_payable + opex * 0.3,
                    
                    'long_term_debt': initial_debt,
                    'total_non_current_liabilities': initial_debt,
                    
                    # Equity
                    'share_capital': initial_equity,
                    'retained_earnings': 0,
                    'total_equity': initial_equity,
                    
                    # Totals (calculated later)
                    'total_assets': 0,
                    'total_liabilities': 0,
                    'total_liabilities_and_equity': 0,
                    'net_working_capital': (accounts_receivable + inventory_finished_goods) - accounts_payable,
                })
    
    # ==========================================================================
    # SECTION 2: FORECAST DATA (VECTORIZED - Sprint 17.5)
    # ==========================================================================
    timeline = forecast_result.get('timeline', [])
    n_periods = len(timeline)
    
    if n_periods == 0:
        df = pd.DataFrame(all_rows)
        if not df.empty:
            df['total_liabilities_and_equity'] = df['total_liabilities'] + df['total_equity']
            df = df.sort_values(['period_year', 'period_month']).reset_index(drop=True)
        return df
    
    # Prepare arrays with proper length
    total_rev_arr = np.array(forecast_result.get('revenue', {}).get('total', [])[:n_periods] + [0] * max(0, n_periods - len(forecast_result.get('revenue', {}).get('total', []))))
    cogs_arr = np.array(forecast_result.get('costs', {}).get('cogs', [])[:n_periods] + [0] * max(0, n_periods - len(forecast_result.get('costs', {}).get('cogs', []))))
    cogs_make_arr = np.array((forecast_result.get('costs', {}).get('cogs_make', [])[:n_periods] if forecast_result.get('costs', {}).get('cogs_make') else []) + [0] * max(0, n_periods - len(forecast_result.get('costs', {}).get('cogs_make', []))))
    opex_arr = np.array(forecast_result.get('costs', {}).get('opex', [])[:n_periods] + [0] * max(0, n_periods - len(forecast_result.get('costs', {}).get('opex', []))))
    ebit_arr = np.array(forecast_result.get('profit', {}).get('ebit', [])[:n_periods] + [0] * max(0, n_periods - len(forecast_result.get('profit', {}).get('ebit', []))))
    
    # Parse period dates (vectorized where possible)
    period_dates = []
    for i, period in enumerate(timeline):
        try:
            period_dates.append(datetime.strptime(period, '%Y-%m'))
        except:
            period_dates.append(datetime.now() + relativedelta(months=i))
    
    # Vectorized calculations
    periods = np.arange(1, n_periods + 1)
    mfg_active_mask = (mfg_start_month > 0) & (periods >= mfg_start_month)
    mfg_ppe_added_mask = np.cumsum(mfg_active_mask.astype(int)) > 0
    
    # Working capital (vectorized)
    accounts_receivable_arr = total_rev_arr * dso / 30
    
    # Inventory calculations (vectorized)
    inventory_raw_materials_arr = np.where(
        mfg_active_mask & (cogs_make_arr > 0),
        mfg_wc,
        0
    )
    inventory_finished_goods_arr = np.where(
        mfg_active_mask & (cogs_make_arr > 0),
        (cogs_arr - cogs_make_arr) * dio / 30,
        cogs_arr * dio / 30
    )
    inventory_total_arr = inventory_raw_materials_arr + inventory_finished_goods_arr
    accounts_payable_arr = cogs_arr * dpo / 30
    
    # Depreciation and CAPEX (vectorized)
    monthly_depreciation_arr = total_rev_arr * depreciation_pct
    monthly_capex_arr = total_rev_arr * capex_pct
    
    # Manufacturing depreciation (vectorized)
    mfg_equip_dep_monthly = mfg_equipment / (10 * 12) if mfg_equipment > 0 else 0
    mfg_facility_dep_monthly = mfg_facility / (20 * 12) if mfg_facility > 0 else 0
    mfg_tooling_dep_monthly = mfg_tooling / (5 * 12) if mfg_tooling > 0 else 0
    mfg_dep_monthly = mfg_equip_dep_monthly + mfg_facility_dep_monthly + mfg_tooling_dep_monthly
    monthly_mfg_depreciation_arr = np.where(mfg_ppe_added_mask, mfg_dep_monthly, 0)
    
    # Cumulative calculations (vectorized)
    accumulated_depreciation_arr = np.cumsum(monthly_depreciation_arr)
    accumulated_mfg_depreciation_arr = np.cumsum(monthly_mfg_depreciation_arr)
    
    # PPE calculations (vectorized)
    ppe_base = assumptions.get('initial_ppe', 15_000_000)
    # PPE grows by cumulative CAPEX: base + sum of all CAPEX up to current period
    cumulative_capex = np.cumsum(monthly_capex_arr)
    ppe_other_arr = ppe_base + cumulative_capex
    
    ppe_manufacturing_arr = np.where(mfg_ppe_added_mask, total_mfg_ppe, 0)
    gross_ppe_arr = ppe_other_arr + ppe_manufacturing_arr
    total_accum_dep_arr = accumulated_depreciation_arr + accumulated_mfg_depreciation_arr
    net_ppe_arr = gross_ppe_arr - total_accum_dep_arr
    
    # Net income and retained earnings (vectorized)
    net_income_arr = ebit_arr * (1 - tax_rate)
    retained_earnings_arr = np.cumsum(net_income_arr)
    
    # Cash balance (vectorized)
    cash_flow_arr = net_income_arr - monthly_capex_arr + monthly_depreciation_arr
    cash_balance_arr = initial_cash + np.cumsum(cash_flow_arr)
    
    # Other calculations (vectorized)
    prepaid_arr = total_rev_arr * 0.02
    accrued_arr = opex_arr * 0.3
    total_current_assets_arr = np.maximum(cash_balance_arr, 0) + accounts_receivable_arr + inventory_total_arr + prepaid_arr
    total_current_liabilities_arr = accounts_payable_arr + accrued_arr
    
    # Build DataFrame from arrays (much faster than appending in loop)
    forecast_data = {
        'period_year': [pd.Timestamp(dt).year for dt in period_dates],
        'period_month': [pd.Timestamp(dt).month for dt in period_dates],
        'period_date': [dt.strftime('%Y-%m-%d') for dt in period_dates],
        'period_label': [dt.strftime('%b %Y') for dt in period_dates],
        'is_actual': [False] * n_periods,
        'data_type': ['Forecast'] * n_periods,
            
            # Assets - Current
        'cash_and_equivalents': np.maximum(cash_balance_arr, 0),
        'accounts_receivable': accounts_receivable_arr,
        'inventory_raw_materials': inventory_raw_materials_arr,
        'inventory_finished_goods': inventory_finished_goods_arr,
        'inventory_total': inventory_total_arr,
        'prepaid_expenses': prepaid_arr,
        'total_current_assets': total_current_assets_arr,
        
        # Assets - Non-Current
        'ppe_other': ppe_other_arr,
        'ppe_manufacturing': ppe_manufacturing_arr,
        'property_plant_equipment': gross_ppe_arr,
        'accumulated_depreciation': total_accum_dep_arr,
        'net_ppe': net_ppe_arr,
        'intangible_assets': [0] * n_periods,
        'total_non_current_assets': net_ppe_arr,
            
            # Liabilities - Current
        'accounts_payable': accounts_payable_arr,
        'accrued_expenses': accrued_arr,
        'short_term_debt': [0] * n_periods,
        'total_current_liabilities': total_current_liabilities_arr,
            
            # Liabilities - Non-Current
        'long_term_debt': [initial_debt] * n_periods,
        'total_non_current_liabilities': [initial_debt] * n_periods,
            
            # Equity
        'share_capital': [initial_equity] * n_periods,
        'retained_earnings': retained_earnings_arr,
        'total_equity': initial_equity + retained_earnings_arr,
            
            # Totals
        'total_assets': total_current_assets_arr + net_ppe_arr,
        'total_liabilities': total_current_liabilities_arr + np.array([initial_debt] * n_periods),
        'total_liabilities_and_equity': [0] * n_periods,
        'net_working_capital': (accounts_receivable_arr + inventory_total_arr) - accounts_payable_arr,
    }
    
    forecast_df = pd.DataFrame(forecast_data)
    all_rows.extend(forecast_df.to_dict('records'))
    
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df['total_liabilities_and_equity'] = df['total_liabilities'] + df['total_equity']
        df = df.sort_values(['period_year', 'period_month']).reset_index(drop=True)
    
    return df


def build_cash_flow(
    forecast_result: Dict[str, Any],
    assumptions: Dict[str, Any],
    balance_sheet_data: pd.DataFrame = None,
    manufacturing_strategy: Dict[str, Any] = None,
    historical_data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Build cash flow statement from forecast results.
    
    NEW in v9.2:
    - Includes historical data for comparison
    - Separates Manufacturing CAPEX from regular CAPEX
    - Shows Commissioning Costs as separate line item
    - Shows Manufacturing Working Capital investment
    """
    all_rows = []
    
    # Get assumptions
    depreciation_pct = assumptions.get('depreciation_pct_revenue', 0.03)
    capex_pct = assumptions.get('capex_pct_revenue', 0.05)
    tax_rate = assumptions.get('tax_rate', 0.28)
    if tax_rate > 1:
        tax_rate = tax_rate / 100
    
    # Working capital assumptions
    dso = assumptions.get('days_sales_outstanding', 45)
    dio = assumptions.get('days_inventory_outstanding', 60)
    dpo = assumptions.get('days_payable_outstanding', 30)
    
    # Manufacturing details
    mfg_equipment = 0
    mfg_facility = 0
    mfg_tooling = 0
    mfg_wc = 0
    mfg_start_month = 0
    mfg_completion_month = 0
    commissioning_costs = []
    if manufacturing_strategy:
        mfg_equipment = manufacturing_strategy.get('equipment_cost', 0)
        mfg_facility = manufacturing_strategy.get('facility_cost', 0)
        mfg_tooling = manufacturing_strategy.get('tooling_cost', 0)
        mfg_wc = manufacturing_strategy.get('working_capital', 0)
        mfg_start_month = manufacturing_strategy.get('commissioning_start_month', 0)
        mfg_completion_month = manufacturing_strategy.get('commissioning_completion_month', 0)
        commissioning_costs = manufacturing_strategy.get('commissioning_monthly_costs', [])
    
    total_mfg_ppe = mfg_equipment + mfg_facility + mfg_tooling
    
    # ==========================================================================
    # SECTION 1: HISTORICAL DATA (if provided)
    # ==========================================================================
    if historical_data is not None and not historical_data.empty:
        hist_df = historical_data.copy()
        
        if 'month' in hist_df.columns and 'period_date' not in hist_df.columns:
            hist_df['period_date'] = pd.to_datetime(hist_df['month'])
        elif 'period_date' in hist_df.columns:
            hist_df['period_date'] = pd.to_datetime(hist_df['period_date'])
        
        if 'period_date' in hist_df.columns:
            hist_df = hist_df.sort_values('period_date')
            prev_ar = 0
            prev_inv = 0
            prev_ap = 0
            beginning_cash = assumptions.get('initial_cash', 5_000_000)
            
            for _, row in hist_df.iterrows():
                period_date = row['period_date']
                if isinstance(period_date, str):
                    try:
                        period_date = datetime.strptime(period_date, '%Y-%m-%d')
                    except:
                        continue
                
                revenue = row.get('revenue', row.get('total_revenue', 0)) or 0
                cogs = row.get('cogs', row.get('total_cogs', 0)) or 0
                opex = row.get('opex', row.get('total_opex', 0)) or 0
                ebit = revenue - cogs - opex
                net_income = ebit * (1 - tax_rate)
                depreciation = revenue * depreciation_pct
                
                current_ar = revenue * dso / 30
                current_inv = cogs * dio / 30
                current_ap = cogs * dpo / 30
                
                change_in_receivables = current_ar - prev_ar
                change_in_inventory = current_inv - prev_inv
                change_in_payables = current_ap - prev_ap
                
                cash_from_operations = (net_income + depreciation - 
                                       change_in_receivables - 
                                       change_in_inventory + 
                                       change_in_payables)
                
                regular_capex = revenue * capex_pct
                cash_from_investing = -regular_capex
                cash_from_financing = 0
                net_change = cash_from_operations + cash_from_investing + cash_from_financing
                ending_cash = beginning_cash + net_change
                
                all_rows.append({
                    'period_year': period_date.year,
                    'period_month': period_date.month,
                    'period_date': period_date.strftime('%Y-%m-%d'),
                    'period_label': period_date.strftime('%b %Y'),
                    'is_actual': True,
                    'data_type': 'Actual',
                    
                    # Operating
                    'net_income': net_income,
                    'depreciation_amortization': depreciation,
                    'change_in_receivables': -change_in_receivables,
                    'change_in_inventory': -change_in_inventory,
                    'change_in_payables': change_in_payables,
                    'other_operating': 0,
                    'cash_from_operations': cash_from_operations,
                    
                    # Investing
                    'capital_expenditures': -regular_capex,
                    'mfg_ppe_investment': 0,
                    'mfg_commissioning_costs': 0,
                    'mfg_working_capital': 0,
                    'asset_sales': 0,
                    'other_investing': 0,
                    'cash_from_investing': cash_from_investing,
                    
                    # Financing
                    'debt_proceeds': 0,
                    'debt_repayments': 0,
                    'equity_issuance': 0,
                    'dividends_paid': 0,
                    'other_financing': 0,
                    'cash_from_financing': cash_from_financing,
                    
                    # Summary
                    'net_change_in_cash': net_change,
                    'beginning_cash': beginning_cash,
                    'ending_cash': ending_cash,
                })
                
                prev_ar = current_ar
                prev_inv = current_inv
                prev_ap = current_ap
                beginning_cash = ending_cash
    
    # ==========================================================================
    # SECTION 2: FORECAST DATA
    # ==========================================================================
    timeline = forecast_result.get('timeline', [])
    total_rev_arr = forecast_result.get('revenue', {}).get('total', [])
    cogs_arr = forecast_result.get('costs', {}).get('cogs', [])
    opex_arr = forecast_result.get('costs', {}).get('opex', [])
    ebit_arr = forecast_result.get('profit', {}).get('ebit', [])
    commissioning_arr = forecast_result.get('costs', {}).get('commissioning', [])
    
    # Get initial values (from end of historical if exists, else from assumptions)
    if all_rows:
        prev_ar = all_rows[-1].get('accounts_receivable', 0) if 'accounts_receivable' in all_rows[-1] else 0
        prev_inv = 0
        prev_ap = 0
        beginning_cash = all_rows[-1].get('ending_cash', assumptions.get('initial_cash', 5_000_000))
    else:
        prev_ar = 0
        prev_inv = 0
        prev_ap = 0
        beginning_cash = assumptions.get('initial_cash', 5_000_000)
    
    mfg_ppe_invested = False
    mfg_wc_invested = False
    
    for i, period in enumerate(timeline):
        try:
            period_date = datetime.strptime(period, '%Y-%m')
        except:
            period_date = datetime.now() + relativedelta(months=i)
        
        # Get values
        revenue = total_rev_arr[i] if i < len(total_rev_arr) else 0
        cogs = cogs_arr[i] if i < len(cogs_arr) else 0
        opex = opex_arr[i] if i < len(opex_arr) else 0
        ebit = ebit_arr[i] if i < len(ebit_arr) else 0
        
        # Calculate net income
        net_income = ebit * (1 - tax_rate)
        
        # Non-cash items
        depreciation = revenue * depreciation_pct
        
        # Working capital changes
        current_ar = revenue * dso / 30
        current_inv = cogs * dio / 30
        current_ap = cogs * dpo / 30
        
        change_in_receivables = current_ar - prev_ar
        change_in_inventory = current_inv - prev_inv
        change_in_payables = current_ap - prev_ap
        
        # Operating cash flow
        cash_from_operations = (net_income + depreciation - 
                               change_in_receivables - 
                               change_in_inventory + 
                               change_in_payables)
        
        # ==========================================================================
        # INVESTING ACTIVITIES (with manufacturing detail)
        # ==========================================================================
        regular_capex = revenue * capex_pct
        
        # Manufacturing PPE investment (at commissioning start)
        mfg_ppe_this_month = 0
        if mfg_start_month > 0 and (i + 1) == mfg_start_month and not mfg_ppe_invested:
            mfg_ppe_this_month = total_mfg_ppe
            mfg_ppe_invested = True
        
        # Commissioning costs (during commissioning period)
        mfg_commissioning_this_month = 0
        if commissioning_arr and i < len(commissioning_arr):
            mfg_commissioning_this_month = commissioning_arr[i]
        elif commissioning_costs:
            # Calculate from commissioning schedule
            month_in_commissioning = (i + 1) - mfg_start_month
            if 0 <= month_in_commissioning < len(commissioning_costs):
                mfg_commissioning_this_month = commissioning_costs[month_in_commissioning]
        
        # Manufacturing working capital (at manufacturing go-live)
        mfg_wc_this_month = 0
        if mfg_completion_month > 0 and (i + 1) == mfg_completion_month and not mfg_wc_invested:
            mfg_wc_this_month = mfg_wc
            mfg_wc_invested = True
        
        total_investing = -(regular_capex + mfg_ppe_this_month + mfg_commissioning_this_month + mfg_wc_this_month)
        
        # Financing (placeholder)
        cash_from_financing = 0
        
        # Net change
        net_change_in_cash = cash_from_operations + total_investing + cash_from_financing
        ending_cash = beginning_cash + net_change_in_cash
        
        all_rows.append({
            'period_year': period_date.year,
            'period_month': period_date.month,
            'period_date': period_date.strftime('%Y-%m-%d'),
            'period_label': period_date.strftime('%b %Y'),
            'is_actual': False,
            'data_type': 'Forecast',
            
            # Operating Activities
            'net_income': net_income,
            'depreciation_amortization': depreciation,
            'change_in_receivables': -change_in_receivables,
            'change_in_inventory': -change_in_inventory,
            'change_in_payables': change_in_payables,
            'other_operating': 0,
            'cash_from_operations': cash_from_operations,
            
            # Investing Activities (with manufacturing detail)
            'capital_expenditures': -regular_capex,
            'mfg_ppe_investment': -mfg_ppe_this_month,
            'mfg_commissioning_costs': -mfg_commissioning_this_month,
            'mfg_working_capital': -mfg_wc_this_month,
            'asset_sales': 0,
            'other_investing': 0,
            'cash_from_investing': total_investing,
            
            # Financing Activities
            'debt_proceeds': 0,
            'debt_repayments': 0,
            'equity_issuance': 0,
            'dividends_paid': 0,
            'other_financing': 0,
            'cash_from_financing': cash_from_financing,
            
            # Summary
            'net_change_in_cash': net_change_in_cash,
            'beginning_cash': beginning_cash,
            'ending_cash': ending_cash,
        })
        
        # Update for next iteration
        prev_ar = current_ar
        prev_inv = current_inv
        prev_ap = current_ap
        beginning_cash = ending_cash
    
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(['period_year', 'period_month']).reset_index(drop=True)
    
    return df


def render_balance_sheet_table(data: pd.DataFrame, view_mode: str = 'annual'):
    """Render balance sheet in professional format with manufacturing detail."""
    if data.empty:
        st.warning("No balance sheet data available.")
        return
    
    # Show data type legend if we have both actuals and forecast
    has_actuals = 'is_actual' in data.columns and data['is_actual'].any()
    has_forecast = 'is_actual' in data.columns and (~data['is_actual']).any()
    
    if has_actuals and has_forecast:
        st.caption("🔵 = Actual | 🟡 = Forecast")
    
    # Aggregate to annual if needed
    if view_mode == 'annual':
        # Group by year, preserving is_actual flag
        numeric_cols = [c for c in data.columns if c not in 
                       ['period_year', 'period_month', 'period_date', 'period_label', 
                        'is_actual', 'data_type']]
        
        annual = data.groupby('period_year').agg({
            **{col: 'sum' if data[col].dtype in ['int64', 'float64'] else 'first' 
               for col in numeric_cols},
            'is_actual': 'any'  # If any month is actual, the year is actual
        }).reset_index()
        
        # Create period_label with "A" marker for historical years
        annual['period_label'] = annual.apply(
            lambda row: f"FY{row['period_year']} A" if row.get('is_actual', False) else f"FY{row['period_year']}",
            axis=1
        )
        data = annual
    
    periods = data['period_label'].tolist()
    is_actual_flags = data['is_actual'].tolist() if 'is_actual' in data.columns else [False] * len(periods)
    
    # Check if we have manufacturing-specific columns
    has_mfg_ppe = 'ppe_manufacturing' in data.columns and data['ppe_manufacturing'].sum() > 0
    has_inventory_split = 'inventory_raw_materials' in data.columns and data['inventory_raw_materials'].sum() > 0
    
    # Define line items dynamically based on available data
    line_items = [
        ('ASSETS', None, 'header'),
        ('Current Assets', None, 'subheader'),
        ('Cash and Equivalents', 'cash_and_equivalents', 'detail'),
        ('Accounts Receivable', 'accounts_receivable', 'detail'),
    ]
    
    # Inventory section - split if manufacturing active
    if has_inventory_split:
        line_items.extend([
            ('Inventory - Raw Materials (Mfg)', 'inventory_raw_materials', 'detail'),
            ('Inventory - Finished Goods', 'inventory_finished_goods', 'detail'),
            ('Total Inventory', 'inventory_total', 'subtotal'),
        ])
    else:
        line_items.append(('Inventory', 'inventory_total', 'detail'))
    
    line_items.extend([
        ('Prepaid Expenses', 'prepaid_expenses', 'detail'),
        ('Total Current Assets', 'total_current_assets', 'subtotal'),
        ('', None, 'spacer'),
        ('Non-Current Assets', None, 'subheader'),
    ])
    
    # PPE section - split if manufacturing active
    if has_mfg_ppe:
        line_items.extend([
            ('PP&E - Other', 'ppe_other', 'detail'),
            ('PP&E - Manufacturing', 'ppe_manufacturing', 'detail'),
            ('Gross PP&E', 'property_plant_equipment', 'subtotal'),
        ])
    else:
        line_items.append(('Property, Plant & Equipment', 'property_plant_equipment', 'detail'))
    
    line_items.extend([
        ('Less: Accumulated Depreciation', 'accumulated_depreciation', 'detail'),
        ('Net PP&E', 'net_ppe', 'detail'),
        ('Total Non-Current Assets', 'total_non_current_assets', 'subtotal'),
        ('', None, 'spacer'),
        ('TOTAL ASSETS', 'total_assets', 'total'),
        ('', None, 'spacer'),
        ('LIABILITIES', None, 'header'),
        ('Current Liabilities', None, 'subheader'),
        ('Accounts Payable', 'accounts_payable', 'detail'),
        ('Accrued Expenses', 'accrued_expenses', 'detail'),
        ('Short-term Debt', 'short_term_debt', 'detail'),
        ('Total Current Liabilities', 'total_current_liabilities', 'subtotal'),
        ('', None, 'spacer'),
        ('Non-Current Liabilities', None, 'subheader'),
        ('Long-term Debt', 'long_term_debt', 'detail'),
        ('Total Non-Current Liabilities', 'total_non_current_liabilities', 'subtotal'),
        ('', None, 'spacer'),
        ('TOTAL LIABILITIES', 'total_liabilities', 'subtotal'),
        ('', None, 'spacer'),
        ('EQUITY', None, 'header'),
        ('Share Capital', 'share_capital', 'detail'),
        ('Retained Earnings', 'retained_earnings', 'detail'),
        ('TOTAL EQUITY', 'total_equity', 'subtotal'),
        ('', None, 'spacer'),
        ('TOTAL LIABILITIES & EQUITY', 'total_liabilities_and_equity', 'total'),
        ('', None, 'spacer'),
        ('Net Working Capital', 'net_working_capital', 'memo'),
    ])
    
    # Build HTML table
    html = '<table style="width:100%; border-collapse:collapse; font-size:0.85rem;">'
    html += '<tr style="background:#1E1E1E;">'
    html += '<th style="text-align:left; padding:8px; border-bottom:2px solid #404040;">Line Item</th>'
    for i, period in enumerate(periods):
        # Color code header based on actual vs forecast
        if is_actual_flags[i]:
            # Historical/Actual columns - different background and border
            header_style = "background:rgba(100,116,139,0.3); color:#FFFFFF; border-left:2px solid rgba(100,116,139,0.5); border-right:2px solid rgba(100,116,139,0.5);"
        else:
            # Forecast columns - default styling
            header_style = "background:#1E1E1E; color:#D4A537;"
        html += f'<th style="text-align:right; padding:8px; border-bottom:2px solid #404040; {header_style}">{period}</th>'
    html += '</tr>'
    
    for label, col, row_type in line_items:
        if row_type == 'spacer':
            html += '<tr><td colspan="100%" style="height:8px;"></td></tr>'
            continue
        
        if row_type == 'header':
            style = 'background:#2a2a2a; font-weight:bold; color:#D4A537;'
        elif row_type == 'subheader':
            style = 'font-weight:bold; color:#FFFFFF;'
        elif row_type == 'subtotal':
            style = 'background:rgba(212,165,55,0.1); font-weight:600; color:#D4A537;'
        elif row_type == 'total':
            style = 'background:#D4A537; font-weight:bold; color:#000;'
        elif row_type == 'memo':
            style = 'font-style:italic; color:#888;'
        else:
            style = 'color:#888;'
        
        html += f'<tr style="{style}">'
        html += f'<td style="padding:6px 8px;">{label}</td>'
        
        # Vectorized: Use direct column access instead of iterrows()
        if col and col in data.columns:
            values = data[col].fillna(0).values
            is_total = row_type in ['subtotal', 'total'] or 'Total' in label
            for idx, value in enumerate(values):
                if is_total:
                    formatted = f'<b>{format_currency(value)}</b>'
                else:
                    formatted = format_currency(value)
                
                # Add distinct background and borders for actuals
                if is_actual_flags[idx]:
                    cell_style = "background:rgba(100,116,139,0.15) !important; border-left:2px solid rgba(100,116,139,0.4); border-right:2px solid rgba(100,116,139,0.4);"
                else:
                    cell_style = ""
                html += f'<td style=\"text-align:right; padding:6px 8px; {cell_style}\">{formatted}</td>'
        else:
            # No column data - add empty cells
            for idx in range(len(data)):
                if is_actual_flags[idx]:
                    cell_style = "background:rgba(100,116,139,0.15) !important; border-left:2px solid rgba(100,116,139,0.4); border-right:2px solid rgba(100,116,139,0.4);"
                else:
                    cell_style = ""
                html += f'<td style="text-align:right; padding:6px 8px; {cell_style}"></td>'
        
        html += '</tr>'
    
    html += '</table>'
    st.markdown(html, unsafe_allow_html=True)


def render_cash_flow_table(data: pd.DataFrame, view_mode: str = 'annual'):
    """Render cash flow statement in professional format with manufacturing detail."""
    if data.empty:
        st.warning("No cash flow data available.")
        return
    
    # Show data type legend if we have both actuals and forecast
    has_actuals = 'is_actual' in data.columns and data['is_actual'].any()
    has_forecast = 'is_actual' in data.columns and (~data['is_actual']).any()
    
    if has_actuals and has_forecast:
        st.caption("🔵 = Actual | 🟡 = Forecast")
    
    # Aggregate to annual if needed
    if view_mode == 'annual':
        numeric_cols = [c for c in data.columns if c not in 
                       ['period_year', 'period_month', 'period_date', 'period_label', 
                        'is_actual', 'data_type', 'beginning_cash', 'ending_cash']]
        
        # Group by year, preserving is_actual flag
        annual = data.groupby('period_year').agg({
            **{col: 'sum' if data[col].dtype in ['int64', 'float64'] else 'first' 
               for col in numeric_cols},
            'is_actual': 'any'  # If any month is actual, the year is actual
        }).reset_index()
        
        # Get ending cash from last month of each year
        ending_data = data.loc[data.groupby('period_year')['period_month'].idxmax()][
            ['period_year', 'ending_cash']
        ]
        annual = annual.merge(ending_data, on='period_year', how='left')
        
        # Create period_label with "A" marker for historical years
        annual['period_label'] = annual.apply(
            lambda row: f"FY{row['period_year']} A" if row.get('is_actual', False) else f"FY{row['period_year']}",
            axis=1
        )
        data = annual
    
    periods = data['period_label'].tolist()
    is_actual_flags = data['is_actual'].tolist() if 'is_actual' in data.columns else [False] * len(periods)
    
    # Check if we have manufacturing-specific investment items
    has_mfg_investments = (
        'mfg_ppe_investment' in data.columns and abs(data['mfg_ppe_investment'].sum()) > 0
    )
    
    # Build line items dynamically
    line_items = [
        ('OPERATING ACTIVITIES', None, 'header'),
        ('Net Income', 'net_income', 'detail'),
        ('Depreciation & Amortization', 'depreciation_amortization', 'detail'),
        ('Changes in Working Capital:', None, 'subheader'),
        ('  (Increase)/Decrease in Receivables', 'change_in_receivables', 'detail'),
        ('  (Increase)/Decrease in Inventory', 'change_in_inventory', 'detail'),
        ('  Increase/(Decrease) in Payables', 'change_in_payables', 'detail'),
        ('Cash from Operations', 'cash_from_operations', 'subtotal'),
        ('', None, 'spacer'),
        ('INVESTING ACTIVITIES', None, 'header'),
        ('Capital Expenditures (Maintenance)', 'capital_expenditures', 'detail'),
    ]
    
    # Add manufacturing investment items if present
    if has_mfg_investments:
        line_items.extend([
            ('Manufacturing PP&E Investment', 'mfg_ppe_investment', 'detail'),
            ('Manufacturing Commissioning Costs', 'mfg_commissioning_costs', 'detail'),
            ('Manufacturing Working Capital', 'mfg_working_capital', 'detail'),
        ])
    
    line_items.extend([
        ('Asset Sales', 'asset_sales', 'detail'),
        ('Cash from Investing', 'cash_from_investing', 'subtotal'),
        ('', None, 'spacer'),
        ('FINANCING ACTIVITIES', None, 'header'),
        ('Debt Proceeds', 'debt_proceeds', 'detail'),
        ('Debt Repayments', 'debt_repayments', 'detail'),
        ('Equity Issuance', 'equity_issuance', 'detail'),
        ('Dividends Paid', 'dividends_paid', 'detail'),
        ('Cash from Financing', 'cash_from_financing', 'subtotal'),
        ('', None, 'spacer'),
        ('NET CHANGE IN CASH', 'net_change_in_cash', 'total'),
        ('Ending Cash Balance', 'ending_cash', 'memo'),
    ])
    
    html = '<table style="width:100%; border-collapse:collapse; font-size:0.85rem;">'
    html += '<tr style="background:#1E1E1E;">'
    html += '<th style="text-align:left; padding:8px; border-bottom:2px solid #404040;">Line Item</th>'
    for i, period in enumerate(periods):
        # Color code header based on actual vs forecast
        if is_actual_flags[i]:
            # Historical/Actual columns - different background and border
            header_style = "background:rgba(100,116,139,0.3); color:#FFFFFF; border-left:2px solid rgba(100,116,139,0.5); border-right:2px solid rgba(100,116,139,0.5);"
        else:
            # Forecast columns - default styling
            header_style = "background:#1E1E1E; color:#D4A537;"
        html += f'<th style="text-align:right; padding:8px; border-bottom:2px solid #404040; {header_style}">{period}</th>'
    html += '</tr>'
    
    for label, col, row_type in line_items:
        if row_type == 'spacer':
            html += '<tr><td colspan="100%" style="height:8px;"></td></tr>'
            continue
        
        if row_type == 'header':
            style = 'background:#2a2a2a; font-weight:bold; color:#D4A537;'
        elif row_type == 'subheader':
            style = 'font-weight:bold; color:#FFFFFF;'
        elif row_type == 'subtotal':
            style = 'background:rgba(212,165,55,0.1); font-weight:600; color:#D4A537;'
        elif row_type == 'total':
            style = 'background:#D4A537; font-weight:bold; color:#000;'
        elif row_type == 'memo':
            style = 'font-style:italic; color:#888;'
        else:
            style = 'color:#888;'
        
        html += f'<tr style="{style}">'
        html += f'<td style="padding:6px 8px;">{label}</td>'
        
        # Vectorized: Use direct column access instead of iterrows()
        if col and col in data.columns:
            values = data[col].fillna(0).values
            is_total = row_type in ['subtotal', 'total']
            for idx, value in enumerate(values):
                if is_total:
                    formatted = f'<b>{format_currency(value)}</b>'
                else:
                    formatted = format_currency(value)
                
                # Add distinct background and borders for actuals
                if is_actual_flags[idx]:
                    cell_style = "background:rgba(100,116,139,0.15) !important; border-left:2px solid rgba(100,116,139,0.4); border-right:2px solid rgba(100,116,139,0.4);"
                else:
                    cell_style = ""
                html += f'<td style="text-align:right; padding:6px 8px; {cell_style}">{formatted}</td>'
        else:
            # No column data - add empty cells
            for idx in range(len(data)):
                if is_actual_flags[idx]:
                    cell_style = "background:rgba(100,116,139,0.15) !important; border-left:2px solid rgba(100,116,139,0.4); border-right:2px solid rgba(100,116,139,0.4);"
                else:
                    cell_style = ""
                html += f'<td style="text-align:right; padding:6px 8px; {cell_style}"></td>'
        
        html += '</tr>'
    
    html += '</table>'
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# INCOME STATEMENT RENDERER
# =============================================================================

def _render_historical_validation_check(data: pd.DataFrame):
    """
    Render validation check comparing imported net income to calculated net income.
    Shows at bottom of income statement for actuals periods.
    """
    # Filter to actuals only
    if 'is_actual' not in data.columns:
        return
    
    actuals_data = data[data['is_actual'] == True].copy()
    
    if actuals_data.empty:
        return
    
    # Get imported net income from historical data (from imported financial statements)
    if 'net_income' not in actuals_data.columns:
        return
    
    imported_net_income = actuals_data['net_income'].fillna(0)
    
    # Calculate net income from components: Revenue - COGS - OPEX - Interest - Tax
    # This should match the imported net income if the data is correct
    revenue = actuals_data.get('total_revenue', pd.Series([0] * len(actuals_data))).fillna(0)
    cogs = actuals_data.get('total_cogs', pd.Series([0] * len(actuals_data))).fillna(0)
    opex = actuals_data.get('total_opex', pd.Series([0] * len(actuals_data))).fillna(0)
    interest = actuals_data.get('interest_expense', pd.Series([0] * len(actuals_data))).fillna(0)
    tax = actuals_data.get('tax_expense', pd.Series([0] * len(actuals_data))).fillna(0)
    
    calculated_net_income = revenue - cogs - opex - interest - tax
    
    # Aggregate totals
    imported_total = float(imported_net_income.sum())
    calculated_total = float(calculated_net_income.sum())
    
    # Compare (allow small rounding differences)
    # Use 1% tolerance or minimum 100 currency units, whichever is larger
    tolerance = max(abs(imported_total) * 0.01, 100)
    difference = abs(imported_total - calculated_total)
    agrees = difference <= tolerance
    
    # Display validation
    st.markdown("---")
    
    if agrees:
        st.success(
            f"✅ **Historical Validation:** "
            f"Net Income per Financial Statements: {format_currency(imported_total)} | "
            f"Calculated Net Income: {format_currency(calculated_total)} | "
            f"**Historics agree with actual upload**"
        )
    else:
        st.warning(
            f"⚠️ **Historical Validation:** "
            f"Net Income per Financial Statements: {format_currency(imported_total)} | "
            f"Calculated Net Income: {format_currency(calculated_total)} | "
            f"Difference: {format_currency(difference)} | "
            f"**Discrepancy detected - please review**"
        )


def render_income_statement_table(
    data: pd.DataFrame,
    show_revenue_detail: bool = True,
    show_opex_detail: bool = True,
    view_mode: str = 'monthly'
):
    """Render the income statement as a formatted HTML table."""
    
    if data.empty:
        st.warning("No financial data available. Please run a forecast first.")
        return
    
    # Get periods to display
    periods = data['period_label'].tolist()
    max_cols = 8 if view_mode == 'annual' else 12
    
    if len(periods) > max_cols:
        periods = periods[:max_cols]
        st.caption(f"📊 Showing first {max_cols} periods. Export for complete data.")
    
    display_data = data[data['period_label'].isin(periods)]
    
    # Build CSS
    css = f"""
    <style>
    .fs-container {{
        overflow-x: auto;
        margin: 1rem 0;
    }}
    .fs-table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'Segoe UI', Tahoma, sans-serif;
        font-size: 12px;
        min-width: 800px;
    }}
    .fs-table th {{
        background: linear-gradient(180deg, {DARK_BG}, {DARKER_BG});
        color: {GOLD};
        padding: 10px 8px;
        text-align: right;
        font-weight: 600;
        border-bottom: 2px solid {GOLD};
        white-space: nowrap;
    }}
    .fs-table th.actual-header {{
        background: linear-gradient(180deg, rgba(100, 116, 139, 0.3), rgba(100, 116, 139, 0.2)) !important;
        color: {TEXT_WHITE};
        border-bottom: 2px solid rgba(100, 116, 139, 0.5);
    }}
    .fs-table th:first-child {{
        text-align: left;
        min-width: 200px;
    }}
    .fs-table td {{
        padding: 6px 8px;
        text-align: right;
        border-bottom: 1px solid {BORDER_COLOR};
        color: {TEXT_WHITE};
    }}
    .fs-table td:first-child {{
        text-align: left;
    }}
    .fs-table tr:hover td {{
        background: {GOLD_LIGHT} !important;
    }}
    .fs-table .section-header td {{
        background: {GOLD_LIGHT} !important;
        color: {GOLD};
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-top: 2px solid {BORDER_COLOR};
        padding: 8px;
    }}
    .fs-table .subtotal td {{
        color: {GOLD};
        font-weight: 600;
        border-top: 1px solid {GOLD_DARK};
        background: rgba(212, 165, 55, 0.05);
    }}
    .fs-table .total td {{
        color: {TEXT_WHITE};
        font-weight: 700;
        background: rgba(212, 165, 55, 0.15) !important;
        border-top: 2px solid {GOLD};
        border-bottom: 2px solid {GOLD};
        font-size: 13px;
    }}
    .fs-table .detail td {{
        color: {TEXT_MUTED};
        font-size: 11px;
    }}
    .fs-table .actual td {{
        background: rgba(100, 116, 139, 0.15) !important;
        border-left: 2px solid rgba(100, 116, 139, 0.4);
        border-right: 2px solid rgba(100, 116, 139, 0.4);
    }}
    .fs-table .actual:first-child td {{
        border-left: none;
    }}
    .fs-table .actual:last-child td {{
        border-right: none;
    }}
    .indent-1 {{ padding-left: 20px !important; }}
    .indent-2 {{ padding-left: 40px !important; }}
    .negative {{ color: {RED} !important; }}
    .positive {{ color: {GREEN} !important; }}
    .pct {{ color: {TEXT_MUTED}; font-size: 10px; margin-left: 4px; }}
    .divider td {{ padding: 4px; border: none; background: transparent !important; }}
    </style>
    """
    
    # Build table
    header_html = f"""
    {css}
    <div class="fs-container">
    <table class="fs-table">
    <thead><tr><th>Line Item</th>
    """
    
    # Vectorized: Use direct column access instead of iterrows()
    period_labels = display_data['period_label'].values
    is_actual_flags = display_data.get('is_actual', pd.Series([False] * len(display_data))).values
    
    # Build header with actual columns marked
    for period_label, is_actual in zip(period_labels, is_actual_flags):
        # Format period label with "A" marker for historical periods
        if is_actual:
            # For annual view, period_label should already have "A" (e.g., "FY2022 A")
            # For monthly view, add "(A)" marker (e.g., "Dec 2022 (A)")
            if view_mode == 'annual':
                # Check if "A" is already in the label, if not add it
                if " A" not in period_label and not period_label.endswith(" A"):
                    # Extract year and add "A"
                    if period_label.startswith("FY"):
                        period_label_display = f"{period_label} A"
                    else:
                        period_label_display = period_label  # Keep as is if format unexpected
                else:
                    period_label_display = period_label
            else:
                # Monthly view - add "(A)" if not already present
                if " (A)" not in period_label and not period_label.endswith(" (A)"):
                    period_label_display = f"{period_label} (A)"
                else:
                    period_label_display = period_label
        else:
            period_label_display = period_label
        
        actual_class = "actual-header" if is_actual else ""
        header_html += f'<th class="{actual_class}">{period_label_display}</th>'
    header_html += "</tr></thead><tbody>"
    
    # Helper functions
    def make_row(label: str, field: str, row_class: str = "", indent: int = 0, 
                 show_as_negative: bool = False, show_pct_of: str = None):
        cells = f'<td class="indent-{indent}">{label}</td>'
        
        # Vectorized: Use direct column access instead of iterrows()
        if field in display_data.columns:
            values = display_data[field].fillna(0).values
        else:
            values = np.zeros(len(display_data))
        
        # Get percentage base values if needed
        pct_base_values = None
        if show_pct_of and show_pct_of in display_data.columns:
            pct_base_values = display_data[show_pct_of].fillna(0).values
        
        # Get actual flags
        is_actual_flags = display_data.get('is_actual', pd.Series([False] * len(display_data))).values
        
        for idx, value in enumerate(values):
            formatted, val_class = format_value(value, is_negative_expense=show_as_negative)
            
            pct_html = ""
            if show_pct_of and pct_base_values is not None and pct_base_values[idx] > 0:
                pct = value / pct_base_values[idx] * 100
                pct_html = f'<span class="pct">({pct:.0f}%)</span>'
            
            actual_class = "actual" if is_actual_flags[idx] else ""
            cells += f'<td class="{val_class} {actual_class}">{formatted}{pct_html}</td>'
        
        return f'<tr class="{row_class}">{cells}</tr>'
    
    def make_section_header(title: str):
        return f'<tr class="section-header"><td colspan="{len(periods) + 1}">{title}</td></tr>'
    
    def make_divider():
        return f'<tr class="divider"><td colspan="{len(periods) + 1}"></td></tr>'
    
    # Build table body
    body_html = ""
    
    # REVENUE SECTION
    body_html += make_section_header("📈 REVENUE")
    
    # Check if manufacturing data is available for revenue split
    has_mfg_revenue = 'revenue_manufactured' in display_data.columns and display_data['revenue_manufactured'].sum() > 0
    
    if show_revenue_detail:
        body_html += make_row("Existing Customers", "revenue_existing", "subtotal", indent=1)
        body_html += make_row("Wear Parts", "rev_wear_existing", "detail", indent=2)
        body_html += make_row("Refurbishment & Service", "rev_service_existing", "detail", indent=2)
        body_html += make_divider()
        body_html += make_row("Prospective Customers", "revenue_prospect", "subtotal", indent=1)
        body_html += make_row("Wear Parts", "rev_wear_prospect", "detail", indent=2)
        body_html += make_row("Refurbishment & Service", "rev_service_prospect", "detail", indent=2)
        
        # NEW: Show bought vs manufactured revenue split if manufacturing is active
        if has_mfg_revenue:
            body_html += make_divider()
            body_html += make_row("Revenue - Purchased Products", "revenue_bought", "subtotal", indent=1)
            body_html += make_row("Revenue - Manufactured Products", "revenue_manufactured", "subtotal", indent=1)
        
        body_html += make_divider()
    
    body_html += make_row("TOTAL REVENUE", "total_revenue", "total")
    
    # COGS SECTION - with manufacturing breakdown if available
    body_html += make_section_header("📦 COST OF GOODS SOLD")
    
    # Check if we have manufacturing COGS breakdown
    has_mfg_cogs = 'cogs_make' in display_data.columns and display_data['cogs_make'].sum() != 0
    has_var_overhead = 'mfg_variable_overhead' in display_data.columns and display_data['mfg_variable_overhead'].sum() != 0
    
    if has_mfg_cogs:
        body_html += make_row("COGS - Purchased Products", "cogs_buy", "detail", indent=1, show_as_negative=True)
        body_html += make_row("COGS - Manufactured (Direct)", "cogs_make", "detail", indent=1, show_as_negative=True)
        
        # Show variable and fixed overhead separately if available
        if has_var_overhead:
            body_html += make_row("Mfg Variable Overhead", "mfg_variable_overhead", "detail", indent=1, show_as_negative=True)
            body_html += make_row("Mfg Fixed Overhead", "mfg_fixed_overhead", "detail", indent=1, show_as_negative=True)
        else:
            # Fall back to combined overhead
            body_html += make_row("Manufacturing Overhead", "mfg_overhead", "detail", indent=1, show_as_negative=True)
        
        body_html += make_row("Manufacturing Depreciation", "mfg_depreciation", "detail", indent=1, show_as_negative=True)
        body_html += make_divider()
    
    body_html += make_row("Total Cost of Goods Sold", "total_cogs", "subtotal", show_as_negative=True)
    
    # GROSS PROFIT
    body_html += make_divider()
    body_html += make_row("GROSS PROFIT", "total_gross_profit", "total", show_pct_of="total_revenue")
    
    # OPERATING EXPENSES
    body_html += make_section_header("💼 OPERATING EXPENSES")
    
    if show_opex_detail:
        body_html += make_row("Personnel & Salaries", "opex_personnel", "detail", indent=1, show_as_negative=True)
        body_html += make_row("Facilities & Utilities", "opex_facilities", "detail", indent=1, show_as_negative=True)
        body_html += make_row("Administrative", "opex_admin", "detail", indent=1, show_as_negative=True)
        body_html += make_row("Sales & Marketing", "opex_sales", "detail", indent=1, show_as_negative=True)
        body_html += make_row("Other Operating", "opex_other", "detail", indent=1, show_as_negative=True)
        body_html += make_divider()
    
    body_html += make_row("Total Operating Expenses", "total_opex", "subtotal", show_as_negative=True)
    
    # OPERATING RESULTS
    body_html += make_section_header("📊 OPERATING RESULTS")
    body_html += make_row("EBITDA", "ebitda", "subtotal", show_pct_of="total_revenue")
    body_html += make_row("Depreciation & Amortization", "depreciation", "detail", indent=1, show_as_negative=True)
    body_html += make_divider()
    body_html += make_row("EBIT (Operating Income)", "ebit", "subtotal", show_pct_of="total_revenue")
    body_html += make_row("Interest Expense", "interest_expense", "detail", indent=1, show_as_negative=True)
    body_html += make_divider()
    body_html += make_row("EBT (Pre-Tax Income)", "ebt", "subtotal")
    body_html += make_row("Income Tax Expense", "tax_expense", "detail", indent=1, show_as_negative=True)
    
    # NET INCOME
    body_html += make_divider()
    body_html += make_row("NET INCOME", "net_income", "total", show_pct_of="total_revenue")
    
    # Close table
    body_html += "</tbody></table></div>"
    
    # Render
    st.markdown(header_html + body_html, unsafe_allow_html=True)
    
    # VALIDATION CHECK: Compare imported vs calculated net income for actuals
    _render_historical_validation_check(display_data)


def _merge_detailed_line_items_into_monthly_data(
    monthly_data: pd.DataFrame,
    detailed_line_items: pd.DataFrame,
    historical_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge detailed line items from historical data into monthly financial data.
    
    This ensures that historical periods show detailed line item breakdowns
    (e.g., Personnel & Salaries, Facilities & Utilities) instead of just R-0.
    
    Args:
        monthly_data: Monthly financial data DataFrame
        detailed_line_items: DataFrame with detailed line items from database
        historical_df: Historical summary data DataFrame
    
    Returns:
        Updated monthly_data with detailed line items filled in for historical periods
    """
    if detailed_line_items.empty or monthly_data.empty:
        return monthly_data
    
    try:
        # Ensure period_date is datetime in both DataFrames
        if 'period_date' in monthly_data.columns:
            monthly_data['period_date'] = pd.to_datetime(monthly_data['period_date'], errors='coerce')
        else:
            # Create period_date from period_year and period_month if needed
            if 'period_year' in monthly_data.columns and 'period_month' in monthly_data.columns:
                monthly_data['period_date'] = pd.to_datetime(
                    monthly_data.apply(
                        lambda row: f"{int(row['period_year'])}-{int(row['period_month']):02d}-01",
                        axis=1
                    ),
                    errors='coerce'
                )
        
        detailed_line_items['period_date'] = pd.to_datetime(detailed_line_items['period_date'], errors='coerce')
        
        # Filter to only historical periods (is_actual == True)
        historical_periods = monthly_data[monthly_data.get('is_actual', False) == True].copy()
        
        if historical_periods.empty:
            return monthly_data
        
        # Map line item names to monthly_data columns
        line_item_mapping = {
            # Operating Expenses
            'Personnel & Salaries': 'opex_personnel',
            'Personnel': 'opex_personnel',
            'Salaries': 'opex_personnel',
            'Wages': 'opex_personnel',
            'Facilities & Utilities': 'opex_facilities',
            'Facilities': 'opex_facilities',
            'Utilities': 'opex_facilities',
            'Rent': 'opex_facilities',
            'Administrative': 'opex_admin',
            'Admin': 'opex_admin',
            'General expenses': 'opex_admin',
            'Sales & Marketing': 'opex_sales',
            'Sales': 'opex_sales',
            'Marketing': 'opex_sales',
            'Other Operating': 'opex_other',
            'Other': 'opex_other',
            'Cleaning': 'opex_other',
            # Revenue (if detailed)
            'Wear Parts': 'rev_wear_existing',
            'Refurbishment & Service': 'rev_service_existing',
            'Refurbishment': 'rev_service_existing',
            'Service': 'rev_service_existing',
        }
        
        # Group detailed line items by period_date and category
        for period_date, period_df in detailed_line_items.groupby('period_date'):
            # Find matching row in monthly_data
            matching_rows = monthly_data[
                (monthly_data['period_date'] == period_date) &
                (monthly_data.get('is_actual', False) == True)
            ]
            
            if matching_rows.empty:
                continue
            
            # Update each matching row
            for idx in matching_rows.index:
                # Aggregate by category for OPEX
                opex_items = period_df[
                    period_df['category'].str.contains('Operating Expenses|OPEX', case=False, na=False)
                ]
                
                if not opex_items.empty:
                    # Group by line_item_name and sum amounts
                    for line_item_name, item_group in opex_items.groupby('line_item_name'):
                        amount = item_group['amount'].sum()
                        # Use absolute value (expenses are typically negative in accounting)
                        amount = abs(amount) if amount < 0 else amount
                        
                        # Map to column name
                        column_name = None
                        for key, col in line_item_mapping.items():
                            if key.lower() in line_item_name.lower():
                                column_name = col
                                break
                        
                        if column_name and column_name in monthly_data.columns:
                            # Only update if current value is 0 or NaN
                            current_val = monthly_data.loc[idx, column_name]
                            if pd.isna(current_val) or current_val == 0:
                                monthly_data.loc[idx, column_name] = amount
                
                # Aggregate revenue line items if available
                revenue_items = period_df[
                    period_df['category'].str.contains('Revenue', case=False, na=False)
                ]
                
                if not revenue_items.empty:
                    for line_item_name, item_group in revenue_items.groupby('line_item_name'):
                        amount = item_group['amount'].sum()
                        # Revenue should be positive
                        amount = abs(amount) if amount < 0 else amount
                        
                        column_name = None
                        for key, col in line_item_mapping.items():
                            if key.lower() in line_item_name.lower():
                                column_name = col
                                break
                        
                        if column_name and column_name in monthly_data.columns:
                            current_val = monthly_data.loc[idx, column_name]
                            if pd.isna(current_val) or current_val == 0:
                                monthly_data.loc[idx, column_name] = amount
        
        return monthly_data
    
    except Exception as e:
        # If merging fails, return original data
        import traceback
        st.warning(f"⚠️ Could not merge detailed line items: {str(e)}")
        return monthly_data


# =============================================================================
# TAB RENDERERS
# =============================================================================

def render_run_forecast_tab(db, scenario_id: str, user_id: str):
    """Render the forecast runner tab - FIXED: Save button now works properly."""
    section_header("Run Forecast", "Generate revenue and expense projections")
    
    assumptions = load_assumptions(db, scenario_id)
    
    if not assumptions:
        alert_box("No assumptions configured. Please complete setup first.", "warning")
        if st.button("Go to Setup", type="primary", key="goto_setup_btn"):
            st.session_state['navigate_to'] = 'setup'
            st.rerun()
        return
    
    # Settings preview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card("Forecast Period", f"{assumptions.get('forecast_duration_months', 60)} months")
    with col2:
        wacc = assumptions.get('wacc', 12)
        metric_card("WACC", f"{wacc}%" if wacc > 1 else f"{wacc*100:.1f}%")
    with col3:
        margin = assumptions.get('margin_consumable_pct') or assumptions.get('gross_margin_liner', 38)
        metric_card("Consumable Margin", f"{margin}%" if margin > 1 else f"{margin*100:.0f}%")
    with col4:
        margin_r = assumptions.get('margin_refurb_pct') or assumptions.get('gross_margin_refurb', 32)
        metric_card("Refurb Margin", f"{margin_r}%" if margin_r > 1 else f"{margin_r*100:.0f}%")
    
    # ==========================================================================
    # FORECAST METHOD TOGGLE (NEW - Clear selection between Pipeline and Trend)
    # ==========================================================================
    st.markdown("### 🔀 Forecast Method")
    
    # Get current method from assumptions
    current_method = assumptions.get('forecast_method', 'pipeline')
    
    col_method1, col_method2 = st.columns(2)
    
    with col_method1:
        use_pipeline = st.radio(
            "Select Forecast Method",
            options=['pipeline', 'trend'],
            format_func=lambda x: '📈 **Pipeline-Based** (Fleet + Prospects)' if x == 'pipeline' else '📊 **Trend-Based** (Historical Trends)',
            index=0 if current_method == 'pipeline' else 1,
            key='forecast_method_toggle',
            horizontal=True,
            help="""
**Pipeline-Based:** Revenue = Fleet consumables/refurb + Prospect pipeline. 
Best for installed-base business models with sales pipeline data.

**Trend-Based:** Revenue follows historical trend curves. 
Best when you have reliable historical financials and want trend-driven forecasts.
            """
        )
        
        # Save method if changed
        if use_pipeline != current_method:
            assumptions['forecast_method'] = use_pipeline
            assumptions['use_trend_forecast'] = (use_pipeline == 'trend')
            db.update_assumptions(scenario_id, user_id, assumptions)
            st.rerun()
    
    with col_method2:
        if use_pipeline == 'pipeline':
            st.info("""
**Pipeline-Based Forecast:**
- Revenue from fleet wear parts + refurb + prospects
- COGS from margin percentages
- OPEX from expense assumptions
            """)
        else:
            st.info("""
**Trend-Based Forecast:**
- Line items follow configured trends
- Aggregates calculated from line items
- Configure in **AI Assumptions → Configure Assumptions**
            """)
    
    st.markdown("---")
    
    # ==========================================================================
    # AI ASSUMPTIONS STATUS (NEW IN v9.0)
    # ==========================================================================
    ai_available, ai_assumptions = check_ai_assumptions_status(db, scenario_id)
    
    if ai_available and AI_ASSUMPTIONS_AVAILABLE:
        st.success("✅ **AI Assumptions Active** - Forecast will use AI-derived values where available")
    else:
        st.info("ℹ️ Using manual/default assumptions. Complete AI Assumptions step for data-driven forecasting.")
    
    st.markdown("---")
    
    # Run options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Forecast Options")
        
        # Trend-based forecast option
        assumptions_data = db.get_scenario_assumptions(scenario_id, user_id) if hasattr(db, 'get_scenario_assumptions') else {}
        
        # Check for forecast_configs (saved from Trend Forecast tab)
        forecast_configs = assumptions_data.get('forecast_configs', {})
        # Also check legacy trend_forecasts key for backward compatibility
        trend_forecasts = assumptions_data.get('trend_forecasts', {})
        
        # Has config if either forecast_configs or trend_forecasts exists
        has_trend_config = bool(forecast_configs) or bool(trend_forecasts)
        
        # Initialize session state if not exists
        if 'use_trend_forecast' not in st.session_state:
            st.session_state['use_trend_forecast'] = False
        
        # If checkbox was enabled but config is missing, disable it
        if st.session_state.get('use_trend_forecast', False) and not has_trend_config:
            st.session_state['use_trend_forecast'] = False
        
        use_trend_forecast = st.checkbox(
            "Use Trend-Based Forecast (from Historical Data)",
            value=st.session_state.get('use_trend_forecast', False),
            disabled=not has_trend_config,
            key="use_trend_forecast",
            help="Use trend analysis from historical financial data instead of pipeline-based growth. Configure in AI Assumptions → Trend Forecast tab."
        )
        
        if use_trend_forecast and has_trend_config:
            config_keys = list(forecast_configs.keys()) if forecast_configs else list(trend_forecasts.keys())
            st.success(f"✅ Trend forecast configured for: {', '.join(config_keys[:5])}{'...' if len(config_keys) > 5 else ''}")
        elif not has_trend_config:
            st.info("💡 Configure trend forecasts in **AI Assumptions → Trend Forecast** tab to enable this option.")
        
        st.markdown("---")
        
        include_monte_carlo = st.checkbox("Include Monte Carlo Simulation", value=True, key="fc_mc_checkbox")
        
        if include_monte_carlo:
            with st.expander("Monte Carlo Settings"):
                mc_iterations = st.number_input("Iterations", min_value=100, max_value=10000, value=1000, step=100, key="fc_mc_iter")
                mc_fleet_cv = st.slider("Fleet Revenue CV (%)", min_value=5, max_value=30, value=10, key="fc_mc_fleet") / 100
                mc_prospect_cv = st.slider("Prospect Revenue CV (%)", min_value=10, max_value=50, value=30, key="fc_mc_prospect") / 100
                mc_cost_cv = st.slider("Cost CV (%)", min_value=5, max_value=25, value=10, key="fc_mc_cost") / 100
        
        # NEW: Manufacturing Strategy Toggle
        vi_scenario = st.session_state.get('vi_scenario')
        has_vi_scenario = vi_scenario is not None
        
        # Only enable checkbox if strategy is not 'buy' (since buy is the default/current approach)
        # If strategy is 'buy', user shouldn't need to check this box
        default_checked = has_vi_scenario and vi_scenario.strategy != 'buy'
        
        include_manufacturing = st.checkbox(
            "Include Manufacturing Strategy (Optional)", 
            value=default_checked,
            key="fc_mfg_checkbox",
            help="Apply manufacturing strategy (Make vs Buy) to COGS calculation. You can run the forecast without this."
        )
        
        # Show strategy info regardless of checkbox state
        if has_vi_scenario:
            strategy = vi_scenario.strategy.lower() if hasattr(vi_scenario, 'strategy') else 'buy'
            
            if strategy == 'make':
                strategy_label = "MAKE"
                start_msg = f"Manufacturing Starts Month {vi_scenario.commissioning.completion_month}"
                if include_manufacturing:
                    st.success(f"✅ **{strategy_label} Strategy Active** - {start_msg}")
                else:
                    st.info(f"📦 **{strategy_label} Strategy Configured** - {start_msg} (Enable checkbox above to apply)")
            elif strategy == 'hybrid':
                strategy_label = f"HYBRID ({vi_scenario.hybrid_make_pct*100:.0f}% Make)"
                start_msg = f"Hybrid Starts Month {vi_scenario.commissioning.completion_month}"
                if include_manufacturing:
                    st.success(f"✅ **{strategy_label} Strategy Active** - {start_msg}")
                else:
                    st.info(f"📦 **{strategy_label} Strategy Configured** - {start_msg} (Enable checkbox above to apply)")
            else:  # buy
                strategy_label = "BUY"
                start_msg = "Buy Retained (Current Approach)"
                st.info(f"📦 **{strategy_label} Strategy** - {start_msg}")
        else:
            if include_manufacturing:
                st.info("ℹ️ **No manufacturing strategy configured.** The forecast will run with the default 'Buy' approach. You can configure manufacturing strategy later if needed.")
                include_manufacturing = False
            else:
                st.info("ℹ️ **Manufacturing is optional.** You can configure it in the Manufacturing Strategy section, or run the forecast with the default 'Buy' approach.")
    
    with col2:
        st.markdown("### What will be calculated")
        st.markdown("""
        - Revenue from existing machine fleet (wear-based)
        - Weighted pipeline revenue (confidence-adjusted)
        - COGS based on gross margins
        - OPEX from expense functions
        - Gross profit and EBIT projections
        """)
        
        if include_manufacturing:
            st.markdown("""
            **Plus Manufacturing Impact:**
            - Adjusted COGS (Buy vs Make portions)
            - Manufacturing overhead & depreciation
            - Commissioning costs by month
            """)
        
        if include_monte_carlo:
            st.markdown("""
            **Plus Monte Carlo:**
            - P10/P50/P90 confidence bands
            - Distribution histograms
            - Valuation range (P10-P50-P90)
            """)
    
    st.markdown("---")
    
    # =========================================================================
    # EXECUTION MODE (Local vs API Worker)
    # =========================================================================
    import os
    import urllib.request
    import urllib.error
    from urllib.parse import urljoin
    
    def _get_api_jwt() -> Optional[str]:
        """
        Best-effort JWT getter for API calls.
        - If you implement Supabase Auth later, stash the access token in session_state (any of these keys).
        - Also supports an env override for local dev: FORECAST_API_JWT
        """
        try:
            for k in ["access_token", "supabase_access_token", "jwt", "api_jwt"]:
                v = st.session_state.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        except Exception:
            pass
        env_jwt = os.getenv("FORECAST_API_JWT", "").strip()
        return env_jwt or None

    def _http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        jwt_token = _get_api_jwt()
        if jwt_token:
            headers["Authorization"] = f"Bearer {jwt_token}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    
    exec_mode = st.radio(
        "Execution mode",
        ["Local (in-app)", "API (background)"],
        horizontal=True,
        key="forecast_exec_mode",
        help="Use API mode to avoid UI hangs for long runs (requires Redis + API worker running).",
    )
    
    api_base_url_default = os.getenv("FORECAST_API_URL", "http://localhost:8000").rstrip("/") + "/"
    api_base_url = api_base_url_default
    if exec_mode == "API (background)":
        api_base_url = st.text_input(
            "API base URL",
            value=api_base_url_default.rstrip("/"),
            help="Example: http://localhost:8000",
            key="forecast_api_base_url",
        ).rstrip("/") + "/"
        st.caption("API job runner executes forecasts in the background (no in-app progress bar).")
        if include_monte_carlo:
            st.info("Monte Carlo will be run in the API job and saved into the snapshot.")
        if include_manufacturing:
            st.info("Manufacturing scenario will be passed to the API job (requires it to be configured).")

    # ---------------------------------------------------------------------
    # Import period settings summary (what months the model expects)
    # ---------------------------------------------------------------------
    try:
        if hasattr(db, "get_scenario_assumptions"):
            _assum = db.get_scenario_assumptions(scenario_id, user_id) or {}
        else:
            _assum = {}
        ip = (_assum or {}).get("import_period_settings") or {}
        if isinstance(ip, dict) and ip:
            with st.expander("📌 Last historics import settings (expected periods)", expanded=False):
                st.caption("These settings are saved during import and used to validate historics coverage before running forecasts.")
                for k, label in [
                    ("historical_income_statement_line_items", "Income Statement"),
                    ("historical_balance_sheet_line_items", "Balance Sheet"),
                    ("historical_cashflow_line_items", "Cash Flow"),
                ]:
                    blk = ip.get(k) or {}
                    sel = blk.get("selected_periods") or []
                    fy_end = blk.get("fiscal_year_end_month")
                    ytd = blk.get("ytd") or {}
                    st.markdown(f"**{label}**")
                    st.write(f"- Selected periods: **{len(sel) if isinstance(sel, list) else 0}**")
                    if isinstance(sel, list) and sel:
                        st.write(f"  - Range: `{sel[0]}` → `{sel[-1]}`")
                    if fy_end:
                        st.write(f"- Fiscal year-end month: **{int(fy_end)}**")
                    if isinstance(ytd, dict) and ytd:
                        try:
                            y_year = ytd.get("year")
                            y_end_m = ytd.get("ytd_end_calendar_month")
                            fill = ytd.get("fill_fy_months") or []
                            st.write(f"- YTD: **Yes** (FY{y_year}, ends at month {y_end_m}, fill FY months: {fill})")
                        except Exception:
                            st.write("- YTD: **Yes**")
                    else:
                        st.write("- YTD: **No**")
                    st.markdown("---")
    except Exception:
        pass
    
    # Run buttons
    if exec_mode == "Local (in-app)":
        if st.button("▶️ Run Full Forecast", type="primary", use_container_width=True, key="run_forecast_main_btn"):
            progress_bar = st.progress(0, text="Starting forecast...")
            status = st.empty()
            
            def update_progress(pct, msg):
                progress_bar.progress(pct, text=msg)
                status.caption(msg)
            
            # Get manufacturing scenario if enabled
            mfg_scenario = None
            if include_manufacturing:
                mfg_scenario = st.session_state.get('vi_scenario')

            # Gate: ensure expected historical imports actually exist in DB (prevents silent "no history" runs)
            try:
                assumptions_for_gate = {}
                if hasattr(db, "get_scenario_assumptions"):
                    assumptions_for_gate = db.get_scenario_assumptions(scenario_id, user_id) or {}
                ok_hist, hist_msgs = validate_historical_import_coverage(db, scenario_id, user_id=user_id, assumptions=assumptions_for_gate)
                if not ok_hist:
                    progress_bar.empty()
                    status.empty()
                    st.error("Cannot run forecast until historics import issues are resolved:")
                    for m in hist_msgs:
                        st.error(f"- {m}")
                    return
            except Exception:
                # Non-blocking on validator failure (but we still prefer it when it works)
                pass
            
            # Run forecast
            results = run_forecast(db, scenario_id, user_id, progress_callback=update_progress, 
                                  manufacturing_scenario=mfg_scenario)
            
            if not results['success']:
                st.error(f"Forecast failed: {results.get('error', 'Unknown error')}")
                progress_bar.empty()
                status.empty()
                return
            
            # Store manufacturing flag in results for snapshot comparison
            results['manufacturing_included'] = include_manufacturing
            results['manufacturing_strategy'] = mfg_scenario.strategy if mfg_scenario else None
            
            # Store in session state
            st.session_state['forecast_results'] = results
            
            # Run Monte Carlo if enabled
            if include_monte_carlo:
                status.caption("Running Monte Carlo simulation...")
                mc_config = {
                    'iterations': mc_iterations,
                    'fleet_cv': mc_fleet_cv,
                    'prospect_cv': mc_prospect_cv,
                    'cost_cv': mc_cost_cv,
                    'seed': 42
                }
                mc_results = run_monte_carlo(results, mc_config, progress_callback=update_progress)
                st.session_state['mc_results'] = mc_results
            
            progress_bar.empty()
            status.empty()
            
            # Force rerun to display results properly
            st.rerun()
    else:
        job_key = f"forecast_api_job_id_{scenario_id}"
        
        cols = st.columns([2, 1, 1])
        with cols[0]:
            if st.button("▶️ Run Forecast via API (Background)", type="primary", use_container_width=True, key="run_forecast_api_btn"):
                try:
                    # Gate before enqueuing (same checks as local mode)
                    try:
                        assumptions_for_gate = {}
                        if hasattr(db, "get_scenario_assumptions"):
                            assumptions_for_gate = db.get_scenario_assumptions(scenario_id, user_id) or {}
                        ok_hist, hist_msgs = validate_historical_import_coverage(db, scenario_id, user_id=user_id, assumptions=assumptions_for_gate)
                        if not ok_hist:
                            st.error("Cannot enqueue forecast until historics import issues are resolved:")
                            for m in hist_msgs:
                                st.error(f"- {m}")
                            return
                    except Exception:
                        pass

                    run_url = urljoin(api_base_url, "v1/forecasts/run")
                    payload = {
                        "scenario_id": scenario_id,
                        "user_id": user_id,
                        "options": {
                            "forecast_duration_months": int(st.session_state.get("forecast_periods", 60)),
                            "forecast_method": st.session_state.get("forecast_method", None),
                            "use_trend_forecast": bool(st.session_state.get("use_trend_forecast", False)),
                            "run_monte_carlo": bool(include_monte_carlo),
                            "mc_iterations": int(mc_iterations),
                            "mc_fleet_cv": float(mc_fleet_cv),
                            "mc_prospect_cv": float(mc_prospect_cv),
                            "mc_cost_cv": float(mc_cost_cv),
                            "mc_seed": 42,
                            "include_manufacturing": bool(include_manufacturing),
                        },
                    }
                    if include_manufacturing:
                        try:
                            vi_scenario = st.session_state.get("vi_scenario")
                            if vi_scenario and hasattr(vi_scenario, "to_dict"):
                                payload["options"]["manufacturing_strategy"] = vi_scenario.to_dict()
                        except Exception:
                            pass
                    resp = _http_json("POST", run_url, payload=payload, timeout=30)
                    job_id = resp.get("job_id")
                    if not job_id:
                        st.error(f"API did not return job_id. Response: {resp}")
                    else:
                        st.session_state[job_key] = job_id
                        st.success(f"Enqueued job: {job_id}")
                except urllib.error.HTTPError as e:
                    try:
                        body = e.read().decode("utf-8")
                    except Exception:
                        body = ""
                    st.error(f"API error ({e.code}): {body or e}")
                except Exception as e:
                    st.error(f"Failed to enqueue job: {e}")
        
        with cols[1]:
            st.button("Refresh status", use_container_width=True, key="refresh_forecast_api_job")
        
        with cols[2]:
            if st.button("Clear job", use_container_width=True, key="clear_forecast_api_job"):
                if job_key in st.session_state:
                    del st.session_state[job_key]
        
        job_id = st.session_state.get(job_key)
        if job_id:
            try:
                status_url = urljoin(api_base_url, f"v1/jobs/{job_id}")
                status_resp = _http_json("GET", status_url, payload=None, timeout=30)
                st.markdown("#### API job status")
                st.json(status_resp)
                
                if status_resp.get("status") == "finished":
                    job_result = status_resp.get("result") or {}
                    compact = job_result.get("result") if isinstance(job_result, dict) else None
                    if compact and isinstance(compact, dict) and compact.get("success"):
                        st.success("✅ API forecast finished successfully (preview payload).")
                        if isinstance(job_result, dict) and job_result.get("enterprise_value") is not None:
                            try:
                                st.metric("Enterprise Value (API)", format_currency(float(job_result.get("enterprise_value") or 0)))
                            except Exception:
                                pass
                        snapshot_id = job_result.get("snapshot_id") if isinstance(job_result, dict) else None
                        if snapshot_id:
                            st.caption(f"Snapshot saved: {snapshot_id}")
                            if st.button("Load snapshot into UI", use_container_width=True, key=f"load_api_snapshot_{snapshot_id}"):
                                try:
                                    # Prefer API snapshot endpoint (keeps UI thin).
                                    snap_api_url = urljoin(api_base_url, f"v1/snapshots/{snapshot_id}?user_id={user_id}")
                                    snap_row = _http_json("GET", snap_api_url, payload=None, timeout=30)

                                    forecast_data = snap_row.get("forecast_data") if isinstance(snap_row, dict) else None
                                    if forecast_data:
                                        st.session_state["forecast_results"] = forecast_data
                                        if snap_row.get("monte_carlo_data"):
                                            st.session_state["mc_results"] = snap_row.get("monte_carlo_data")
                                        if snap_row.get("valuation_data"):
                                            st.session_state["valuation_data"] = snap_row.get("valuation_data")
                                        if snap_row.get("enterprise_value") is not None:
                                            st.session_state["enterprise_value"] = snap_row.get("enterprise_value")
                                        st.success("Loaded snapshot into UI (via API).")
                                        st.rerun()
                                    else:
                                        # Fallback: load directly from DB (legacy behavior)
                                        snap = (
                                            db.client.table("forecast_snapshots")
                                            .select("forecast_data, monte_carlo_data, valuation_data, enterprise_value")
                                            .eq("id", snapshot_id)
                                            .limit(1)
                                            .execute()
                                        )
                                        if snap.data and snap.data[0].get("forecast_data"):
                                            st.session_state["forecast_results"] = snap.data[0]["forecast_data"]
                                            if snap.data[0].get("monte_carlo_data"):
                                                st.session_state["mc_results"] = snap.data[0]["monte_carlo_data"]
                                            if snap.data[0].get("valuation_data"):
                                                st.session_state["valuation_data"] = snap.data[0]["valuation_data"]
                                            if snap.data[0].get("enterprise_value") is not None:
                                                st.session_state["enterprise_value"] = snap.data[0]["enterprise_value"]
                                            st.success("Loaded snapshot into UI (via DB fallback).")
                                            st.rerun()
                                        else:
                                            st.error("Snapshot found but forecast_data was empty.")
                                except Exception as e:
                                    st.error(f"Failed to load snapshot: {e}")
                    elif compact and isinstance(compact, dict) and not compact.get("success", True):
                        st.error(f"API forecast failed: {compact.get('error')}")
                elif status_resp.get("status") == "failed":
                    st.error("❌ API job failed. See error in payload above.")
            except Exception as e:
                st.error(f"Failed to fetch job status: {e}")
    
    # =========================================================================
    # RESULTS & SAVE SECTION - OUTSIDE THE RUN BUTTON BLOCK (FIXED!)
    # This section renders whenever forecast_results exists in session state
    # =========================================================================
    results = st.session_state.get('forecast_results')
    
    if results and results.get('success'):
        st.success(f"✅ Forecast complete! ({results.get('summary', {}).get('machine_count', 0)} machines, {results.get('summary', {}).get('prospect_count', 0)} prospects)")
        
        # NEW in v9.0: Show assumption source
        assumptions_source = results.get('assumptions_source', 'Manual')
        ai_used = results.get('ai_assumptions_used', [])
        if ai_used:
            st.info(f"📊 Assumptions source: **{assumptions_source}** ({len(ai_used)} AI-derived values used)")
        
        # Quick summary
        summary = results.get('summary', {})
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            metric_card("Total Revenue", format_currency(summary.get('total_revenue', 0)))
        with col2:
            metric_card("Gross Profit", format_currency(summary.get('total_gross_profit', 0)))
        with col3:
            metric_card("Pipeline Revenue", format_currency(summary.get('total_pipeline', 0)))
        with col4:
            metric_card("GP Margin", f"{summary.get('avg_gross_margin', 0)*100:.1f}%")
        with col5:
            metric_card("EBIT", format_currency(summary.get('total_ebit', 0)))
        
        # Save snapshot section
        st.markdown("---")
        st.markdown("### 💾 Save This Forecast")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            snapshot_name = st.text_input(
                "Snapshot Name", 
                value=f"Forecast {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                key="snapshot_name_input_main"
            )
        with col2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            save_clicked = st.button("💾 Save Snapshot", use_container_width=True, key="save_snapshot_main_btn")
        
        if save_clicked:
            mc_results = st.session_state.get('mc_results')
            
            with st.spinner("Saving snapshot..."):
                success = save_snapshot(
                    db=db,
                    scenario_id=scenario_id,
                    user_id=user_id,
                    forecast_results=results,
                    mc_results=mc_results,
                    snapshot_name=snapshot_name
                )
            
            if success:
                st.success("✅ Snapshot saved successfully!")
                st.balloons()
            else:
                st.error("❌ Failed to save snapshot. Check error messages above.")


def render_results_tab(db, scenario_id: str, user_id: str):
    """Render the forecast results tab with professional financial statements."""
    section_header("Forecast Results", "View projections and financial statements")
    
    # DIAGNOSTIC: Show debugging info if available
    ytd_diag = st.session_state.get('ytd_diagnostic')
    hist_diag = st.session_state.get('historical_data_diagnostic')
    
    if ytd_diag or hist_diag:
        with st.expander("🔍 Data Diagnostic (Debug Info)", expanded=False):
            # Database Table Check
            st.markdown("#### 🗄️ Database Status")
            try:
                # Check if tables have data
                hf_result = db.client.table('historic_financials').select('id', count='exact').eq('scenario_id', scenario_id).limit(1).execute()
                hf_count = hf_result.count if hasattr(hf_result, 'count') else len(hf_result.data or [])
                
                he_result = db.client.table('historic_expense_categories').select('id', count='exact').eq('scenario_id', scenario_id).limit(1).execute()
                he_count = he_result.count if hasattr(he_result, 'count') else len(he_result.data or [])
                
                li_result = db.client.table('historical_income_statement_line_items').select('id', count='exact').eq('scenario_id', scenario_id).limit(1).execute()
                li_count = li_result.count if hasattr(li_result, 'count') else len(li_result.data or [])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if hf_count > 0:
                        st.success(f"✅ historic_financials: {hf_count} rows")
                    else:
                        st.error("❌ historic_financials: Empty")
                with col2:
                    if he_count > 0:
                        st.success(f"✅ expense_categories: {he_count} rows")
                    else:
                        st.warning("⚠️ expense_categories: Empty")
                with col3:
                    if li_count > 0:
                        st.success(f"✅ line_items: {li_count} rows")
                    else:
                        st.warning("⚠️ line_items: Empty")
                
                if hf_count == 0 and he_count == 0 and li_count == 0:
                    st.error("""
**⚠️ No historical data in database!**

Your data may only be in session memory and will be lost when you close the browser.

**To fix this:**
1. Go to **Setup → Import Data**
2. Re-import your historical financial data
3. The data will now persist to the database
                    """)
            except Exception as e:
                st.warning(f"Could not check database tables: {str(e)[:100]}")
            
            st.markdown("---")
            
            # Historical Data Info
            if hist_diag:
                st.markdown("#### Historical Data")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", hist_diag.get('total_rows', 0))
                with col2:
                    st.metric("Date Range", hist_diag.get('date_range', 'N/A'))
                with col3:
                    has_detail = hist_diag.get('has_expense_detail', False)
                    st.metric("Has OPEX Detail", "✅ Yes" if has_detail else "❌ No")
                
                years = hist_diag.get('years_present', [])
                if years:
                    st.info(f"Years in data: {', '.join(map(str, years))}")
                
                expense_cols = hist_diag.get('expense_columns', [])
                if expense_cols:
                    st.success(f"OPEX detail columns: {', '.join(expense_cols)}")
                else:
                    st.warning("⚠️ No OPEX detail columns found. Check if `historic_expense_categories` table has data.")
                
                source_cols = hist_diag.get('expense_source_columns', [])
                if source_cols:
                    st.caption(f"Expense source columns (raw): {', '.join(source_cols)}")

                # Upsample diagnostics
                if hist_diag.get('upsampled'):
                    strategy = hist_diag.get('upsample_strategy', '')
                    if 'sales_pattern' in strategy:
                        st.success(f"✅ Upsampled using **sales pattern** for revenue, pro-rata for OPEX")
                    else:
                        st.info(f"Upsampled using pro-rata monthly distribution")
                    
                    st.caption(f"Rows created: {hist_diag.get('upsample_rows_created', 0)}; Years: {', '.join(map(str, hist_diag.get('upsample_years', [])))}")
                    
                    # Show column details
                    rev_cols = hist_diag.get('revenue_cols_patterned', [])
                    opex_cols = hist_diag.get('opex_cols_prorata', [])
                    if rev_cols:
                        st.caption(f"Revenue cols (patterned): {', '.join(rev_cols[:5])}{' ...' if len(rev_cols) > 5 else ''}")
                    if opex_cols:
                        st.caption(f"OPEX cols (pro-rata): {', '.join(opex_cols[:5])}{' ...' if len(opex_cols) > 5 else ''}")
                else:
                    st.caption("Upsample status: Not applied (data already monthly)")
                
                st.markdown("---")
            
            # YTD Info
            if ytd_diag:
                st.markdown("#### YTD Actuals (Current Year)")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Year", ytd_diag.get('current_year', 'N/A'))
                with col2:
                    st.metric("Current Month", ytd_diag.get('current_month', 'N/A'))
                with col3:
                    st.metric("YTD Months Found", ytd_diag.get('ytd_months_found', 0))
                with col4:
                    ytd_rev = ytd_diag.get('ytd_revenue_total', 0)
                    st.metric("YTD Revenue Total", f"R {ytd_rev:,.0f}")
                
                ytd_periods = ytd_diag.get('ytd_periods', [])
                if ytd_periods:
                    st.success(f"✅ YTD periods loaded: {', '.join(ytd_periods)}")
                else:
                    st.warning("⚠️ No YTD periods found in historical data for current year")
                    st.info("""
**Troubleshooting:**
1. Check if `historic_financials` table has data for the current year
2. Verify the `month` or `period_date` column has correct dates
3. Ensure data is imported for all months up to current month
                    """)
                
                missing = ytd_diag.get('missing_months', [])
                if missing:
                    pretty_missing = ", ".join([datetime(2000, m, 1).strftime('%b') for m in missing])
                    st.error(f"Missing YTD months for current year: {pretty_missing}")

            st.markdown("---")

            # Canonical bucket diagnostics (monthly)
            try:
                st.markdown("#### 📋 Monthly Bucket Diagnostics (from imported history)")
                # Load the exact historical dataset used by forecasting/AI assumptions
                try:
                    from components.ai_assumptions_engine import load_historical_data
                    _hist = load_historical_data(db, scenario_id, user_id)
                except Exception:
                    _hist = load_historical_financials(db, scenario_id)

                if _hist is None or _hist.empty:
                    st.warning("No historical rows available to diagnose.")
                else:
                    _df = _hist.copy()
                    if 'period_date' in _df.columns:
                        _df['period_date'] = pd.to_datetime(_df['period_date'], errors='coerce')
                    if 'period_date' not in _df.columns or _df['period_date'].isna().all():
                        st.warning("Historical data is missing valid `period_date` values.")
                    else:
                        _df['month'] = _df['period_date'].dt.to_period('M').dt.to_timestamp()

                        def _col(name: str) -> float:
                            return 0.0

                        bucket_cols = [
                            ('total_revenue', 'Revenue'),
                            ('total_cogs', 'COGS'),
                            ('total_opex', 'OPEX'),
                            ('depreciation', 'Depreciation'),
                            ('interest_expense', 'Interest'),
                            ('tax_expense', 'Tax'),
                            ('other_income', 'Other income'),
                            ('net_income', 'Net income'),
                        ]

                        # Ensure missing columns exist so groupby works
                        for col, _label in bucket_cols:
                            if col not in _df.columns:
                                _df[col] = 0.0

                        diag_monthly = (
                            _df.groupby('month', dropna=True)[[c for c, _ in bucket_cols]]
                            .sum()
                            .sort_index()
                            .reset_index()
                        )
                        diag_monthly.rename(columns={'month': 'period_date', **{c: lbl for c, lbl in bucket_cols}}, inplace=True)

                        # Quick sanity checks
                        # Revenue missing months flag
                        zero_rev_months = diag_monthly[diag_monthly['Revenue'].fillna(0) == 0]['period_date']
                        if len(zero_rev_months) > 0:
                            st.warning(
                                "Revenue is zero for some months: "
                                + ", ".join(pd.to_datetime(zero_rev_months).dt.strftime('%Y-%m').head(12).tolist())
                                + (" ..." if len(zero_rev_months) > 12 else "")
                            )

                        st.dataframe(diag_monthly, use_container_width=True, hide_index=True)
                        st.caption("Tip: If Revenue/COGS/OPEX are missing or all-zero, the issue is bucket mapping or missing imports.")
            except Exception as e:
                st.warning(f"Could not render bucket diagnostics: {str(e)[:160]}")
    
    results = st.session_state.get('forecast_results')
    mc_results = st.session_state.get('mc_results')
    
    # Try loading from latest snapshot if no results
    if not results:
        snapshots = load_snapshots(db, scenario_id, limit=1)
        
        if snapshots:
            snapshot = snapshots[0]
            try:
                results = _snapshot_to_forecast_results(snapshot)
                alert_box(f"Showing saved snapshot: {snapshot.get('snapshot_name', 'Latest')}", "info")
            except:
                pass
    
    if not results:
        if empty_state(
            "No Forecast Results",
            "Run a forecast first to see results here",
            icon="📊",
            button_label="Run Forecast",
            button_key="run_forecast_from_results_btn"
        ):
            st.session_state['forecast_tab'] = 0
            st.rerun()
        return
    
    # Load historical data - use the same logic as AI Assumptions to get aggregated data
    try:
        from components.ai_assumptions_engine import load_historical_data
        historical_df = load_historical_data(db, scenario_id, user_id)
        
        # Ensure is_actual flag is set for historical data
        if not historical_df.empty:
            historical_df['is_actual'] = True
            historical_df['data_type'] = 'Actual'
    except Exception:
        # Fallback to old method
        historical_df = load_historical_financials(db, scenario_id)
        if not historical_df.empty:
            historical_df['is_actual'] = True
            historical_df['data_type'] = 'Actual'
    
    # Summary KPIs
    summary = results.get('summary', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Total Revenue", format_currency(summary.get('total_revenue', 0)))
    with col2:
        metric_card("COGS", format_currency(summary.get('total_cogs', 0)))
    with col3:
        metric_card("Gross Profit", format_currency(summary.get('total_gross_profit', 0)))
    with col4:
        metric_card("OPEX", format_currency(summary.get('total_opex', 0)))
    with col5:
        metric_card("EBIT", format_currency(summary.get('total_ebit', 0)))
    
    st.markdown("---")
    
    # Sub-tabs for different views - NEW: Added Commentary tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Charts", "📑 Financial Statements", "📊 Analysis", "📥 Export", "📝 Commentary"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = create_revenue_chart(results, mc_results, historical_df)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = create_profitability_chart(results)
            st.plotly_chart(fig, use_container_width=True)
        
        fig = create_margin_chart(results)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Show historical data info
        if not historical_df.empty:
            st.info(f"📊 Showing {len(historical_df)} months of historical actuals + forecast data")
        
        # Sub-tabs for different financial statements
        fs_tab1, fs_tab2, fs_tab3 = st.tabs(["📑 Income Statement", "📊 Balance Sheet", "💵 Cash Flow"])
        
        with fs_tab1:
            # Financial statement controls
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                view_mode = st.radio(
                    "📅 Period View",
                    options=["monthly", "annual"],
                    format_func=format_period_view_label,
                    horizontal=True,
                    key="fs_view_mode_results"
                )
            
            with col2:
                show_annual_subtotals = st.checkbox(
                    "📊 Annual Subtotals",
                    value=view_mode == "monthly",
                    disabled=view_mode == "annual",
                    key="fs_annual_subs_results"
                )
            
            with col3:
                show_revenue_detail = st.checkbox(
                    "📈 Revenue Detail",
                    value=True,
                    key="fs_rev_detail_results"
                )
            
            with col4:
                show_opex_detail = st.checkbox(
                    "💼 Expense Detail",
                    value=True,
                    key="fs_opex_detail_results"
                )
            
            st.markdown("---")
            
            # Build and render financial data
            assumptions = results.get('assumptions') or load_assumptions(db, scenario_id) or {}
            monthly_data = build_monthly_financials(results, assumptions, historical_df)
            
            # NEW: Load detailed line items for historical periods and merge into display data
            try:
                from components.ai_assumptions_engine import load_detailed_line_items
                detailed_is_items = load_detailed_line_items(db, scenario_id, 'income_statement', user_id)
                
                if not detailed_is_items.empty and not monthly_data.empty:
                    # Merge detailed line items into monthly_data for historical periods
                    monthly_data = _merge_detailed_line_items_into_monthly_data(
                        monthly_data, detailed_is_items, historical_df
                    )
            except Exception as e:
                # If detailed line items can't be loaded, continue with aggregated data
                pass
            
            if not monthly_data.empty:
                if view_mode == "annual":
                    display_data = aggregate_to_annual(monthly_data)
                else:
                    display_data = monthly_data
                
                # Legend
                st.markdown(f"""
                <div style="
                    display: flex; 
                    gap: 2rem; 
                    margin-bottom: 1rem; 
                    font-size: 0.8rem;
                    padding: 0.5rem 1rem;
                    background: {DARK_BG};
                    border-radius: 4px;
                ">
                    <span><span style="color: {GOLD};">■</span> Subtotal</span>
                    <span><span style="color: {TEXT_WHITE}; font-weight: bold;">■</span> Total</span>
                    <span><span style="color: {RED};">(123)</span> Expense</span>
                    <span><span style="color: {TEXT_MUTED};">■</span> Detail</span>
                    <span><span style="background: rgba(100, 116, 139, 0.3); padding: 2px 6px;">■</span> Actual</span>
                </div>
                """, unsafe_allow_html=True)
                
                render_income_statement_table(
                    data=display_data,
                    show_revenue_detail=show_revenue_detail,
                    show_opex_detail=show_opex_detail,
                    view_mode=view_mode
                )
            else:
                st.warning("Could not build financial statements from forecast data.")
        
        with fs_tab2:
            # Balance Sheet
            st.markdown("### Balance Sheet")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                bs_view_mode = st.radio(
                    "Period View",
                    options=["annual", "monthly"],
                    format_func=format_annual_view_label,
                    horizontal=True,
                    key="bs_view_mode"
                )
            
            assumptions = results.get('assumptions') or load_assumptions(db, scenario_id) or {}
            
            # Get manufacturing strategy if included in forecast
            mfg_strategy_data = None
            if results.get('manufacturing_included'):
                vi_scenario = st.session_state.get('vi_scenario')
                if vi_scenario:
                    mfg_strategy_data = {
                        'equipment_cost': vi_scenario.capex.equipment_cost,
                        'facility_cost': vi_scenario.capex.facility_cost,
                        'tooling_cost': vi_scenario.capex.tooling_cost,
                        'working_capital': vi_scenario.capex.working_capital,
                        'commissioning_start_month': vi_scenario.commissioning.start_month,
                        'commissioning_completion_month': vi_scenario.commissioning.completion_month,
                        'commissioning_monthly_costs': vi_scenario.commissioning.monthly_costs,
                    }
            
            balance_sheet_data = build_balance_sheet(
                results, assumptions, historical_df, 
                manufacturing_strategy=mfg_strategy_data
            )
            
            if not balance_sheet_data.empty:
                render_balance_sheet_table(balance_sheet_data, view_mode=bs_view_mode)
            else:
                st.info("Balance sheet projections will be generated from forecast data.")
        
        with fs_tab3:
            # Cash Flow Statement
            st.markdown("### Cash Flow Statement")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                cf_view_mode = st.radio(
                    "Period View",
                    options=["annual", "monthly"],
                    format_func=format_annual_view_label,
                    horizontal=True,
                    key="cf_view_mode"
                )
            
            assumptions = results.get('assumptions') or load_assumptions(db, scenario_id) or {}
            
            # Get manufacturing strategy if included in forecast
            mfg_strategy_data = None
            if results.get('manufacturing_included'):
                vi_scenario = st.session_state.get('vi_scenario')
                if vi_scenario:
                    mfg_strategy_data = {
                        'equipment_cost': vi_scenario.capex.equipment_cost,
                        'facility_cost': vi_scenario.capex.facility_cost,
                        'tooling_cost': vi_scenario.capex.tooling_cost,
                        'working_capital': vi_scenario.capex.working_capital,
                        'commissioning_start_month': vi_scenario.commissioning.start_month,
                        'commissioning_completion_month': vi_scenario.commissioning.completion_month,
                        'commissioning_monthly_costs': vi_scenario.commissioning.monthly_costs,
                    }
            
            cash_flow_data = build_cash_flow(
                results, assumptions, 
                balance_sheet_data=None,
                manufacturing_strategy=mfg_strategy_data,
                historical_data=historical_df
            )
            
            if not cash_flow_data.empty:
                render_cash_flow_table(cash_flow_data, view_mode=cf_view_mode)
            else:
                st.info("Cash flow projections will be generated from forecast data.")
    
    with tab3:
        st.markdown("### Revenue Composition")
        col1, col2, col3 = st.columns(3)
        total_rev = summary.get('total_revenue', 1)
        with col1:
            pct = summary.get('total_consumables', 0) / max(total_rev, 1) * 100
            metric_card("Consumables", format_currency(summary.get('total_consumables', 0)), f"{pct:.1f}%")
        with col2:
            pct = summary.get('total_refurb', 0) / max(total_rev, 1) * 100
            metric_card("Refurbishment", format_currency(summary.get('total_refurb', 0)), f"{pct:.1f}%")
        with col3:
            pct = summary.get('total_pipeline', 0) / max(total_rev, 1) * 100
            metric_card("Pipeline", format_currency(summary.get('total_pipeline', 0)), f"{pct:.1f}%")
        
        st.markdown("### Key Metrics")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Machine Count | {summary.get('machine_count', 0)} |
        | Prospect Count | {summary.get('prospect_count', 0)} |
        | Data Source | {summary.get('data_source', 'unknown')} |
        | GP Margin Used | {summary.get('margin_consumable_used', 0)*100:.1f}% |
        | Forecast Months | {summary.get('forecast_months', 60)} |
        """)
    
    with tab4:
        st.markdown("### Export Options")
        
        # Build export data
        assumptions = results.get('assumptions') or load_assumptions(db, scenario_id) or {}
        export_df = build_monthly_financials(results, assumptions, historical_df)
        
        # Get manufacturing strategy if included in forecast (FIX: Define in this tab's scope)
        mfg_strategy_data = None
        if results.get('manufacturing_included'):
            vi_scenario = st.session_state.get('vi_scenario')
            if vi_scenario:
                mfg_strategy_data = {
                    'equipment_cost': vi_scenario.capex.equipment_cost,
                    'facility_cost': vi_scenario.capex.facility_cost,
                    'tooling_cost': vi_scenario.capex.tooling_cost,
                    'working_capital': vi_scenario.capex.working_capital,
                    'commissioning_start_month': vi_scenario.commissioning.start_month,
                    'commissioning_completion_month': getattr(vi_scenario.commissioning, 'completion_month', None) or (
                        vi_scenario.commissioning.start_month + vi_scenario.commissioning.duration_months - 1
                        if hasattr(vi_scenario.commissioning, 'duration_months') else None
                    ),
                    'commissioning_monthly_costs': vi_scenario.commissioning.monthly_costs,
                }
        
        if not export_df.empty:
            # CSV Downloads
            st.markdown("#### CSV Export")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Income Statement (CSV)",
                    data=csv,
                    file_name=f"income_statement_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_income_stmt_btn"
                )
            
            with col2:
                summary_df = pd.DataFrame([summary])
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary (CSV)",
                    data=csv_summary,
                    file_name=f"forecast_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_summary_btn"
                )

            st.markdown("---")
            
            # NEW: PDF Export
            st.markdown("#### PDF Export")
            try:
                from components.pdf_exporter import render_pdf_export_ui
                # Build balance sheet and cash flow data for PDF
                balance_sheet_data = build_balance_sheet(
                    results, assumptions, historical_df, 
                    manufacturing_strategy=mfg_strategy_data
                )
                cash_flow_data = build_cash_flow(
                    results, assumptions, balance_sheet_data, 
                    manufacturing_strategy=mfg_strategy_data,
                    historical_data=historical_df
                )
                scenario_name = assumptions.get('scenario_name', 'Scenario')
                render_pdf_export_ui(export_df, balance_sheet_data, cash_flow_data, scenario_name)
            except ImportError:
                st.info("PDF export requires reportlab. Install with: pip install reportlab")
            except (NameError, TypeError) as e:
                st.error(f"Error building financial statements: {e}")
                st.info("Please ensure forecast includes manufacturing strategy data if applicable.")
            
            st.markdown("---")
            
            # NEW: Excel Export
            st.markdown("#### Excel Export")
            try:
                from components.excel_exporter import render_excel_export_ui
                balance_sheet_data = build_balance_sheet(
                    results, assumptions, historical_df,
                    manufacturing_strategy=mfg_strategy_data
                )
                cash_flow_data = build_cash_flow(
                    results, assumptions, balance_sheet_data,
                    manufacturing_strategy=mfg_strategy_data,
                    historical_data=historical_df
                )
                scenario_name = assumptions.get('scenario_name', 'Scenario')
                render_excel_export_ui(export_df, balance_sheet_data, cash_flow_data, scenario_name)
            except ImportError:
                st.info("Excel export requires openpyxl. Install with: pip install openpyxl")
            except (NameError, TypeError) as e:
                st.error(f"Error building financial statements: {e}")
                st.info("Please ensure forecast includes manufacturing strategy data if applicable.")
    
    with tab5:
        # NEW: Financial Commentary
        st.markdown("### 📝 Financial Commentary Generator")
        
        # Get manufacturing strategy if included in forecast (FIX: Define in this tab's scope)
        mfg_strategy_data = None
        if results.get('manufacturing_included'):
            vi_scenario = st.session_state.get('vi_scenario')
            if vi_scenario:
                mfg_strategy_data = {
                    'equipment_cost': vi_scenario.capex.equipment_cost,
                    'facility_cost': vi_scenario.capex.facility_cost,
                    'tooling_cost': vi_scenario.capex.tooling_cost,
                    'working_capital': vi_scenario.capex.working_capital,
                    'commissioning_start_month': vi_scenario.commissioning.start_month,
                    'commissioning_completion_month': getattr(vi_scenario.commissioning, 'completion_month', None) or (
                        vi_scenario.commissioning.start_month + vi_scenario.commissioning.duration_months - 1
                        if hasattr(vi_scenario.commissioning, 'duration_months') else None
                    ),
                    'commissioning_monthly_costs': vi_scenario.commissioning.monthly_costs,
                }
        
        try:
            from components.commentary_generator import render_commentary_ui
            
            # Build data for commentary
            balance_sheet_data = build_balance_sheet(
                results, assumptions, historical_df,
                manufacturing_strategy=mfg_strategy_data
            )
            cash_flow_data = build_cash_flow(
                results, assumptions, balance_sheet_data,
                manufacturing_strategy=mfg_strategy_data,
                historical_data=historical_df
            )
            
            scenario_name = assumptions.get('scenario_name', 'Scenario')
            render_commentary_ui(
                export_df, balance_sheet_data, cash_flow_data,
                results, assumptions, scenario_name
            )
        except ImportError:
            st.info("Commentary generator is available. Ensure `components/commentary_generator.py` exists.")
        except (NameError, TypeError) as e:
            st.error(f"Error building financial statements: {e}")
            st.info("Please ensure forecast includes manufacturing strategy data if applicable.")



# =============================================================================
# MONTE CARLO TAB - ENHANCED WITH STANDALONE RUN (v8.4)
# =============================================================================

def get_distribution_for_element(ai_assumptions, element_name: str):
    """
    Get distribution parameters for a forecast element.
    
    Maps forecast element names to AI assumption keys.
    """
    if not ai_assumptions or not hasattr(ai_assumptions, 'assumptions'):
        return None
    
    # Map forecast element names to AI assumption keys
    element_to_assumption_map = {
        'revenue': 'total_revenue',
        'total_revenue': 'total_revenue',
        'cogs': 'total_cogs',
        'total_cogs': 'total_cogs',
        'opex': 'total_opex',
        'total_opex': 'total_opex',
        'gross_profit': 'total_gross_profit',
        'total_gross_profit': 'total_gross_profit',
    }
    
    assumption_key = element_to_assumption_map.get(element_name, element_name)
    
    if assumption_key in ai_assumptions.assumptions:
        assumption = ai_assumptions.assumptions[assumption_key]
        if assumption.use_distribution:
            return assumption.user_distribution
    
    return None


def get_historical_distribution_params(distribution_params):
    """
    Extract historical mean and standard deviation from distribution parameters.
    
    Returns: (historical_mean, historical_std) tuple
    """
    if not distribution_params or distribution_params.distribution_type == "static":
        return None, None
    
    try:
        if distribution_params.distribution_type == "normal":
            return distribution_params.mean, distribution_params.std
        elif distribution_params.distribution_type == "lognormal":
            # For lognormal: mean = exp(μ + σ²/2), std = sqrt((exp(σ²) - 1) * exp(2μ + σ²))
            mean = np.exp(distribution_params.mean + distribution_params.std**2 / 2)
            std = np.sqrt((np.exp(distribution_params.std**2) - 1) * np.exp(2 * distribution_params.mean + distribution_params.std**2))
            return mean, std
        elif distribution_params.distribution_type == "triangular":
            mean = (distribution_params.min_val + distribution_params.mode_val + distribution_params.max_val) / 3
            # Approximate std for triangular
            std = np.sqrt((distribution_params.min_val**2 + distribution_params.mode_val**2 + distribution_params.max_val**2 - 
                          distribution_params.min_val * distribution_params.mode_val - 
                          distribution_params.min_val * distribution_params.max_val - 
                          distribution_params.mode_val * distribution_params.max_val) / 18)
            return mean, std
        elif distribution_params.distribution_type == "beta":
            mean = distribution_params.loc + distribution_params.scale * (
                distribution_params.alpha / (distribution_params.alpha + distribution_params.beta)
            )
            variance = distribution_params.scale**2 * (
                distribution_params.alpha * distribution_params.beta / 
                ((distribution_params.alpha + distribution_params.beta)**2 * (distribution_params.alpha + distribution_params.beta + 1))
            )
            std = np.sqrt(variance)
            return mean, std
        elif distribution_params.distribution_type == "uniform":
            mean = (distribution_params.low + distribution_params.high) / 2
            std = (distribution_params.high - distribution_params.low) / np.sqrt(12)
            return mean, std
        else:
            if hasattr(distribution_params, 'static_value'):
                return distribution_params.static_value, 0.0
            return None, None
    except Exception:
        return None, None


def sample_from_period_specific_distribution(forecast_value: float, historical_mean: float, 
                                            historical_std: float, distribution_type: str, 
                                            distribution_params, seed_offset: int = 0):
    """
    Sample from a distribution centered on the forecast value with proportionally scaled standard deviation.
    
    Principle: Use forecast value as new mean, scale stdev proportionally.
    Example: If historical mean=100m, stdev=15m, and forecast=200m, then:
    - New mean = 200m
    - New stdev = 15m * (200m / 100m) = 30m
    
    Args:
        forecast_value: The forecast value for this period (becomes the new mean)
        historical_mean: Historical mean from distribution parameters
        historical_std: Historical standard deviation from distribution parameters
        distribution_type: Type of distribution (normal, lognormal, etc.)
        distribution_params: Original distribution parameters
        seed_offset: Offset for random seed (for reproducibility)
    
    Returns:
        Sampled value from period-specific distribution
    """
    if historical_mean is None or historical_std is None or historical_mean == 0:
        return forecast_value
    
    # Calculate scaling factor
    scale_factor = forecast_value / historical_mean if historical_mean != 0 else 1.0
    
    # Scale the standard deviation proportionally
    scaled_std = historical_std * scale_factor
    
    # Ensure non-negative standard deviation
    if scaled_std < 0:
        scaled_std = abs(scaled_std)
    
    try:
        # Set random seed with offset for reproducibility
        np.random.seed(42 + seed_offset)
        
        if distribution_type == "normal":
            # Sample from normal distribution with forecast as mean and scaled std
            return np.random.normal(forecast_value, scaled_std)
        elif distribution_type == "lognormal":
            # For lognormal, we need to convert back to log-space parameters
            # If we want mean=forecast_value and std=scaled_std in normal space:
            # μ = ln(mean² / sqrt(std² + mean²))
            # σ = sqrt(ln(1 + std²/mean²))
            if forecast_value > 0 and scaled_std > 0:
                log_mean = np.log(forecast_value**2 / np.sqrt(scaled_std**2 + forecast_value**2))
                log_std = np.sqrt(np.log(1 + (scaled_std**2 / forecast_value**2)))
                return np.random.lognormal(log_mean, log_std)
            else:
                return forecast_value
        elif distribution_type == "triangular":
            # For triangular, scale min/mode/max proportionally
            if hasattr(distribution_params, 'min_val') and hasattr(distribution_params, 'max_val'):
                scaled_min = distribution_params.min_val * scale_factor
                scaled_max = distribution_params.max_val * scale_factor
                scaled_mode = forecast_value  # Mode is the forecast value
                # Ensure min <= mode <= max
                scaled_min = min(scaled_min, forecast_value)
                scaled_max = max(scaled_max, forecast_value)
                return np.random.triangular(scaled_min, scaled_mode, scaled_max)
            else:
                return forecast_value
        elif distribution_type == "beta":
            # For beta, maintain shape parameters but scale location and scale
            if hasattr(distribution_params, 'loc') and hasattr(distribution_params, 'scale'):
                # Scale the location and scale proportionally
                scaled_loc = distribution_params.loc * scale_factor
                scaled_scale = distribution_params.scale * scale_factor
                sample = np.random.beta(distribution_params.alpha, distribution_params.beta)
                return scaled_loc + scaled_scale * sample
            else:
                return forecast_value
        elif distribution_type == "uniform":
            # For uniform, scale low and high proportionally
            if hasattr(distribution_params, 'low') and hasattr(distribution_params, 'high'):
                scaled_low = distribution_params.low * scale_factor
                scaled_high = distribution_params.high * scale_factor
                return np.random.uniform(scaled_low, scaled_high)
            else:
                return forecast_value
        else:
            # Static or unknown: return forecast value
            return forecast_value
    except Exception:
        # Fallback: return forecast value if sampling fails
        return forecast_value


def run_monte_carlo_enhanced(base_results: Dict, config: Dict, 
                             ai_assumptions: Optional[Any] = None,
                             forecast_configs: Optional[Dict] = None,
                             historical_data: Optional[pd.DataFrame] = None,
                             progress_callback=None) -> Dict[str, Any]:
    """
    Enhanced Monte Carlo simulation with separate COGS/OPEX CV.
    
    NEW IN v8.4:
    - Separate COGS and OPEX coefficient of variation
    - More detailed tracking for sensitivity analysis
    
    NEW IN v9.0:
    - ai_assumptions parameter for using AI-fitted distributions
    - use_ai_distributions config option
    - Distribution source tracking
    
    NEW IN v9.1:
    - forecast_configs parameter for trend-based forecasts
    - historical_data parameter for trend generation
    - Applies distributions to trend parameters when using trend-based forecasts
    """
    mc_results = {
        'success': False,
        'iterations': config.get('iterations', 1000),
        'config': config,
        'percentiles': {
            'revenue': {},
            'gross_profit': {},
            'ebit': {}
        },
        'distributions': {},
        'sensitivity': {},
        'assumptions_source': 'CV-based'  # NEW in v9.0
    }
    
    try:
        n_iterations = config.get('iterations', 1000)
        fleet_cv = config.get('fleet_cv', 0.10)
        prospect_cv = config.get('prospect_cv', 0.30)
        cogs_cv = config.get('cogs_cv', config.get('cost_cv', 0.10))
        opex_cv = config.get('opex_cv', config.get('cost_cv', 0.10))
        seed = config.get('seed', 42)
        use_ai_distributions = config.get('use_ai_distributions', False)
        
        np.random.seed(seed)
        
        base_consumables = np.array(base_results['revenue']['consumables'])
        base_refurb = np.array(base_results['revenue']['refurb'])
        base_pipeline = np.array(base_results['revenue']['pipeline'])
        base_cogs = np.array(base_results['costs']['cogs'])
        base_opex = np.array(base_results['costs']['opex'])
        
        n_months = len(base_consumables)
        
        all_revenue = np.zeros((n_iterations, n_months))
        all_gp = np.zeros((n_iterations, n_months))
        all_ebit = np.zeros((n_iterations, n_months))
        
        sensitivity_data = {
            'fleet': [],
            'prospect': [],
            'cogs': [],
            'opex': []
        }
        
        # NEW in v9.0: Check if we should use AI distributions
        ai_samples = None
        if use_ai_distributions and AI_ASSUMPTIONS_AVAILABLE and ai_assumptions:
            try:
                if hasattr(ai_assumptions, 'assumptions_saved') and ai_assumptions.assumptions_saved:
                    ai_samples = sample_from_assumptions(ai_assumptions, n_iterations)
                    if ai_samples:
                        mc_results['assumptions_source'] = 'AI Distributions'
            except Exception:
                ai_samples = None
        
        # NEW in v9.1: Check if we have trend-based forecasts with distributions
        use_trend_distributions = (
            forecast_configs is not None 
            and ai_assumptions is not None
            and hasattr(ai_assumptions, 'assumptions_saved') 
            and ai_assumptions.assumptions_saved
        )
        
        # Get distributions for trend-based elements (only for non-calculated elements)
        trend_distributions = {}
        historical_params = {}  # Store historical mean/std for each element
        
        if use_trend_distributions:
            for element_name, element_config in forecast_configs.items():
                # Skip calculated elements - they're derived from other simulated values
                from components.forecast_correlation_engine import FORECAST_ELEMENTS
                element_def = FORECAST_ELEMENTS.get(element_name, {})
                if element_def.get('is_calculated', False):
                    continue
                
                if element_config.get('method') == 'trend_fit':
                    dist = get_distribution_for_element(ai_assumptions, element_name)
                    if dist:
                        trend_distributions[element_name] = dist
                        # Extract historical mean and std for scaling
                        hist_mean, hist_std = get_historical_distribution_params(dist)
                        historical_params[element_name] = {
                            'mean': hist_mean,
                            'std': hist_std,
                            'distribution_type': dist.distribution_type,
                            'distribution_params': dist
                        }
                        mc_results['assumptions_source'] = 'Period-Specific Distributions (Forecast-Centered)'
        
        for i in range(n_iterations):
            if progress_callback and i % 100 == 0:
                progress_callback(0.1 + 0.8 * (i / n_iterations), f"Monte Carlo iteration {i+1:,}/{n_iterations:,}")
            
            # NEW in v9.2: Period-specific distributions centered on forecast values
            if use_trend_distributions and trend_distributions:
                # Sample period-by-period using forecast value as mean
                revenue_sim = np.zeros(n_months)
                cogs_sim = np.zeros(n_months)
                opex_sim = np.zeros(n_months)
                
                # Revenue (consumables + refurb) - use total revenue forecast
                if 'revenue' in historical_params or 'total_revenue' in historical_params:
                    params = historical_params.get('revenue') or historical_params.get('total_revenue')
                    dist_type = params['distribution_type']
                    dist_params = params['distribution_params']
                    hist_mean = params['mean']
                    hist_std = params['std']
                    
                    # Get total revenue forecast (consumables + refurb)
                    base_total_revenue = base_consumables + base_refurb
                    
                    for month_idx in range(n_months):
                        forecast_val = base_total_revenue[month_idx]
                        seed_offset = i * n_months + month_idx
                        sampled_val = sample_from_period_specific_distribution(
                            forecast_val, hist_mean, hist_std, dist_type, dist_params, seed_offset
                        )
                        revenue_sim[month_idx] = sampled_val
                else:
                    # Fallback: use CV-based approach
                    fleet_factor = np.random.lognormal(0, fleet_cv, n_months)
                    revenue_sim = (base_consumables + base_refurb) * fleet_factor
                
                # COGS
                if 'cogs' in historical_params or 'total_cogs' in historical_params:
                    params = historical_params.get('cogs') or historical_params.get('total_cogs')
                    dist_type = params['distribution_type']
                    dist_params = params['distribution_params']
                    hist_mean = params['mean']
                    hist_std = params['std']
                    
                    for month_idx in range(n_months):
                        forecast_val = base_cogs[month_idx]
                        seed_offset = i * n_months + month_idx
                        sampled_val = sample_from_period_specific_distribution(
                            forecast_val, hist_mean, hist_std, dist_type, dist_params, seed_offset
                        )
                        cogs_sim[month_idx] = sampled_val
                else:
                    # Fallback: use CV-based approach
                    cogs_factor = np.random.lognormal(0, cogs_cv, n_months)
                    cogs_sim = base_cogs * cogs_factor
                
                # OPEX
                if 'opex' in historical_params or 'total_opex' in historical_params:
                    params = historical_params.get('opex') or historical_params.get('total_opex')
                    dist_type = params['distribution_type']
                    dist_params = params['distribution_params']
                    hist_mean = params['mean']
                    hist_std = params['std']
                    
                    for month_idx in range(n_months):
                        forecast_val = base_opex[month_idx]
                        seed_offset = i * n_months + month_idx
                        sampled_val = sample_from_period_specific_distribution(
                            forecast_val, hist_mean, hist_std, dist_type, dist_params, seed_offset
                        )
                        opex_sim[month_idx] = sampled_val
                else:
                    # Fallback: use CV-based approach
                    opex_factor = np.random.lognormal(0, opex_cv, n_months)
                    opex_sim = base_opex * opex_factor
                
                # Pipeline revenue (prospect) - still uses CV (not trend-based)
                prospect_factor = np.random.lognormal(0, prospect_cv, n_months)
                pipeline_sim = base_pipeline * prospect_factor
                
                # Total revenue = fleet revenue + pipeline
                revenue_sim = revenue_sim + pipeline_sim
                
                # Store fleet revenue separately for sensitivity analysis
                fleet_revenue_sim = revenue_sim - pipeline_sim
                
            else:
                # Fallback: Use AI samples or CV-based approach (legacy)
                if ai_samples and 'revenue_growth_pct' in ai_samples:
                    growth_rate = ai_samples['revenue_growth_pct'][i] if i < len(ai_samples['revenue_growth_pct']) else 0
                    growth_factor = 1 + (growth_rate / 100 - 0.05)
                    fleet_factor = np.ones(n_months) * max(0.5, min(1.5, growth_factor))
                else:
                    fleet_factor = np.random.lognormal(0, fleet_cv, n_months)
                
                if ai_samples and 'gross_margin_pct' in ai_samples:
                    margin_value = ai_samples['gross_margin_pct'][i] if i < len(ai_samples['gross_margin_pct']) else 38
                    margin_factor = margin_value / 38
                    cogs_factor = np.ones(n_months) / max(0.5, min(1.5, margin_factor))
                else:
                    cogs_factor = np.random.lognormal(0, cogs_cv, n_months)
                
                prospect_factor = np.random.lognormal(0, prospect_cv, n_months)
                opex_factor = np.random.lognormal(0, opex_cv, n_months)
                
                consumables_sim = base_consumables * fleet_factor
                refurb_sim = base_refurb * fleet_factor
                pipeline_sim = base_pipeline * prospect_factor
                revenue_sim = consumables_sim + refurb_sim + pipeline_sim
                cogs_sim = base_cogs * cogs_factor
                opex_sim = base_opex * opex_factor
                
                # For sensitivity analysis (legacy path)
                fleet_revenue_sim = consumables_sim + refurb_sim
            
            # Calculated elements (gross_profit, EBIT) are derived from simulated values
            # No MC parameters applied - they're calculated from revenue, COGS, OPEX
            gp_sim = revenue_sim - cogs_sim
            ebit_sim = gp_sim - opex_sim
            
            all_revenue[i] = revenue_sim
            all_gp[i] = gp_sim
            all_ebit[i] = ebit_sim
            
            # Sensitivity data collection
            if use_trend_distributions and trend_distributions:
                # Period-specific distributions: fleet revenue stored separately
                sensitivity_data['fleet'].append(np.sum(fleet_revenue_sim))
                sensitivity_data['prospect'].append(np.sum(pipeline_sim))
            else:
                # Legacy: consumables_sim and refurb_sim are separate
                sensitivity_data['fleet'].append(np.sum(consumables_sim + refurb_sim))
                sensitivity_data['prospect'].append(np.sum(pipeline_sim))
            
            sensitivity_data['cogs'].append(np.sum(cogs_sim))
            sensitivity_data['opex'].append(np.sum(opex_sim))
        
        if progress_callback:
            progress_callback(0.95, "Calculating statistics...")
        
        mc_results['percentiles']['revenue'] = {
            'p5': np.percentile(all_revenue, 5, axis=0).tolist(),
            'p10': np.percentile(all_revenue, 10, axis=0).tolist(),
            'p25': np.percentile(all_revenue, 25, axis=0).tolist(),
            'p50': np.percentile(all_revenue, 50, axis=0).tolist(),
            'p75': np.percentile(all_revenue, 75, axis=0).tolist(),
            'p90': np.percentile(all_revenue, 90, axis=0).tolist(),
            'p95': np.percentile(all_revenue, 95, axis=0).tolist(),
            'mean': np.mean(all_revenue, axis=0).tolist(),
            'std': np.std(all_revenue, axis=0).tolist()
        }
        
        mc_results['percentiles']['gross_profit'] = {
            'p5': np.percentile(all_gp, 5, axis=0).tolist(),
            'p10': np.percentile(all_gp, 10, axis=0).tolist(),
            'p25': np.percentile(all_gp, 25, axis=0).tolist(),
            'p50': np.percentile(all_gp, 50, axis=0).tolist(),
            'p75': np.percentile(all_gp, 75, axis=0).tolist(),
            'p90': np.percentile(all_gp, 90, axis=0).tolist(),
            'p95': np.percentile(all_gp, 95, axis=0).tolist()
        }
        
        mc_results['percentiles']['ebit'] = {
            'p5': np.percentile(all_ebit, 5, axis=0).tolist(),
            'p10': np.percentile(all_ebit, 10, axis=0).tolist(),
            'p25': np.percentile(all_ebit, 25, axis=0).tolist(),
            'p50': np.percentile(all_ebit, 50, axis=0).tolist(),
            'p75': np.percentile(all_ebit, 75, axis=0).tolist(),
            'p90': np.percentile(all_ebit, 90, axis=0).tolist(),
            'p95': np.percentile(all_ebit, 95, axis=0).tolist()
        }
        
        mc_results['distributions'] = {
            'total_revenue': np.sum(all_revenue, axis=1).tolist(),
            'total_gp': np.sum(all_gp, axis=1).tolist(),
            'total_ebit': np.sum(all_ebit, axis=1).tolist()
        }
        
        mc_results['sensitivity'] = {
            'fleet_revenue': {
                'mean': float(np.mean(sensitivity_data['fleet'])),
                'std': float(np.std(sensitivity_data['fleet'])),
                'p10': float(np.percentile(sensitivity_data['fleet'], 10)),
                'p90': float(np.percentile(sensitivity_data['fleet'], 90)),
                'label': 'Fleet Revenue'
            },
            'prospect_revenue': {
                'mean': float(np.mean(sensitivity_data['prospect'])),
                'std': float(np.std(sensitivity_data['prospect'])),
                'p10': float(np.percentile(sensitivity_data['prospect'], 10)),
                'p90': float(np.percentile(sensitivity_data['prospect'], 90)),
                'label': 'Pipeline Revenue'
            },
            'cogs': {
                'mean': float(np.mean(sensitivity_data['cogs'])),
                'std': float(np.std(sensitivity_data['cogs'])),
                'p10': float(np.percentile(sensitivity_data['cogs'], 10)),
                'p90': float(np.percentile(sensitivity_data['cogs'], 90)),
                'label': 'Cost of Goods Sold'
            },
            'opex': {
                'mean': float(np.mean(sensitivity_data['opex'])),
                'std': float(np.std(sensitivity_data['opex'])),
                'p10': float(np.percentile(sensitivity_data['opex'], 10)),
                'p90': float(np.percentile(sensitivity_data['opex'], 90)),
                'label': 'Operating Expenses'
            }
        }
        
        mc_results['success'] = True
        
        if progress_callback:
            progress_callback(1.0, "Monte Carlo complete!")
        
    except Exception as e:
        mc_results['error'] = str(e)
        mc_results['success'] = False
    
    return mc_results

def render_mc_distributions(mc_results: Dict):
    """Render distribution histograms for Monte Carlo results."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Total Revenue Distribution")
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Histogram(
            x=mc_results['distributions']['total_revenue'],
            nbinsx=50,
            marker_color=CHART_COLORS['primary'],
            name='Revenue'
        ))
        
        mean_rev = np.mean(mc_results['distributions']['total_revenue'])
        fig_rev.add_vline(x=mean_rev, line_dash="dash", line_color=GOLD, 
                         annotation_text=f"Mean: {format_currency(mean_rev)}")
        
        fig_rev.update_layout(
            xaxis_title='Total Revenue (R)',
            yaxis_title='Frequency',
            height=300,
            showlegend=False
        )
        fig_rev.update_xaxes(tickformat=',.0f', tickprefix='R ')
        st.plotly_chart(fig_rev, use_container_width=True)
    
    with col2:
        st.markdown("#### Total EBIT Distribution")
        fig_ebit = go.Figure()
        fig_ebit.add_trace(go.Histogram(
            x=mc_results['distributions']['total_ebit'],
            nbinsx=50,
            marker_color=CHART_COLORS['secondary'],
            name='EBIT'
        ))
        
        mean_ebit = np.mean(mc_results['distributions']['total_ebit'])
        fig_ebit.add_vline(x=mean_ebit, line_dash="dash", line_color=GOLD,
                          annotation_text=f"Mean: {format_currency(mean_ebit)}")
        
        fig_ebit.update_layout(
            xaxis_title='Total EBIT (R)',
            yaxis_title='Frequency',
            height=300,
            showlegend=False
        )
        fig_ebit.update_xaxes(tickformat=',.0f', tickprefix='R ')
        st.plotly_chart(fig_ebit, use_container_width=True)
    
    st.markdown("#### Total Gross Profit Distribution")
    fig_gp = go.Figure()
    fig_gp.add_trace(go.Histogram(
        x=mc_results['distributions']['total_gp'],
        nbinsx=50,
        marker_color=GREEN,
        name='Gross Profit'
    ))
    
    mean_gp = np.mean(mc_results['distributions']['total_gp'])
    fig_gp.add_vline(x=mean_gp, line_dash="dash", line_color=GOLD,
                     annotation_text=f"Mean: {format_currency(mean_gp)}")
    
    fig_gp.update_layout(
        xaxis_title='Total Gross Profit (R)',
        yaxis_title='Frequency',
        height=280,
        showlegend=False
    )
    fig_gp.update_xaxes(tickformat=',.0f', tickprefix='R ')
    st.plotly_chart(fig_gp, use_container_width=True)


def render_mc_confidence_bands(mc_results: Dict, forecast_results: Dict):
    """Render confidence bands over time."""
    
    if not forecast_results:
        st.warning("Base forecast results required for confidence bands.")
        return
    
    timeline = forecast_results.get('timeline', [])
    if not timeline:
        st.warning("No timeline data available.")
        return
    
    dates = pd.to_datetime(timeline)
    
    metric_choice = st.selectbox(
        "Select Metric",
        ["Revenue", "Gross Profit", "EBIT"],
        key="mc_bands_metric"
    )
    
    metric_map = {
        "Revenue": ('revenue', CHART_COLORS['primary']),
        "Gross Profit": ('gross_profit', GREEN),
        "EBIT": ('ebit', GOLD)
    }
    
    metric_key, color = metric_map[metric_choice]
    percentiles = mc_results['percentiles'].get(metric_key, {})
    
    if not percentiles:
        st.warning(f"No percentile data for {metric_choice}")
        return
    
    fig = go.Figure()
    
    if 'p5' in percentiles and 'p95' in percentiles:
        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates)[::-1],
            y=percentiles['p95'] + percentiles['p5'][::-1],
            fill='toself',
            fillcolor='rgba(212, 165, 55, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            name='P5-P95',
            showlegend=True
        ))
    
    if 'p10' in percentiles and 'p90' in percentiles:
        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates)[::-1],
            y=percentiles['p90'] + percentiles['p10'][::-1],
            fill='toself',
            fillcolor='rgba(212, 165, 55, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='P10-P90',
            showlegend=True
        ))
    
    if 'p25' in percentiles and 'p75' in percentiles:
        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates)[::-1],
            y=percentiles['p75'] + percentiles['p25'][::-1],
            fill='toself',
            fillcolor='rgba(212, 165, 55, 0.35)',
            line=dict(color='rgba(0,0,0,0)'),
            name='P25-P75',
            showlegend=True
        ))
    
    if 'p50' in percentiles:
        fig.add_trace(go.Scatter(
            x=dates,
            y=percentiles['p50'],
            mode='lines',
            line=dict(color=GOLD, width=2),
            name='Median (P50)'
        ))
    
    if 'mean' in percentiles:
        fig.add_trace(go.Scatter(
            x=dates,
            y=percentiles['mean'],
            mode='lines',
            line=dict(color=color, width=2, dash='dash'),
            name='Mean'
        ))
    
    fig.update_layout(
        title=f"{metric_choice} - Confidence Bands Over Time",
        xaxis_title='Month',
        yaxis_title=f'{metric_choice} (R)',
        height=450,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    fig.update_yaxes(tickformat=',.0f', tickprefix='R ')
    
    st.plotly_chart(fig, use_container_width=True)


def render_mc_tornado_chart(mc_results: Dict):
    """Render tornado chart for sensitivity analysis."""
    
    sensitivity = mc_results.get('sensitivity', {})
    
    if not sensitivity:
        st.info("Sensitivity data not available. Run simulation again to generate.")
        return
    
    st.markdown("#### Impact on Total EBIT")
    st.caption("Shows how each variable's uncertainty affects the outcome (P10 to P90 range)")
    
    impacts = []
    for key, data in sensitivity.items():
        if isinstance(data, dict) and 'p10' in data and 'p90' in data:
            range_val = data['p90'] - data['p10']
            impacts.append({
                'variable': data.get('label', key),
                'p10': data['p10'],
                'p90': data['p90'],
                'range': range_val,
                'mean': data['mean']
            })
    
    if not impacts:
        st.info("No sensitivity data to display.")
        return
    
    impacts.sort(key=get_impact_range_key, reverse=True)
    
    fig = go.Figure()
    
    variables = [i['variable'] for i in impacts]
    p10_values = [i['p10'] for i in impacts]
    p90_values = [i['p90'] for i in impacts]
    mean_values = [i['mean'] for i in impacts]
    
    fig.add_trace(go.Bar(
        y=variables,
        x=[-abs(p10 - mean) for p10, mean in zip(p10_values, mean_values)],
        orientation='h',
        name='Downside (P10)',
        marker_color=RED,
        text=[format_currency(v) for v in p10_values],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        y=variables,
        x=[abs(p90 - mean) for p90, mean in zip(p90_values, mean_values)],
        orientation='h',
        name='Upside (P90)',
        marker_color=GREEN,
        text=[format_currency(v) for v in p90_values],
        textposition='auto'
    ))
    
    fig.update_layout(
        barmode='relative',
        height=350,
        xaxis_title='Impact on Value (R)',
        yaxis_title='',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    fig.update_xaxes(tickformat=',.0f')
    
    st.plotly_chart(fig, use_container_width=True)


def render_mc_statistics(mc_results: Dict):
    """Render detailed statistics summary."""
    
    total_rev_dist = np.array(mc_results['distributions']['total_revenue'])
    total_gp_dist = np.array(mc_results['distributions']['total_gp'])
    total_ebit_dist = np.array(mc_results['distributions']['total_ebit'])
    
    st.markdown("#### Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Revenue**")
        st.markdown(f"- **Mean**: {format_currency(np.mean(total_rev_dist))}")
        st.markdown(f"- **Std Dev**: {format_currency(np.std(total_rev_dist))}")
        st.markdown(f"- **P5 (Downside)**: {format_currency(np.percentile(total_rev_dist, 5))}")
        st.markdown(f"- **P10**: {format_currency(np.percentile(total_rev_dist, 10))}")
        st.markdown(f"- **P25**: {format_currency(np.percentile(total_rev_dist, 25))}")
        st.markdown(f"- **P50 (Median)**: {format_currency(np.percentile(total_rev_dist, 50))}")
        st.markdown(f"- **P75**: {format_currency(np.percentile(total_rev_dist, 75))}")
        st.markdown(f"- **P90**: {format_currency(np.percentile(total_rev_dist, 90))}")
        st.markdown(f"- **P95 (Upside)**: {format_currency(np.percentile(total_rev_dist, 95))}")
    
    with col2:
        st.markdown("**📊 Gross Profit**")
        st.markdown(f"- **Mean**: {format_currency(np.mean(total_gp_dist))}")
        st.markdown(f"- **Std Dev**: {format_currency(np.std(total_gp_dist))}")
        st.markdown(f"- **P5 (Downside)**: {format_currency(np.percentile(total_gp_dist, 5))}")
        st.markdown(f"- **P10**: {format_currency(np.percentile(total_gp_dist, 10))}")
        st.markdown(f"- **P25**: {format_currency(np.percentile(total_gp_dist, 25))}")
        st.markdown(f"- **P50 (Median)**: {format_currency(np.percentile(total_gp_dist, 50))}")
        st.markdown(f"- **P75**: {format_currency(np.percentile(total_gp_dist, 75))}")
        st.markdown(f"- **P90**: {format_currency(np.percentile(total_gp_dist, 90))}")
        st.markdown(f"- **P95 (Upside)**: {format_currency(np.percentile(total_gp_dist, 95))}")
    
    with col3:
        st.markdown("**📊 EBIT**")
        st.markdown(f"- **Mean**: {format_currency(np.mean(total_ebit_dist))}")
        st.markdown(f"- **Std Dev**: {format_currency(np.std(total_ebit_dist))}")
        st.markdown(f"- **P5 (Downside)**: {format_currency(np.percentile(total_ebit_dist, 5))}")
        st.markdown(f"- **P10**: {format_currency(np.percentile(total_ebit_dist, 10))}")
        st.markdown(f"- **P25**: {format_currency(np.percentile(total_ebit_dist, 25))}")
        st.markdown(f"- **P50 (Median)**: {format_currency(np.percentile(total_ebit_dist, 50))}")
        st.markdown(f"- **P75**: {format_currency(np.percentile(total_ebit_dist, 75))}")
        st.markdown(f"- **P90**: {format_currency(np.percentile(total_ebit_dist, 90))}")
        st.markdown(f"- **P95 (Upside)**: {format_currency(np.percentile(total_ebit_dist, 95))}")
    
    st.markdown("---")
    st.markdown("#### Simulation Configuration")
    config = mc_results.get('config', {})
    
    config_cols = st.columns(6)
    with config_cols[0]:
        iter_val = config.get('iterations', 'N/A')
        st.metric("Iterations", f"{iter_val:,}" if isinstance(iter_val, (int, float)) else str(iter_val))
    with config_cols[1]:
        fleet_cv = config.get('fleet_cv', 0)
        st.metric("Fleet CV", f"{fleet_cv*100:.0f}%" if isinstance(fleet_cv, (int, float)) else "N/A")
    with config_cols[2]:
        prospect_cv = config.get('prospect_cv', 0)
        st.metric("Pipeline CV", f"{prospect_cv*100:.0f}%" if isinstance(prospect_cv, (int, float)) else "N/A")
    with config_cols[3]:
        cogs_cv = config.get('cogs_cv', config.get('cost_cv', 0))
        st.metric("COGS CV", f"{cogs_cv*100:.0f}%" if isinstance(cogs_cv, (int, float)) else "N/A")
    with config_cols[4]:
        opex_cv = config.get('opex_cv', config.get('cost_cv', 0))
        st.metric("OPEX CV", f"{opex_cv*100:.0f}%" if isinstance(opex_cv, (int, float)) else "N/A")
    with config_cols[5]:
        st.metric("Seed", config.get('seed', 'N/A'))
    
    st.markdown("---")
    if st.button("📥 Export Results to CSV", key="mc_export_btn"):
        export_df = pd.DataFrame({
            'Metric': ['Revenue', 'Revenue', 'Revenue', 'Gross Profit', 'Gross Profit', 'Gross Profit', 'EBIT', 'EBIT', 'EBIT'],
            'Statistic': ['P10', 'P50', 'P90'] * 3,
            'Value': [
                np.percentile(total_rev_dist, 10),
                np.percentile(total_rev_dist, 50),
                np.percentile(total_rev_dist, 90),
                np.percentile(total_gp_dist, 10),
                np.percentile(total_gp_dist, 50),
                np.percentile(total_gp_dist, 90),
                np.percentile(total_ebit_dist, 10),
                np.percentile(total_ebit_dist, 50),
                np.percentile(total_ebit_dist, 90),
            ]
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="monte_carlo_results.csv",
            mime="text/csv",
            key="mc_download_btn"
        )


def render_monte_carlo_tab(db, scenario_id: str, user_id: str):
    """
    Render the Monte Carlo simulation tab with standalone run capability.
    
    NEW IN v8.4:
    - Full settings panel always visible
    - Run Monte Carlo button (requires forecast_results)
    - Enhanced visualizations (distributions, bands, tornado, stats)
    - Separate COGS/OPEX CV settings
    - Seed control for reproducibility
    """
    section_header("Monte Carlo Simulation", "Risk analysis and confidence intervals")
    
    forecast_results = st.session_state.get('forecast_results')
    mc_results = st.session_state.get('mc_results')
    
    # ==========================================================================
    # SETTINGS PANEL
    # ==========================================================================
    st.markdown("### Simulation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mc_iterations = st.number_input(
            "Iterations",
            min_value=100,
            max_value=10000,
            value=st.session_state.get('mc_iterations', 1000),
            step=100,
            key="mc_tab_iterations",
            help="Number of simulation runs. More iterations = more accurate but slower."
        )
        
        mc_fleet_cv = st.slider(
            "Fleet Revenue CV (%)",
            min_value=5,
            max_value=30,
            value=st.session_state.get('mc_fleet_cv', 10),
            key="mc_tab_fleet_cv",
            help="Coefficient of variation for existing fleet revenue variability"
        ) / 100
        
        mc_prospect_cv = st.slider(
            "Prospect/Pipeline CV (%)",
            min_value=10,
            max_value=50,
            value=st.session_state.get('mc_prospect_cv', 30),
            key="mc_tab_prospect_cv",
            help="CV for pipeline revenue - higher because prospects are less certain"
        ) / 100
    
    with col2:
        mc_cogs_cv = st.slider(
            "COGS CV (%)",
            min_value=3,
            max_value=20,
            value=st.session_state.get('mc_cogs_cv', 8),
            key="mc_tab_cogs_cv",
            help="Coefficient of variation for cost of goods sold"
        ) / 100
        
        mc_opex_cv = st.slider(
            "OPEX CV (%)",
            min_value=3,
            max_value=25,
            value=st.session_state.get('mc_opex_cv', 12),
            key="mc_tab_opex_cv",
            help="Coefficient of variation for operating expenses"
        ) / 100
        
        mc_seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=99999,
            value=st.session_state.get('mc_seed', 42),
            key="mc_tab_seed",
            help="Set seed for reproducible results. Same seed = same results."
        )
    
    # ==========================================================================
    # AI DISTRIBUTIONS (NEW IN v9.0)
    # ==========================================================================
    ai_available, ai_assumptions = check_ai_assumptions_status(db, scenario_id)
    
    st.markdown("---")
    st.markdown("### AI Distribution Settings")
    
    if ai_available and AI_ASSUMPTIONS_AVAILABLE:
        st.success("✅ AI Assumptions available for this scenario")
        use_ai_distributions = st.checkbox(
            "Use AI-fitted distributions",
            value=st.session_state.get('mc_use_ai_dist', True),
            key="mc_tab_use_ai_dist",
            help="When enabled, uses probability distributions fitted by the AI Assumptions Engine instead of CV-based approach"
        )
    else:
        st.info("ℹ️ AI Assumptions not saved. Using coefficient of variation (CV) based distributions.")
        use_ai_distributions = False
        ai_assumptions = None
    
    st.markdown("---")
    
    # ==========================================================================
    # RUN BUTTON
    # ==========================================================================
    if not forecast_results or not forecast_results.get('success'):
        alert_box(
            "⚠️ Run a forecast first before running Monte Carlo simulation. "
            "Go to the 'Run Forecast' tab to generate base forecast data.",
            "warning"
        )
        run_disabled = True
    else:
        run_disabled = False
        st.info(f"📊 Base forecast loaded: {forecast_results.get('summary', {}).get('total_months', len(forecast_results.get('timeline', [])))} months of projections ready for simulation.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_mc_clicked = st.button(
            "🎲 Run Monte Carlo Simulation",
            type="primary",
            use_container_width=True,
            disabled=run_disabled,
            key="mc_tab_run_btn"
        )
    
    if run_mc_clicked and forecast_results:
        progress_bar = st.progress(0, text="Initializing Monte Carlo simulation...")
        status_text = st.empty()
        
        def update_progress(pct, msg):
            progress_bar.progress(min(pct, 1.0), text=msg)
            status_text.caption(msg)
        
        mc_config = {
            'iterations': mc_iterations,
            'fleet_cv': mc_fleet_cv,
            'prospect_cv': mc_prospect_cv,
            'cogs_cv': mc_cogs_cv,
            'opex_cv': mc_opex_cv,
            'cost_cv': (mc_cogs_cv + mc_opex_cv) / 2,
            'seed': mc_seed,
            'use_ai_distributions': use_ai_distributions  # NEW in v9.0
        }
        
        st.session_state['mc_iterations'] = mc_iterations
        st.session_state['mc_fleet_cv'] = int(mc_fleet_cv * 100)
        st.session_state['mc_prospect_cv'] = int(mc_prospect_cv * 100)
        st.session_state['mc_cogs_cv'] = int(mc_cogs_cv * 100)
        st.session_state['mc_opex_cv'] = int(mc_opex_cv * 100)
        st.session_state['mc_seed'] = mc_seed
        st.session_state['mc_use_ai_dist'] = use_ai_distributions  # NEW in v9.0
        
        update_progress(0.1, "Running Monte Carlo simulation...")
        # Get forecast_configs and historical_data for trend-based distributions
        assumptions_data = forecast_results.get('assumptions') or load_assumptions(db, scenario_id) or {}
        forecast_configs = assumptions_data.get('forecast_configs', {})
        
        # Load historical data if needed for trend distributions
        historical_data = None
        if forecast_configs:
            try:
                from components.ai_assumptions_engine import load_historical_data
                historical_data = load_historical_data(db, scenario_id, user_id)
            except Exception:
                historical_data = None
        
        mc_results = run_monte_carlo_enhanced(
            forecast_results, 
            mc_config,
            ai_assumptions=ai_assumptions,  # NEW in v9.0
            progress_callback=update_progress
        )
        
        st.session_state['mc_results'] = mc_results
        
        progress_bar.empty()
        status_text.empty()
        
        if mc_results.get('success'):
            st.success(f"✅ Monte Carlo simulation complete! {mc_iterations:,} iterations processed.")
            st.balloons()
        else:
            st.error(f"❌ Simulation failed: {mc_results.get('error', 'Unknown error')}")
        
        st.rerun()
    
    # ==========================================================================
    # RESULTS DISPLAY
    # ==========================================================================
    mc_results = st.session_state.get('mc_results')
    
    if not mc_results or not mc_results.get('success'):
        empty_state(
            "No Simulation Results Yet",
            "Configure settings above and click 'Run Monte Carlo Simulation' to generate results.",
            icon="🎲"
        )
        return
    
    st.markdown("---")
    st.markdown("### Simulation Results")
    iter_count = mc_results.get('iterations', 0)
    dist_source = mc_results.get('assumptions_source', 'CV-based')  # NEW in v9.0
    st.caption(f"Based on {iter_count:,} iterations • Distribution source: **{dist_source}**" if isinstance(iter_count, (int, float)) else "Simulation results")
    
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "📊 Distribution", 
        "📈 Confidence Bands", 
        "🌪️ Sensitivity",
        "📋 Statistics"
    ])
    
    with viz_tab1:
        render_mc_distributions(mc_results)
    
    with viz_tab2:
        render_mc_confidence_bands(mc_results, forecast_results)
    
    with viz_tab3:
        render_mc_tornado_chart(mc_results)
    
    with viz_tab4:
        render_mc_statistics(mc_results)


def render_valuation_tab(db, scenario_id: str, user_id: str):
    """Render DCF valuation tab (NEW IN 8.1)."""
    section_header("DCF Valuation", "Enterprise and equity value analysis")
    
    results = st.session_state.get('forecast_results')
    
    if not results or not results.get('success'):
        empty_state(
            "No Forecast Results",
            "Run a forecast first to perform valuation",
            icon="💰"
        )
        return
    
    assumptions = results.get('assumptions') or load_assumptions(db, scenario_id) or {}
    
    # Build forecast DataFrame for valuation
    df = pd.DataFrame({
        'revenue_total': results.get('revenue', {}).get('total', []),
        'gross_profit': results.get('profit', {}).get('gross', []),
        'opex': results.get('costs', {}).get('opex', [])
    })
    df['ebitda'] = df['gross_profit'] - df['opex']
    
    # Valuation inputs
    st.markdown("### Valuation Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        wacc_default = assumptions.get('wacc', 12)
        if wacc_default <= 1:
            wacc_default = wacc_default * 100
        wacc = st.number_input("WACC %", min_value=5.0, max_value=25.0, value=float(wacc_default), step=0.5, key="val_wacc") / 100
    
    with col2:
        growth_default = assumptions.get('terminal_growth_rate', 3)
        if growth_default <= 1:
            growth_default = growth_default * 100
        growth = st.number_input("Terminal Growth %", min_value=0.0, max_value=10.0, value=float(growth_default), step=0.5, key="val_g") / 100
    
    with col3:
        exit_mult = st.number_input("Exit Multiple (EBITDA)", min_value=2.0, max_value=15.0, value=float(assumptions.get('exit_multiple_ebitda', 5.0)), step=0.5, key="val_em")
    
    with col4:
        net_debt = st.number_input("Net Debt (R)", min_value=-100000000.0, max_value=100000000.0, value=0.0, step=1000000.0, key="val_nd")
    
    # Run valuation
    val_assumptions = assumptions.copy()
    val_assumptions['wacc'] = wacc
    val_assumptions['terminal_growth_rate'] = growth
    val_assumptions['exit_multiple_ebitda'] = exit_mult
    
    engine = ValuationEngine(val_assumptions)
    result = engine.calculate_valuation(df, net_debt)
    
    st.markdown("---")
    st.markdown("### Valuation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Enterprise Value", format_currency(result.enterprise_value))
    with col2:
        st.metric("Equity Value", format_currency(result.equity_value))
    with col3:
        st.metric("Terminal Value", format_currency(result.terminal_value_perpetuity), f"{result.tv_as_pct_ev:.0f}% of EV")
    with col4:
        st.metric("Implied EV/EBITDA", f"{result.implied_ev_ebitda:.1f}x")
    
    # ==========================================================================
    # NEW: Monte Carlo Valuation Range
    # ==========================================================================
    mc_results = st.session_state.get('mc_results')
    
    if mc_results and mc_results.get('success'):
        st.markdown("---")
        st.markdown("### 📊 Monte Carlo Valuation Range")
        st.caption("Enterprise value distribution based on Monte Carlo simulation")
        
        # Calculate valuation for each Monte Carlo iteration
        ev_distribution = []
        mc_distributions = mc_results.get('distributions', {})
        
        if 'gross_profit' in mc_distributions or 'revenue' in mc_distributions:
            n_iterations = len(mc_distributions.get('gross_profit', mc_distributions.get('revenue', [[]])))
            
            # Run valuation for each iteration
            for i in range(min(n_iterations, 1000)):  # Cap at 1000 iterations for performance
                try:
                    # Get iteration data
                    if 'gross_profit' in mc_distributions:
                        iter_gp = mc_distributions['gross_profit'][i]
                    else:
                        # Estimate from revenue
                        iter_rev = mc_distributions['revenue'][i]
                        avg_margin = results.get('summary', {}).get('avg_gross_margin', 0.38)
                        iter_gp = [r * avg_margin for r in iter_rev]
                    
                    iter_opex = results.get('costs', {}).get('opex', [])
                    
                    # Build iteration DataFrame
                    iter_df = pd.DataFrame({
                        'revenue_total': mc_distributions.get('revenue', [results.get('revenue', {}).get('total', [])])[i] if 'revenue' in mc_distributions else results.get('revenue', {}).get('total', []),
                        'gross_profit': iter_gp,
                        'opex': iter_opex
                    })
                    iter_df['ebitda'] = iter_df['gross_profit'] - iter_df['opex']
                    
                    # Calculate valuation for this iteration
                    iter_result = engine.calculate_valuation(iter_df, net_debt)
                    ev_distribution.append(iter_result.enterprise_value)
                except Exception:
                    continue
            
            if ev_distribution:
                ev_array = np.array(ev_distribution)
                ev_p10 = np.percentile(ev_array, 10)
                ev_p50 = np.percentile(ev_array, 50)
                ev_p90 = np.percentile(ev_array, 90)
                
                # Display percentiles
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("P10 (Conservative)", format_currency(ev_p10))
                with col2:
                    st.metric("P50 (Base Case)", format_currency(ev_p50))
                with col3:
                    st.metric("P90 (Optimistic)", format_currency(ev_p90))
                with col4:
                    range_pct = ((ev_p90 - ev_p10) / ev_p50) * 100 if ev_p50 > 0 else 0
                    st.metric("Value Range", f"±{range_pct/2:.0f}%")
                
                # Distribution chart
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=ev_array / 1e6,
                    nbinsx=50,
                    name='EV Distribution',
                    marker_color='#3b82f6',
                    opacity=0.7
                ))
                
                # Add percentile lines
                fig_dist.add_vline(x=ev_p10/1e6, line_dash="dash", line_color="#ef4444",
                                   annotation_text=f"P10: R{ev_p10/1e6:.1f}M")
                fig_dist.add_vline(x=ev_p50/1e6, line_dash="solid", line_color="#22c55e",
                                   annotation_text=f"P50: R{ev_p50/1e6:.1f}M")
                fig_dist.add_vline(x=ev_p90/1e6, line_dash="dash", line_color="#3b82f6",
                                   annotation_text=f"P90: R{ev_p90/1e6:.1f}M")
                
                fig_dist.update_layout(
                    title="Enterprise Value Distribution (Monte Carlo)",
                    xaxis_title="Enterprise Value (R millions)",
                    yaxis_title="Frequency",
                    showlegend=False,
                    height=350
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("💡 Run forecast with Monte Carlo simulation enabled to see valuation range")
    
    # Sensitivity analysis
    st.markdown("---")
    with st.expander("📈 Sensitivity Analysis", expanded=True):
        st.markdown("**Enterprise Value by WACC and Terminal Growth**")
        
        sens_df = engine.run_sensitivity(df, net_debt)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=sens_df.values,
            x=sens_df.columns.tolist(),
            y=sens_df.index.tolist(),
            colorscale='RdYlGn',
            text=[[f"R{v/1e6:.1f}M" for v in row] for row in sens_df.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="WACC: %{y}<br>Growth: %{x}<br>EV: R%{z:,.0f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Enterprise Value Sensitivity",
            xaxis_title="Terminal Growth Rate",
            yaxis_title="WACC",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Value bridge
    with st.expander("📊 Value Bridge"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            | Component | Value |
            |-----------|-------|
            | PV of Discrete Cash Flows | {format_currency(result.pv_discrete_cf)} |
            | PV of Terminal Value | {format_currency(result.pv_terminal_value)} |
            | **Enterprise Value** | **{format_currency(result.enterprise_value)}** |
            | Less: Net Debt | {format_currency(-net_debt)} |
            | **Equity Value** | **{format_currency(result.equity_value)}** |
            """)
        
        with col2:
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Terminal Value (Perpetuity) | {format_currency(result.terminal_value_perpetuity)} |
            | Terminal Value (Exit Multiple) | {format_currency(result.terminal_value_multiple)} |
            | TV as % of EV | {result.tv_as_pct_ev:.1f}% |
            | Implied EV/Revenue | {result.implied_ev_revenue:.2f}x |
            """)


def render_snapshots_tab(db, scenario_id: str, user_id: str):
    """Render the snapshots management tab."""
    section_header("Forecast Snapshots", "Save and compare forecast versions")
    
    snapshots = load_snapshots(db, scenario_id, limit=20, user_id=user_id)
    
    if not snapshots:
        empty_state(
            "No Snapshots Saved",
            "Save your first forecast snapshot to track versions",
            icon="💾"
        )
        return
    
    # Display snapshots table
    st.markdown("### Saved Snapshots")
    
    for idx, snapshot in enumerate(snapshots):
        with st.expander(f"📷 {snapshot.get('snapshot_name', 'Unnamed')} - {snapshot.get('snapshot_date', 'N/A')}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Revenue:** {format_currency(snapshot.get('total_revenue_forecast', 0))}")
            with col2:
                st.markdown(f"**Gross Profit:** {format_currency(snapshot.get('total_gross_profit_forecast', 0))}")
            with col3:
                locked = "🔒 Locked" if snapshot.get('is_locked') else "🔓 Unlocked"
                st.markdown(f"**Status:** {locked}")
            
            if snapshot.get('notes'):
                st.caption(snapshot['notes'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📥 Load", key=f"load_snap_{snapshot['id']}_{idx}"):
                    try:
                        # Prefer API snapshot endpoint if configured
                        api_base = (os.getenv("FORECAST_API_URL") or "").strip().rstrip("/")
                        if api_base:
                            try:
                                url = urljoin(api_base + "/", f"v1/snapshots/{snapshot['id']}?user_id={user_id}")
                                _headers = {"Content-Type": "application/json"}
                                try:
                                    jwt_token = os.getenv("FORECAST_API_JWT", "").strip()
                                    if jwt_token:
                                        _headers["Authorization"] = f"Bearer {jwt_token}"
                                except Exception:
                                    pass
                                req = urllib.request.Request(url, headers=_headers, method="GET")
                                with urllib.request.urlopen(req, timeout=15) as resp:
                                    raw = resp.read().decode("utf-8")
                                    snap_row = json.loads(raw) if raw else {}
                                if isinstance(snap_row, dict) and snap_row.get("forecast_data"):
                                    st.session_state["forecast_results"] = snap_row.get("forecast_data")
                                    if snap_row.get("monte_carlo_data"):
                                        st.session_state["mc_results"] = snap_row.get("monte_carlo_data")
                                    if snap_row.get("valuation_data"):
                                        st.session_state["valuation_data"] = snap_row.get("valuation_data")
                                    if snap_row.get("enterprise_value") is not None:
                                        st.session_state["enterprise_value"] = snap_row.get("enterprise_value")
                                else:
                                    st.session_state['forecast_results'] = _snapshot_to_forecast_results(snapshot)
                            except Exception:
                                st.session_state['forecast_results'] = _snapshot_to_forecast_results(snapshot)
                        else:
                            st.session_state['forecast_results'] = _snapshot_to_forecast_results(snapshot)
                        st.success("Snapshot loaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading snapshot: {e}")
            
            with col3:
                if not snapshot.get('is_locked'):
                    if st.button("🗑️ Delete", key=f"del_snap_{snapshot['id']}_{idx}"):
                        if delete_snapshot(db, snapshot['id'], user_id):
                            st.success("Snapshot deleted!")
                            st.rerun()


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_forecast_section(db, scenario_id: str, user_id: str):
    """
    Render the complete Forecast & Analysis section.
    
    Args:
        db: Database connector
        scenario_id: Current scenario ID
        user_id: Current user ID
    """
    inject_custom_css()
    
    st.markdown(f"""
    <div style="margin-bottom: 2rem;">
        <h1 style="
            font-size: 2rem;
            font-weight: 700;
            color: #f1f5f9;
            margin: 0 0 0.5rem 0;
        ">📊 Forecast & Analysis</h1>
        <p style="color: #94a3b8;">
            Run projections, analyze risk, and track scenarios
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sub-tabs - NOW INCLUDES VALUATION (NEW IN 8.1) and COMPARISON (NEW)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "▶️ Run Forecast",
        "📈 Results",
        "🔄 Compare Methods",  # NEW: Scenario comparison
        "🎲 Monte Carlo",
        "💰 Valuation",
        "💾 Snapshots"
    ])
    
    with tab1:
        render_run_forecast_tab(db, scenario_id, user_id)
    
    with tab2:
        render_results_tab(db, scenario_id, user_id)
    
    with tab3:
        # NEW: Scenario comparison tab
        try:
            from components.scenario_comparison import render_comparison_tab
            render_comparison_tab(db, scenario_id, user_id)
        except ImportError as e:
            st.error(f"Could not load comparison module: {e}")
        except Exception as e:
            st.error(f"Error in comparison tab: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    with tab4:
        render_monte_carlo_tab(db, scenario_id, user_id)
    
    with tab5:
        render_valuation_tab(db, scenario_id, user_id)
    
    with tab6:
        render_snapshots_tab(db, scenario_id, user_id)


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="CE Africa - Forecast",
        page_icon="📊",
        layout="wide"
    )
    
    inject_custom_css()
    
    st.info("This component requires database connection. Run via main app.")