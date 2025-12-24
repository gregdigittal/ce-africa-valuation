"""
CE Africa Funding UI
====================
Streamlit UI components for:
- Funding events management (debt/equity)
- Overdraft facility configuration
- Trade finance setup
- IRR analysis and goal seek
- Funding visualization

Usage:
    from components.funding_ui import render_funding_section
    render_funding_section(db, scenario_id, user_id, forecast_results)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Any
import json

# Import funding engine
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from funding_engine import (
        FundingEngine, FundingScenario, DebtTranche, EquityInvestment,
        OverdraftFacility, TradeFinanceFacility, CapitalInvestment,
        DebtType, RepaymentType, EquityType, TradeFinanceType,
        create_debt_tranche_from_dict, create_equity_from_dict, create_overdraft_from_dict
    )
    FUNDING_ENGINE_AVAILABLE = True
except ImportError as e:
    FUNDING_ENGINE_AVAILABLE = False
    print(f"Funding engine import error: {e}")

# Trade Finance UI
try:
    from components.trade_finance_ui import render_trade_finance_tab
    TRADE_FINANCE_UI_AVAILABLE = True
except ImportError:
    try:
        from trade_finance_ui import render_trade_finance_tab
        TRADE_FINANCE_UI_AVAILABLE = True
    except ImportError:
        TRADE_FINANCE_UI_AVAILABLE = False
        # Define a placeholder function to prevent errors
        def render_trade_finance_tab(*args, **kwargs):
            st.info("Trade Finance module not available. Please ensure `trade_finance_engine.py` and `trade_finance_ui.py` are installed.")


# =============================================================================
# THEME COLORS
# =============================================================================

COLORS = {
    'bg_base': '#09090B',
    'bg_elevated': '#0F0F11',
    'bg_surface': '#18181B',
    'border': '#27272A',
    'text_primary': '#FAFAFA',
    'text_secondary': '#A1A1AA',
    'text_tertiary': '#71717A',
    'accent': '#D4A537',
    'success': '#22C55E',
    'warning': '#F59E0B',
    'error': '#EF4444',
    'info': '#3B82F6',
    'debt': '#EF4444',      # Red for debt
    'equity': '#22C55E',    # Green for equity
    'overdraft': '#F59E0B', # Orange for overdraft
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_currency(value: float, prefix: str = 'R', decimals: int = 1) -> str:
    """Format number as currency."""
    if value is None:
        return f"{prefix} 0"
    if abs(value) >= 1_000_000_000:
        return f"{prefix} {value / 1_000_000_000:.{decimals}f}B"
    elif abs(value) >= 1_000_000:
        return f"{prefix} {value / 1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{prefix} {value / 1_000:.{decimals}f}K"
    else:
        return f"{prefix} {value:,.0f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format number as percentage."""
    if value is None:
        return "0%"
    return f"{value * 100:.{decimals}f}%"


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply dark theme to chart."""
    fig.update_layout(
        paper_bgcolor=COLORS['bg_elevated'],
        plot_bgcolor=COLORS['bg_elevated'],
        font=dict(family='Inter, sans-serif', color=COLORS['text_secondary'], size=12),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(gridcolor=COLORS['border'], linecolor=COLORS['border']),
        yaxis=dict(gridcolor=COLORS['border'], linecolor=COLORS['border']),
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def load_funding_events(db, scenario_id: str, user_id: str) -> List[Dict]:
    """Load funding events from database."""
    try:
        result = db.client.table('funding_events').select('*').eq(
            'scenario_id', scenario_id
        ).eq('user_id', user_id).eq('is_active', True).order('event_date').execute()
        return result.data or []
    except Exception as e:
        st.error(f"Error loading funding events: {e}")
        return []


def save_funding_event(db, scenario_id: str, user_id: str, data: Dict) -> bool:
    """Save a funding event."""
    try:
        data['scenario_id'] = scenario_id
        data['user_id'] = user_id
        db.client.table('funding_events').insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Error saving funding event: {e}")
        return False


def update_funding_event(db, event_id: str, user_id: str, data: Dict) -> bool:
    """Update a funding event."""
    try:
        db.client.table('funding_events').update(data).eq('id', event_id).eq('user_id', user_id).execute()
        return True
    except Exception as e:
        st.error(f"Error updating funding event: {e}")
        return False


def delete_funding_event(db, event_id: str, user_id: str) -> bool:
    """Soft delete a funding event."""
    try:
        db.client.table('funding_events').update({'is_active': False}).eq(
            'id', event_id).eq('user_id', user_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting funding event: {e}")
        return False


def load_overdraft_facility(db, scenario_id: str, user_id: str) -> Optional[Dict]:
    """Load overdraft facility configuration."""
    try:
        result = db.client.table('overdraft_facilities').select('*').eq(
            'scenario_id', scenario_id
        ).eq('user_id', user_id).single().execute()
        return result.data
    except:
        return None


def save_overdraft_facility(db, scenario_id: str, user_id: str, data: Dict) -> bool:
    """Save or update overdraft facility."""
    try:
        data['scenario_id'] = scenario_id
        data['user_id'] = user_id
        
        # Check if exists first
        existing = load_overdraft_facility(db, scenario_id, user_id)
        
        if existing:
            # Update existing
            db.client.table('overdraft_facilities').update(data).eq(
                'scenario_id', scenario_id
            ).eq('user_id', user_id).execute()
        else:
            # Insert new
            db.client.table('overdraft_facilities').insert(data).execute()
        
        return True
    except Exception as e:
        st.error(f"Error saving overdraft: {e}")
        return False


def load_trade_finance_facilities(db, scenario_id: str, user_id: str) -> List[Dict]:
    """Load trade finance facilities."""
    try:
        result = db.client.table('trade_finance_facilities').select('*').eq(
            'scenario_id', scenario_id
        ).eq('user_id', user_id).eq('is_active', True).execute()
        return result.data or []
    except:
        return []


def save_trade_finance_facility(db, scenario_id: str, user_id: str, data: Dict) -> bool:
    """Save trade finance facility."""
    try:
        data['scenario_id'] = scenario_id
        data['user_id'] = user_id
        db.client.table('trade_finance_facilities').insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Error saving trade finance: {e}")
        return False


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_funding_timeline_chart(funded_cf: Dict, height: int = 400) -> go.Figure:
    """Create funding timeline visualization."""
    timeline = funded_cf.get('timeline', [])
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Cash Flow Components', 'Cumulative Cash & Debt Balance'),
        row_heights=[0.5, 0.5]
    )
    
    # Cash flow components (stacked bar)
    fig.add_trace(go.Bar(
        x=timeline,
        y=funded_cf.get('base_cash_flow', []),
        name='Operating CF',
        marker_color=COLORS['accent'],
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=timeline,
        y=[-x for x in funded_cf.get('debt_principal_payment', [])],
        name='Debt Principal',
        marker_color=COLORS['debt'],
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=timeline,
        y=[-x for x in funded_cf.get('debt_interest_payment', [])],
        name='Debt Interest',
        marker_color='#FF6B6B',
    ), row=1, col=1)
    
    # Cumulative cash and debt balance
    fig.add_trace(go.Scatter(
        x=timeline,
        y=[x / 1_000_000 for x in funded_cf.get('cumulative_cash', [])],
        name='Cash Balance',
        line=dict(color=COLORS['success'], width=2),
        fill='tozeroy',
        fillcolor='rgba(34, 197, 94, 0.2)',
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=timeline,
        y=[x / 1_000_000 for x in funded_cf.get('debt_balance', [])],
        name='Debt Balance',
        line=dict(color=COLORS['debt'], width=2, dash='dash'),
    ), row=2, col=1)
    
    if any(funded_cf.get('overdraft_balance', [])):
        fig.add_trace(go.Scatter(
            x=timeline,
            y=[x / 1_000_000 for x in funded_cf.get('overdraft_balance', [])],
            name='Overdraft',
            line=dict(color=COLORS['overdraft'], width=2, dash='dot'),
        ), row=2, col=1)
    
    fig.update_layout(
        barmode='relative',
        height=height,
        title='Funding Analysis',
        yaxis_title='Cash Flow (R)',
        yaxis2_title='Balance (R millions)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    
    return apply_chart_theme(fig)


def create_capital_structure_chart(scenario: 'FundingScenario', height: int = 300) -> go.Figure:
    """Create capital structure pie chart."""
    values = []
    labels = []
    colors = []
    
    if scenario.total_equity > 0:
        values.append(scenario.total_equity)
        labels.append('Equity')
        colors.append(COLORS['equity'])
    
    for tranche in scenario.debt_tranches:
        if tranche.outstanding_principal > 0:
            values.append(tranche.outstanding_principal)
            labels.append(tranche.name)
            colors.append(COLORS['debt'])
    
    if scenario.overdraft and scenario.overdraft.current_drawn > 0:
        values.append(scenario.overdraft.current_drawn)
        labels.append('Overdraft')
        colors.append(COLORS['overdraft'])
    
    if not values:
        values = [1]
        labels = ['No funding']
        colors = [COLORS['text_tertiary']]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside',
    )])
    
    fig.update_layout(
        title='Capital Structure',
        height=height,
        showlegend=False,
    )
    
    return apply_chart_theme(fig)


def create_irr_sensitivity_chart(sensitivity_df: pd.DataFrame, height: int = 350) -> go.Figure:
    """Create IRR sensitivity heatmap."""
    # Parse the DataFrame to extract numeric IRR values
    equity_labels = sensitivity_df['Equity'].tolist()
    tv_cols = [col for col in sensitivity_df.columns if col != 'Equity']
    
    z_values = []
    for _, row in sensitivity_df.iterrows():
        row_values = []
        for col in tv_cols:
            val = row[col]
            if val == 'N/A':
                row_values.append(None)
            else:
                try:
                    row_values.append(float(val.strip('%')) / 100)
                except:
                    row_values.append(None)
        z_values.append(row_values)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=tv_cols,
        y=equity_labels,
        colorscale='RdYlGn',
        text=[[f'{v:.1%}' if v else 'N/A' for v in row] for row in z_values],
        texttemplate='%{text}',
        textfont=dict(size=11),
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title='IRR Sensitivity Analysis',
        xaxis_title='Terminal Value Multiplier',
        yaxis_title='Equity Multiplier',
        height=height,
    )
    
    return apply_chart_theme(fig)


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_debt_form(key_prefix: str = "debt"):
    """Render form for adding debt instrument."""
    st.markdown("#### Add Debt Instrument")
    
    col1, col2 = st.columns(2)
    
    with col1:
        debt_type = st.selectbox(
            "Debt Type",
            options=[
                ('debt_term_loan', 'Term Loan'),
                ('debt_mezzanine', 'Mezzanine'),
                ('debt_convertible', 'Convertible Note'),
                ('debt_trade_finance', 'Trade Finance'),
            ],
            format_func=lambda x: x[1],
            key=f"{key_prefix}_type"
        )[0]
        
        amount = st.number_input(
            "Principal Amount (R)",
            min_value=0.0,
            value=1_000_000.0,
            step=100_000.0,
            key=f"{key_prefix}_amount"
        )
        
        event_date = st.date_input(
            "Draw Date",
            value=date.today(),
            key=f"{key_prefix}_date"
        )
    
    with col2:
        interest_rate = st.number_input(
            "Interest Rate (% p.a.)",
            min_value=0.0,
            max_value=50.0,
            value=12.0,
            step=0.5,
            key=f"{key_prefix}_rate"
        )
        
        term_months = st.number_input(
            "Term (months)",
            min_value=1,
            max_value=360,
            value=60,
            key=f"{key_prefix}_term"
        )
        
        repayment_type = st.selectbox(
            "Repayment Structure",
            options=[
                ('amortizing', 'Amortizing (Equal P+I)'),
                ('bullet', 'Bullet (Principal at Maturity)'),
                ('interest_only', 'Interest Only'),
                ('pik', 'PIK (Payment-in-Kind)'),
            ],
            format_func=lambda x: x[1],
            key=f"{key_prefix}_repay_type"
        )[0]
    
    # Advanced options
    with st.expander("Advanced Options"):
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            grace_period = st.number_input(
                "Grace Period (months)",
                min_value=0,
                max_value=24,
                value=0,
                key=f"{key_prefix}_grace"
            )
            
            balloon_pct = st.number_input(
                "Balloon Payment (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                key=f"{key_prefix}_balloon"
            )
        
        with adv_col2:
            if debt_type == 'debt_mezzanine':
                pik_rate = st.number_input(
                    "PIK Rate (% p.a.)",
                    min_value=0.0,
                    max_value=20.0,
                    value=3.0,
                    key=f"{key_prefix}_pik"
                )
            else:
                pik_rate = 0.0
            
            if debt_type == 'debt_convertible':
                conversion_price = st.number_input(
                    "Conversion Price (R/share)",
                    min_value=0.0,
                    value=100.0,
                    key=f"{key_prefix}_conv_price"
                )
            else:
                conversion_price = 0.0
    
    description = st.text_input("Description", key=f"{key_prefix}_desc")
    
    return {
        'event_type': debt_type,
        'amount': amount,
        'event_date': event_date.isoformat(),
        'interest_rate': interest_rate,
        'term_months': term_months,
        'repayment_type': repayment_type,
        'grace_period_months': grace_period,
        'balloon_pct': balloon_pct,
        'pik_rate': pik_rate if debt_type == 'debt_mezzanine' else None,
        'conversion_price': conversion_price if debt_type == 'debt_convertible' else None,
        'description': description,
    }


def render_equity_form(key_prefix: str = "equity"):
    """Render form for adding equity investment."""
    st.markdown("#### Add Equity Investment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        equity_type = st.selectbox(
            "Equity Type",
            options=[
                ('equity_ordinary', 'Ordinary Shares'),
                ('equity_preference', 'Preference Shares'),
            ],
            format_func=lambda x: x[1],
            key=f"{key_prefix}_type"
        )[0]
        
        amount = st.number_input(
            "Investment Amount (R)",
            min_value=0.0,
            value=5_000_000.0,
            step=500_000.0,
            key=f"{key_prefix}_amount"
        )
        
        event_date = st.date_input(
            "Investment Date",
            value=date.today(),
            key=f"{key_prefix}_date"
        )
    
    with col2:
        investor_name = st.text_input(
            "Investor Name",
            value="",
            key=f"{key_prefix}_investor"
        )
        
        share_price = st.number_input(
            "Share Price (R)",
            min_value=0.01,
            value=100.0,
            key=f"{key_prefix}_price"
        )
        
        if equity_type == 'equity_preference':
            dividend_rate = st.number_input(
                "Dividend Rate (% p.a.)",
                min_value=0.0,
                max_value=20.0,
                value=8.0,
                key=f"{key_prefix}_div_rate"
            )
        else:
            dividend_rate = 0.0
    
    shares_issued = amount / share_price if share_price > 0 else 0
    st.info(f"Shares to be issued: {shares_issued:,.0f}")
    
    return {
        'event_type': equity_type,
        'amount': amount,
        'event_date': event_date.isoformat(),
        'investor_name': investor_name,
        'share_price': share_price,
        'shares_issued': shares_issued,
        'dividend_rate': dividend_rate if equity_type == 'equity_preference' else None,
        'description': f"{investor_name} - {equity_type.replace('equity_', '').title()} Shares",
    }


def render_overdraft_config(db, scenario_id: str, user_id: str):
    """Render overdraft facility configuration."""
    st.markdown("#### Overdraft Facility")
    
    existing = load_overdraft_facility(db, scenario_id, user_id)
    
    col1, col2 = st.columns(2)
    
    with col1:
        is_active = st.checkbox(
            "Enable Overdraft Facility",
            value=existing.get('is_active', True) if existing else True,
            key="od_active"
        )
        
        facility_limit = st.number_input(
            "Facility Limit (R)",
            min_value=0.0,
            value=float(existing.get('facility_limit', 5_000_000)) if existing else 5_000_000.0,
            step=500_000.0,
            key="od_limit"
        )
        
        interest_rate = st.number_input(
            "Interest Rate (% p.a.)",
            min_value=0.0,
            max_value=30.0,
            value=float(existing.get('interest_rate', 12)) if existing else 12.0,
            step=0.5,
            key="od_rate"
        )
    
    with col2:
        auto_repay = st.checkbox(
            "Auto-Repay When Cash Positive",
            value=existing.get('auto_repay', True) if existing else True,
            key="od_auto_repay"
        )
        
        arrangement_fee = st.number_input(
            "Arrangement Fee (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(existing.get('arrangement_fee_pct', 0.5)) if existing else 0.5,
            step=0.1,
            key="od_arr_fee"
        )
        
        commitment_fee = st.number_input(
            "Commitment Fee (% on undrawn)",
            min_value=0.0,
            max_value=2.0,
            value=float(existing.get('commitment_fee_pct', 0.25)) if existing else 0.25,
            step=0.05,
            key="od_commit_fee"
        )
    
    if st.button("Save Overdraft Configuration", key="save_od"):
        data = {
            'facility_limit': facility_limit,
            'interest_rate': interest_rate,
            'arrangement_fee_pct': arrangement_fee,
            'commitment_fee_pct': commitment_fee,
            'auto_repay': auto_repay,
            'is_active': is_active,
        }
        if save_overdraft_facility(db, scenario_id, user_id, data):
            st.success("Overdraft configuration saved!")
            st.rerun()


def render_funding_events_list(db, scenario_id: str, user_id: str):
    """Render list of existing funding events."""
    events = load_funding_events(db, scenario_id, user_id)
    
    if not events:
        st.info("No funding events configured. Add debt or equity above.")
        return
    
    st.markdown("#### Current Funding Structure")
    
    # Separate debt and equity
    debt_events = [e for e in events if e['event_type'].startswith('debt')]
    equity_events = [e for e in events if e['event_type'].startswith('equity')]
    
    if equity_events:
        st.markdown("**Equity**")
        for e in equity_events:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.write(f"📈 {e.get('description', 'Equity')} - {e.get('investor_name', 'Unknown')}")
            with col2:
                st.write(format_currency(e['amount']))
            with col3:
                st.write(e['event_date'])
            with col4:
                if st.button("🗑️", key=f"del_eq_{e['id']}"):
                    delete_funding_event(db, e['id'], user_id)
                    st.rerun()
    
    if debt_events:
        st.markdown("**Debt**")
        for e in debt_events:
            col1, col2, col3, col4, col5 = st.columns([2.5, 1.5, 1.5, 1.5, 1])
            with col1:
                debt_icon = "🏦" if 'term' in e['event_type'] else "💰"
                st.write(f"{debt_icon} {e.get('description', 'Debt')}")
            with col2:
                st.write(format_currency(e['amount']))
            with col3:
                st.write(f"{e.get('interest_rate', 0):.1f}%")
            with col4:
                st.write(f"{e.get('term_months', 0)} mo")
            with col5:
                if st.button("🗑️", key=f"del_debt_{e['id']}"):
                    delete_funding_event(db, e['id'], user_id)
                    st.rerun()


def render_irr_analysis(
    db,
    scenario_id: str,
    user_id: str,
    forecast_results: Dict[str, Any]
):
    """Render IRR analysis section."""
    st.markdown("### IRR Analysis")
    
    if not FUNDING_ENGINE_AVAILABLE:
        st.warning("Funding engine not available.")
        return
    
    # Check if forecast results are available
    if not forecast_results:
        st.info("💡 **Forecast Required** - Run a forecast first to enable IRR analysis.")
        st.markdown("""
        To calculate IRR, the system needs:
        - Forecast cash flows
        - Profit & Loss data
        - Balance sheet projections
        
        Go to **Forecast** section and run a forecast, then return here.
        """)
        return
    
    # Load funding data
    events = load_funding_events(db, scenario_id, user_id)
    overdraft_data = load_overdraft_facility(db, scenario_id, user_id)
    
    # Build funding scenario
    scenario = FundingScenario()
    
    for e in events:
        if e['event_type'].startswith('debt'):
            scenario.debt_tranches.append(create_debt_tranche_from_dict(e))
        elif e['event_type'].startswith('equity'):
            scenario.equity_investments.append(create_equity_from_dict(e))
    
    if overdraft_data:
        scenario.overdraft = create_overdraft_from_dict(overdraft_data)
    
    # Check if we have data to analyze
    if not scenario.equity_investments and not scenario.debt_tranches:
        st.info("Add funding events to enable IRR analysis.")
        return
    
    # Apply funding
    engine = FundingEngine()
    funded_cf = engine.apply_funding(forecast_results, scenario)
    
    # IRR inputs
    col1, col2 = st.columns(2)
    
    with col1:
        terminal_multiple = st.slider(
            "Exit Multiple (x EBITDA)",
            min_value=3.0,
            max_value=12.0,
            value=6.0,
            step=0.5,
            key="irr_exit_mult"
        )
    
    with col2:
        holding_period = st.slider(
            "Holding Period (years)",
            min_value=3,
            max_value=10,
            value=5,
            key="irr_holding"
        )
    
    # Calculate terminal value
    # Use final year EBITDA * multiple
    ebit_data = forecast_results.get('profit', {}).get('ebit', [])
    if ebit_data:
        # Annualize final EBIT
        final_12m_ebit = sum(ebit_data[-12:]) if len(ebit_data) >= 12 else sum(ebit_data)
        # Add back depreciation assumption for EBITDA (simplified: EBITDA ≈ 1.1 * EBIT)
        final_ebitda = final_12m_ebit * 1.1
        terminal_value = final_ebitda * terminal_multiple
    else:
        terminal_value = scenario.total_equity * 3  # Fallback
    
    st.metric("Implied Exit Value", format_currency(terminal_value))
    
    # Calculate IRR
    total_equity = scenario.total_equity
    if total_equity > 0:
        equity_irr = engine.calculate_equity_irr(
            funded_cf,
            total_equity,
            terminal_value,
            holding_period * 12
        )
        
        # Display IRR
        col1, col2, col3 = st.columns(3)
        
        with col1:
            irr_color = COLORS['success'] if equity_irr > 0.20 else (
                COLORS['warning'] if equity_irr > 0.10 else COLORS['error']
            )
            st.markdown(f"""
            <div style="background: {COLORS['bg_surface']}; padding: 1rem; border-radius: 8px; text-align: center;">
                <p style="color: {COLORS['text_tertiary']}; margin: 0; font-size: 0.8rem;">EQUITY IRR</p>
                <p style="color: {irr_color}; margin: 0; font-size: 1.8rem; font-weight: 700;">{equity_irr:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            moic = (terminal_value + sum(funded_cf.get('net_cash_flow', []))) / total_equity if total_equity > 0 else 0
            st.markdown(f"""
            <div style="background: {COLORS['bg_surface']}; padding: 1rem; border-radius: 8px; text-align: center;">
                <p style="color: {COLORS['text_tertiary']}; margin: 0; font-size: 0.8rem;">MOIC</p>
                <p style="color: {COLORS['text_primary']}; margin: 0; font-size: 1.8rem; font-weight: 700;">{moic:.2f}x</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            de_ratio = scenario.debt_to_equity_ratio
            st.markdown(f"""
            <div style="background: {COLORS['bg_surface']}; padding: 1rem; border-radius: 8px; text-align: center;">
                <p style="color: {COLORS['text_tertiary']}; margin: 0; font-size: 0.8rem;">D/E RATIO</p>
                <p style="color: {COLORS['text_primary']}; margin: 0; font-size: 1.8rem; font-weight: 700;">{de_ratio:.2f}x</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Sensitivity analysis
        st.markdown("---")
        st.markdown("#### Sensitivity Analysis")
        
        sensitivity_df = engine.sensitivity_analysis(
            funded_cf,
            total_equity,
            terminal_value,
            equity_range=(0.7, 1.3),
            tv_range=(0.7, 1.3),
            steps=5
        )
        
        # Display as table
        st.dataframe(sensitivity_df, hide_index=True, use_container_width=True)
    
    # Goal Seek
    st.markdown("---")
    st.markdown("#### Goal Seek")
    
    gs_col1, gs_col2 = st.columns(2)
    
    with gs_col1:
        target_irr = st.number_input(
            "Target IRR (%)",
            min_value=5.0,
            max_value=50.0,
            value=25.0,
            step=1.0,
            key="gs_target_irr"
        ) / 100
    
    with gs_col2:
        if st.button("Calculate Required Equity", key="gs_calc"):
            required_equity = engine.goal_seek_equity_for_irr(
                target_irr,
                funded_cf,
                terminal_value,
                total_equity,
                min_equity=total_equity * 0.3,
                max_equity=total_equity * 3
            )
            
            if required_equity:
                st.success(f"Required equity for {target_irr:.0%} IRR: {format_currency(required_equity)}")
                change = (required_equity / total_equity - 1) * 100
                st.info(f"This is {change:+.1f}% vs current equity of {format_currency(total_equity)}")
            else:
                st.warning("Target IRR not achievable within constraints.")


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_funding_section(
    db,
    scenario_id: str,
    user_id: str,
    forecast_results: Dict[str, Any] = None
):
    """
    Main entry point for Funding section.
    """
    st.header("Funding & Returns")
    
    if not FUNDING_ENGINE_AVAILABLE:
        st.error("Funding engine not available. Please ensure `funding_engine.py` is installed.")
        return
    
    tabs = st.tabs(["Funding Structure", "Overdraft", "Trade Finance", "IRR Analysis"])
    
    # Tab 1: Funding Structure
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("Add Debt", expanded=False):
                debt_data = render_debt_form("new_debt")
                if st.button("Add Debt Instrument", key="add_debt_btn", type="primary"):
                    if save_funding_event(db, scenario_id, user_id, debt_data):
                        st.success("Debt instrument added!")
                        st.rerun()
        
        with col2:
            with st.expander("Add Equity", expanded=False):
                equity_data = render_equity_form("new_equity")
                if st.button("Add Equity Investment", key="add_equity_btn", type="primary"):
                    if save_funding_event(db, scenario_id, user_id, equity_data):
                        st.success("Equity investment added!")
                        st.rerun()
        
        st.markdown("---")
        render_funding_events_list(db, scenario_id, user_id)
        
        # Visualization
        if forecast_results:
            events = load_funding_events(db, scenario_id, user_id)
            overdraft_data = load_overdraft_facility(db, scenario_id, user_id)
            
            if events or overdraft_data:
                st.markdown("---")
                
                # Build scenario and apply funding
                scenario = FundingScenario()
                for e in events:
                    if e['event_type'].startswith('debt'):
                        scenario.debt_tranches.append(create_debt_tranche_from_dict(e))
                    elif e['event_type'].startswith('equity'):
                        scenario.equity_investments.append(create_equity_from_dict(e))
                
                if overdraft_data:
                    scenario.overdraft = create_overdraft_from_dict(overdraft_data)
                
                engine = FundingEngine()
                funded_cf = engine.apply_funding(forecast_results, scenario)
                
                # Charts
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    timeline_fig = create_funding_timeline_chart(funded_cf)
                    st.plotly_chart(timeline_fig, use_container_width=True)
                
                with col2:
                    structure_fig = create_capital_structure_chart(scenario)
                    st.plotly_chart(structure_fig, use_container_width=True)
    
    # Tab 2: Overdraft
    with tabs[1]:
        render_overdraft_config(db, scenario_id, user_id)
    
    # Tab 3: Trade Finance
    with tabs[2]:
        if TRADE_FINANCE_UI_AVAILABLE:
            render_trade_finance_tab(db, scenario_id, user_id, forecast_results)
        else:
            st.markdown("### Trade Finance")
            st.info("Trade Finance module not available. Please ensure `trade_finance_engine.py` and `trade_finance_ui.py` are installed.")
            
            st.markdown("""
            **Trade Finance Features:**
            
            **Simplified Mode:**
            - Set % of purchases to finance
            - Average payment structures (domestic vs import)
            - Auto-calculate duties, VAT, shipping costs
            
            **Creditor-Linked Mode:**
            - Configure trade finance per supplier
            - Define custom payment stages
            - 50% on order, 50% on completion
            - Import: 30% order, 40% shipment, 30% arrival + duties
            
            **What-If Analysis:**
            - "What if we increased import sourcing to 50%?"
            - "What if we negotiated better finance rates?"
            - LLM-powered scenario recommendations
            """)
    
    # Tab 4: IRR Analysis
    with tabs[3]:
        if forecast_results:
            render_irr_analysis(db, scenario_id, user_id, forecast_results)
        else:
            st.warning("Run a forecast first to enable IRR analysis.")


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Funding UI Test", layout="wide")
    st.warning("Running in standalone test mode.")
