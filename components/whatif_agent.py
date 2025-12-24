"""
What-If Agent Component
=======================
Sprint 17: Interactive what-if scenario analysis.
Sprint 18: Enhanced with full calculation engine and sensitivity analysis.

Version: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List
from copy import deepcopy

def render_whatif_agent(
    db,
    scenario_id: str,
    user_id: str,
    baseline_forecast: Optional[Dict[str, Any]] = None
):
    """
    Render the What-If Agent section for scenario analysis.
    
    Args:
        db: SupabaseHandler instance
        scenario_id: Current scenario UUID
        user_id: Current user UUID
        baseline_forecast: Optional baseline forecast results
    """
    st.header("🔮 What-If Agent")
    st.markdown("""
    **Sprint 18 Enhanced + Sprint 23 LLM: Interactive Scenario Analysis with Natural Language Optimization**
    
    The What-If Agent allows you to:
    - ✅ Adjust key assumptions interactively with sliders
    - ✅ See immediate impact on forecast results (real-time)
    - ✅ Compare baseline vs adjusted scenarios side-by-side
    - ✅ Run sensitivity analysis with tornado diagrams
    - ✅ **NEW: Ask questions in natural language** (e.g., "Maximize return with 25% equity limit")
    - ✅ **NEW: Automatic scenario optimization** based on your objectives
    - ✅ Identify most impactful parameters
    - ✅ Export what-if analysis results
    """)
    
    if not baseline_forecast:
        st.info("""
        💡 **No Baseline Forecast Found**
        
        Run a forecast first to create a baseline, then use the What-If Agent
        to explore different scenarios.
        """)
        
        if st.button("→ Go to Forecast", type="primary"):
            st.session_state.current_section = 'forecast'
            st.rerun()
        return
    
    st.success("✅ Baseline forecast loaded. What-If analysis ready.")
    
    # Tabs for different modes
    mode_tabs = st.tabs(["🎯 Natural Language", "⚙️ Manual Adjustments", "📊 Sensitivity Analysis"])
    
    # ==========================================================================
    # TAB 1: Natural Language Optimization (Sprint 23)
    # ==========================================================================
    with mode_tabs[0]:
        render_natural_language_tab(db, scenario_id, user_id, baseline_forecast)
    
    # ==========================================================================
    # TAB 2: Manual Adjustments (Original)
    # ==========================================================================
    with mode_tabs[1]:
        render_manual_adjustments_tab(baseline_forecast)
    
    # ==========================================================================
    # TAB 3: Sensitivity Analysis (Sprint 18)
    # ==========================================================================
    with mode_tabs[2]:
        render_sensitivity_analysis_tab(baseline_forecast)


def render_natural_language_tab(
    db,
    scenario_id: str,
    user_id: str,
    baseline_forecast: Dict[str, Any]
):
    """Render natural language optimization tab."""
    st.markdown("### 🤖 Natural Language Scenario Optimization")
    st.markdown("""
    Ask questions in natural language to automatically optimize your scenario.
    
    **Examples:**
    - "Find the optimal mix of debt, equity, overdraft, and trade finance to maximize the return to shareholders but with the limit of equity that the shareholders are prepared to give up to a private equity investor of 25%"
    - "Maximize IRR with debt less than 50%"
    - "Minimize cost of capital while maintaining 20% margin"
    """)
    
    # Initialize LLM engine
    try:
        from components.llm_prompt_engine import LLMPromptEngine, ScenarioOptimizer
        llm_engine = LLMPromptEngine()
        llm_available = llm_engine.is_available()
    except ImportError:
        llm_available = False
        st.info("💡 LLM integration not available. Install OpenAI or Anthropic SDK to enable natural language queries.")
    
    # Query input
    query = st.text_area(
        "Enter your optimization query:",
        placeholder='e.g., "Find optimal funding mix to maximize return with max 25% equity dilution"',
        height=100,
        key="nl_query"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        parse_button = st.button("🔍 Parse Query", type="primary", use_container_width=True)
    
    with col2:
        optimize_button = st.button("🚀 Optimize Scenario", type="secondary", use_container_width=True, disabled=not query)
    
    if parse_button and query:
        with st.spinner("Parsing your query..."):
            if llm_available:
                parsed = llm_engine.parse_query(query)
            else:
                # Use fallback parser
                parsed = llm_engine._fallback_parse(query) if hasattr(llm_engine, '_fallback_parse') else {
                    "intent": "optimize",
                    "objective": {"type": "maximize", "metric": "return_to_shareholders"},
                    "constraints": [],
                    "parameters": {}
                }
            
            st.session_state['nl_parsed_query'] = parsed
            
            # Display parsed query
            st.markdown("#### 📋 Parsed Query")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Objective:**")
                st.json({
                    "Type": parsed['objective']['type'],
                    "Metric": parsed['objective']['metric']
                })
            
            with col2:
                st.markdown("**Constraints:**")
                if parsed.get('constraints'):
                    for constraint in parsed['constraints']:
                        st.write(f"- {constraint.get('description', constraint.get('type', 'Unknown'))}")
                else:
                    st.write("None")
            
            if parsed.get('notes'):
                st.info(f"ℹ️ {parsed['notes']}")
    
    if optimize_button and query:
        # Get parsed query or parse now
        parsed = st.session_state.get('nl_parsed_query')
        if not parsed:
            with st.spinner("Parsing query..."):
                if llm_available:
                    parsed = llm_engine.parse_query(query)
                else:
                    parsed = llm_engine._fallback_parse(query) if hasattr(llm_engine, '_fallback_parse') else {
                        "intent": "optimize",
                        "objective": {"type": "maximize", "metric": "return_to_shareholders"},
                        "constraints": []
                    }
        
        # Run optimization
        with st.spinner("Optimizing scenario... This may take a moment."):
            optimizer = ScenarioOptimizer(baseline_forecast)
            
            # Check if this is a funding optimization
            is_funding_optimization = any(
                'debt' in str(c.get('type', '')).lower() or 
                'equity' in str(c.get('type', '')).lower() or
                'funding' in query.lower()
                for c in parsed.get('constraints', [])
            )
            
            if is_funding_optimization:
                # Funding mix optimization
                from components.llm_prompt_engine import optimize_funding_mix
                result = optimize_funding_mix(
                    baseline_forecast,
                    parsed['objective']['metric'],
                    parsed.get('constraints', []),
                    db=db,
                    scenario_id=scenario_id,
                    user_id=user_id
                )
            else:
                # Forecast parameter optimization
                result = optimizer.optimize(
                    parsed['objective'],
                    parsed.get('constraints', [])
                )
            
            if result.get('success'):
                st.success("✅ Optimization complete!")
                
                # Display results
                st.markdown("#### 🎯 Optimal Solution")
                
                if 'optimal_parameters' in result:
                    optimal_params = result['optimal_parameters']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Revenue", f"{optimal_params.get('revenue_pct', 0)*100:+.1f}%")
                    with col2:
                        st.metric("Utilization", f"{optimal_params.get('utilization_pct', 0)*100:+.1f}%")
                    with col3:
                        st.metric("COGS", f"{optimal_params.get('cogs_pct', 0)*100:+.1f}%")
                    with col4:
                        st.metric("OPEX", f"{optimal_params.get('opex_pct', 0)*100:+.1f}%")
                    
                    # Show optimal forecast
                    if 'optimal_forecast' in result:
                        st.markdown("---")
                        st.markdown("#### 📊 Optimal Forecast Results")
                        display_whatif_results(
                            baseline_forecast,
                            result['optimal_forecast'],
                            optimal_params
                        )
                
                elif 'optimal_mix' in result:
                    # Funding mix results
                    optimal_mix = result['optimal_mix']
                    
                    st.markdown("**Optimal Funding Mix:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Debt", f"{optimal_mix.get('debt_pct', 0)*100:.1f}%")
                    with col2:
                        st.metric("Equity", f"{optimal_mix.get('equity_pct', 0)*100:.1f}%")
                    with col3:
                        st.metric("Overdraft", f"{optimal_mix.get('overdraft_pct', 0)*100:.1f}%")
                    with col4:
                        st.metric("Trade Finance", f"{optimal_mix.get('trade_finance_pct', 0)*100:.1f}%")
                    
                    if 'optimal_irr' in result:
                        st.metric("Optimal Equity IRR", f"{result['optimal_irr']*100:.1f}%")
            else:
                st.error(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")


def render_manual_adjustments_tab(baseline_forecast: Dict[str, Any]):
    """Render manual adjustments tab (original functionality)."""
    st.markdown("### ⚙️ Manual Parameter Adjustments")
    
    # What-If Parameters
    st.markdown("#### Adjust Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Revenue Assumptions")
        revenue_adjustment = st.slider(
            "Revenue Adjustment (%)",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=1.0,
            help="Adjust revenue by percentage"
        )
        
        utilization_adjustment = st.slider(
            "Utilization Adjustment (%)",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=1.0,
            help="Adjust machine utilization"
        )
    
    with col2:
        st.markdown("#### Cost Assumptions")
        cogs_adjustment = st.slider(
            "COGS Adjustment (%)",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Adjust cost of goods sold"
        )
        
        opex_adjustment = st.slider(
            "OPEX Adjustment (%)",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=1.0,
            help="Adjust operating expenses"
        )
    
    # Store adjustments in session state for real-time updates
    adjustments = {
        'revenue_pct': revenue_adjustment / 100,
        'utilization_pct': utilization_adjustment / 100,
        'cogs_pct': cogs_adjustment / 100,
        'opex_pct': opex_adjustment / 100
    }
    
    # Calculate adjusted forecast
    adjusted_forecast = calculate_adjusted_forecast(baseline_forecast, adjustments)
    
    # Show results automatically (no button needed for real-time)
    if adjusted_forecast:
        display_whatif_results(baseline_forecast, adjusted_forecast, adjustments)
        # Optional: show DCF valuation delta (if available)
        try:
            from api.valuation import run_dcf_valuation
            base_assumptions = baseline_forecast.get("assumptions", {}) or {}
            base_val, base_ev = run_dcf_valuation(baseline_forecast, base_assumptions, net_debt=0.0)
            adj_val, adj_ev = run_dcf_valuation(adjusted_forecast, base_assumptions, net_debt=0.0)
            st.markdown("#### 💰 Valuation (DCF)")
            st.metric("Enterprise Value (Adjusted)", f"R {adj_ev:,.0f}", delta=f"R {adj_ev - base_ev:,.0f}")
        except Exception:
            pass
    
    # Scenario Comparison
    st.markdown("---")
    st.markdown("### Save What-If Scenario")
    
    scenario_name = st.text_input(
        "Scenario Name",
        value=f"What-If: {revenue_adjustment:+.0f}% Revenue, {cogs_adjustment:+.0f}% COGS",
        help="Name for this what-if scenario"
    )
    
    if st.button("Save Scenario", type="secondary"):
        st.success(f"✅ Scenario '{scenario_name}' saved (feature in development)")
    
    # Scenario Comparison
    st.markdown("---")
    st.markdown("### Save What-If Scenario")
    
    scenario_name = st.text_input(
        "Scenario Name",
        value=f"What-If: {revenue_adjustment:+.0f}% Revenue, {cogs_adjustment:+.0f}% COGS",
        help="Name for this what-if scenario"
    )
    
    if st.button("Save Scenario", type="secondary"):
        st.success(f"✅ Scenario '{scenario_name}' saved (feature in development)")
    
    # Information
    st.markdown("---")
    with st.expander("ℹ️ About What-If Agent"):
        st.markdown("""
        **What-If Agent (Sprint 17)**
        
        This component enables interactive scenario analysis:
        
        - **Parameter Adjustment:** Adjust key assumptions with sliders
        - **Real-time Impact:** See immediate effect on financials
        - **Scenario Comparison:** Compare multiple what-if scenarios
        - **Export:** Save and export what-if analysis
        
        **Status:** ✅ Full interactive analysis implemented (Sprint 18)
        
        **Features:**
        - Real-time forecast recalculation
        - Side-by-side comparison charts
        - Sensitivity analysis with tornado diagrams
        - Parameter impact ranking
        - Detailed comparison tables
        """)
    
    # Store current adjustments for sensitivity analysis
    st.session_state['whatif_adjustments'] = adjustments
    st.session_state['whatif_baseline'] = baseline_forecast
    st.session_state['whatif_adjusted'] = adjusted_forecast


def render_sensitivity_analysis_tab(baseline_forecast: Dict[str, Any]):
    """Render sensitivity analysis tab."""
    st.markdown("### 🔍 Sensitivity Analysis")
    
    if st.button("Run Sensitivity Analysis", type="primary", use_container_width=True):
        sensitivity_results = run_sensitivity_analysis(baseline_forecast)
        if sensitivity_results:
            display_sensitivity_analysis(sensitivity_results)
        else:
            st.warning("Sensitivity analysis failed. Please check baseline forecast.")


# =============================================================================
# CALCULATION ENGINE (Sprint 18)
# =============================================================================

def calculate_adjusted_forecast(
    baseline: Dict[str, Any],
    adjustments: Dict[str, float]
) -> Optional[Dict[str, Any]]:
    """
    Apply adjustments to baseline forecast and recalculate financial statements.
    
    Args:
        baseline: Baseline forecast results
        adjustments: Dictionary of adjustment percentages
            - revenue_pct: Revenue adjustment (e.g., 0.10 = +10%)
            - utilization_pct: Utilization adjustment
            - cogs_pct: COGS adjustment
            - opex_pct: OPEX adjustment
    
    Returns:
        Adjusted forecast results with same structure as baseline
    """
    if not baseline:
        return None
    
    # Deep copy to avoid modifying baseline
    adjusted = deepcopy(baseline)
    
    # Extract revenue arrays
    revenue = adjusted.get('revenue', {})
    costs = adjusted.get('costs', {})
    profit = adjusted.get('profit', {})
    
    # Get baseline arrays (convert to numpy for easier manipulation)
    revenue_total = np.array(revenue.get('total', []))
    revenue_consumables = np.array(revenue.get('consumables', []))
    revenue_refurb = np.array(revenue.get('refurb', []))
    revenue_pipeline = np.array(revenue.get('pipeline', []))
    
    cogs = np.array(costs.get('cogs', []))
    opex = np.array(costs.get('opex', []))
    
    # Apply revenue adjustments
    revenue_multiplier = 1.0 + adjustments.get('revenue_pct', 0.0)
    utilization_multiplier = 1.0 + adjustments.get('utilization_pct', 0.0)
    
    # Revenue is affected by both revenue adjustment and utilization
    combined_revenue_mult = revenue_multiplier * utilization_multiplier
    
    revenue_consumables = revenue_consumables * combined_revenue_mult
    revenue_refurb = revenue_refurb * combined_revenue_mult
    revenue_pipeline = revenue_pipeline * combined_revenue_mult
    revenue_total = revenue_total * combined_revenue_mult
    
    # Update revenue in adjusted forecast
    revenue['consumables'] = revenue_consumables.tolist()
    revenue['refurb'] = revenue_refurb.tolist()
    revenue['pipeline'] = revenue_pipeline.tolist()
    revenue['total'] = revenue_total.tolist()
    
    # Apply COGS adjustment
    # COGS typically scales with revenue, but can be adjusted independently
    cogs_multiplier = 1.0 + adjustments.get('cogs_pct', 0.0)
    # Apply both revenue impact (COGS scales with revenue) and independent adjustment
    cogs = cogs * combined_revenue_mult * cogs_multiplier
    
    # Update COGS in adjusted forecast
    costs['cogs'] = cogs.tolist()
    
    # Apply OPEX adjustment (independent of revenue)
    opex_multiplier = 1.0 + adjustments.get('opex_pct', 0.0)
    opex = opex * opex_multiplier
    
    # Update OPEX in adjusted forecast
    costs['opex'] = opex.tolist()
    
    # Recalculate gross profit
    gross_profit = revenue_total - cogs
    profit['gross'] = gross_profit.tolist()
    
    # Recalculate EBIT (gross profit - OPEX)
    ebit = gross_profit - opex
    profit['ebit'] = ebit.tolist()
    
    # Recalculate summary statistics
    if 'summary' in adjusted:
        summary = adjusted['summary']
        summary['total_revenue'] = float(revenue_total.sum())
        summary['total_cogs'] = float(cogs.sum())
        summary['total_opex'] = float(opex.sum())
        summary['total_gross_profit'] = float(gross_profit.sum())
        summary['total_ebit'] = float(ebit.sum())
        summary['gross_margin_pct'] = float((gross_profit.sum() / revenue_total.sum() * 100) if revenue_total.sum() > 0 else 0)
        summary['ebit_margin_pct'] = float((ebit.sum() / revenue_total.sum() * 100) if revenue_total.sum() > 0 else 0)
    
    return adjusted


def display_whatif_results(
    baseline: Dict[str, Any],
    adjusted: Dict[str, Any],
    adjustments: Dict[str, float]
):
    """
    Display side-by-side comparison of baseline vs adjusted forecast.
    
    Args:
        baseline: Baseline forecast results
        adjusted: Adjusted forecast results
        adjustments: Dictionary of adjustments applied
    """
    st.markdown("---")
    st.markdown("### 📊 What-If Analysis Results")
    
    # Key Metrics Comparison
    baseline_summary = baseline.get('summary', {})
    adjusted_summary = adjusted.get('summary', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        baseline_rev = baseline_summary.get('total_revenue', 0)
        adjusted_rev = adjusted_summary.get('total_revenue', 0)
        rev_change = adjusted_rev - baseline_rev
        rev_change_pct = (rev_change / baseline_rev * 100) if baseline_rev > 0 else 0
        
        st.metric(
            "Total Revenue",
            f"R {adjusted_rev:,.0f}M",
            delta=f"{rev_change_pct:+.1f}%"
        )
    
    with col2:
        baseline_ebit = baseline_summary.get('total_ebit', 0)
        adjusted_ebit = adjusted_summary.get('total_ebit', 0)
        ebit_change = adjusted_ebit - baseline_ebit
        ebit_change_pct = (ebit_change / baseline_ebit * 100) if baseline_ebit != 0 else 0
        
        st.metric(
            "Total EBIT",
            f"R {adjusted_ebit:,.0f}M",
            delta=f"{ebit_change_pct:+.1f}%"
        )
    
    with col3:
        baseline_margin = baseline_summary.get('ebit_margin_pct', 0)
        adjusted_margin = adjusted_summary.get('ebit_margin_pct', 0)
        margin_change = adjusted_margin - baseline_margin
        
        st.metric(
            "EBIT Margin",
            f"{adjusted_margin:.1f}%",
            delta=f"{margin_change:+.1f}pp"
        )
    
    with col4:
        baseline_gp = baseline_summary.get('total_gross_profit', 0)
        adjusted_gp = adjusted_summary.get('total_gross_profit', 0)
        gp_change = adjusted_gp - baseline_gp
        gp_change_pct = (gp_change / baseline_gp * 100) if baseline_gp > 0 else 0
        
        st.metric(
            "Gross Profit",
            f"R {adjusted_gp:,.0f}M",
            delta=f"{gp_change_pct:+.1f}%"
        )
    
    # Comparison Chart
    st.markdown("---")
    st.markdown("#### 📈 Baseline vs Adjusted Comparison")
    
    timeline = baseline.get('timeline', [])
    if timeline:
        fig = create_comparison_chart(baseline, adjusted, timeline)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Comparison Table
    st.markdown("---")
    st.markdown("#### 📋 Detailed Comparison")
    
    comparison_data = create_comparison_table(baseline, adjusted)
    if comparison_data is not None and not comparison_data.empty:
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)


def create_comparison_chart(
    baseline: Dict[str, Any],
    adjusted: Dict[str, Any],
    timeline: List[str]
) -> go.Figure:
    """Create comparison chart showing baseline vs adjusted."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue', 'EBIT', 'Gross Profit', 'EBIT Margin'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Revenue
    baseline_rev = baseline.get('revenue', {}).get('total', [])
    adjusted_rev = adjusted.get('revenue', {}).get('total', [])
    
    fig.add_trace(go.Scatter(
        x=timeline, y=baseline_rev,
        name='Baseline Revenue',
        line=dict(color='#3b82f6', dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=timeline, y=adjusted_rev,
        name='Adjusted Revenue',
        line=dict(color='#10b981', width=2)
    ), row=1, col=1)
    
    # EBIT
    baseline_ebit = baseline.get('profit', {}).get('ebit', [])
    adjusted_ebit = adjusted.get('profit', {}).get('ebit', [])
    
    fig.add_trace(go.Scatter(
        x=timeline, y=baseline_ebit,
        name='Baseline EBIT',
        line=dict(color='#3b82f6', dash='dash'),
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=timeline, y=adjusted_ebit,
        name='Adjusted EBIT',
        line=dict(color='#10b981', width=2),
        showlegend=False
    ), row=1, col=2)
    
    # Gross Profit
    baseline_gp = baseline.get('profit', {}).get('gross', [])
    adjusted_gp = adjusted.get('profit', {}).get('gross', [])
    
    fig.add_trace(go.Scatter(
        x=timeline, y=baseline_gp,
        name='Baseline GP',
        line=dict(color='#3b82f6', dash='dash'),
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=timeline, y=adjusted_gp,
        name='Adjusted GP',
        line=dict(color='#10b981', width=2),
        showlegend=False
    ), row=2, col=1)
    
    # EBIT Margin
    baseline_rev_arr = np.array(baseline_rev)
    adjusted_rev_arr = np.array(adjusted_rev)
    baseline_ebit_arr = np.array(baseline_ebit)
    adjusted_ebit_arr = np.array(adjusted_ebit)
    
    baseline_margin = (baseline_ebit_arr / baseline_rev_arr * 100) if baseline_rev_arr.sum() > 0 else np.zeros_like(baseline_ebit_arr)
    adjusted_margin = (adjusted_ebit_arr / adjusted_rev_arr * 100) if adjusted_rev_arr.sum() > 0 else np.zeros_like(adjusted_ebit_arr)
    
    fig.add_trace(go.Scatter(
        x=timeline, y=baseline_margin,
        name='Baseline Margin',
        line=dict(color='#3b82f6', dash='dash'),
        showlegend=False
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=timeline, y=adjusted_margin,
        name='Adjusted Margin',
        line=dict(color='#10b981', width=2),
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        height=600,
        title_text="Baseline vs Adjusted Forecast Comparison",
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def create_comparison_table(
    baseline: Dict[str, Any],
    adjusted: Dict[str, Any]
) -> pd.DataFrame:
    """Create detailed comparison table."""
    baseline_summary = baseline.get('summary', {})
    adjusted_summary = adjusted.get('summary', {})
    
    metrics = [
        ('Total Revenue', 'total_revenue', 'M'),
        ('Total COGS', 'total_cogs', 'M'),
        ('Total OPEX', 'total_opex', 'M'),
        ('Gross Profit', 'total_gross_profit', 'M'),
        ('EBIT', 'total_ebit', 'M'),
        ('Gross Margin', 'gross_margin_pct', '%'),
        ('EBIT Margin', 'ebit_margin_pct', '%'),
    ]
    
    rows = []
    for label, key, unit in metrics:
        baseline_val = baseline_summary.get(key, 0)
        adjusted_val = adjusted_summary.get(key, 0)
        change = adjusted_val - baseline_val
        change_pct = (change / baseline_val * 100) if baseline_val != 0 else 0
        
        if unit == 'M':
            baseline_str = f"R {baseline_val:,.0f}M"
            adjusted_str = f"R {adjusted_val:,.0f}M"
            change_str = f"R {change:,.0f}M ({change_pct:+.1f}%)"
        else:
            baseline_str = f"{baseline_val:.1f}%"
            adjusted_str = f"{adjusted_val:.1f}%"
            change_str = f"{change:+.1f}pp"
        
        rows.append({
            'Metric': label,
            'Baseline': baseline_str,
            'Adjusted': adjusted_str,
            'Change': change_str
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# SENSITIVITY ANALYSIS (Sprint 18)
# =============================================================================

def run_sensitivity_analysis(
    baseline: Dict[str, Any],
    parameter_ranges: Dict[str, tuple] = None
) -> Dict[str, Any]:
    """
    Run sensitivity analysis on key parameters.
    
    Args:
        baseline: Baseline forecast results
        parameter_ranges: Optional custom ranges for parameters
            Format: {'revenue_pct': (-0.2, 0.2), 'cogs_pct': (-0.15, 0.15)}
    
    Returns:
        Dictionary with sensitivity results
    """
    if not baseline:
        return None
    
    # Default parameter ranges (±20% for revenue, ±15% for costs)
    if parameter_ranges is None:
        parameter_ranges = {
            'revenue_pct': (-0.20, 0.20),
            'utilization_pct': (-0.15, 0.15),
            'cogs_pct': (-0.15, 0.15),
            'opex_pct': (-0.20, 0.20)
        }
    
    # Get baseline summary for comparison
    baseline_summary = baseline.get('summary', {})
    baseline_ebit = baseline_summary.get('total_ebit', 0)
    baseline_revenue = baseline_summary.get('total_revenue', 0)
    
    sensitivity_results = {
        'parameters': [],
        'ebit_impact': [],
        'revenue_impact': [],
        'margin_impact': []
    }
    
    # Test each parameter independently
    for param_name, (low, high) in parameter_ranges.items():
        # Test low value
        low_adjustments = {param_name: low}
        low_adjusted = calculate_adjusted_forecast(baseline, low_adjustments)
        if low_adjusted:
            low_summary = low_adjusted.get('summary', {})
            low_ebit = low_summary.get('total_ebit', 0)
            low_revenue = low_summary.get('total_revenue', 0)
            low_ebit_change = low_ebit - baseline_ebit
            low_revenue_change = low_revenue - baseline_revenue
        else:
            low_ebit_change = 0
            low_revenue_change = 0
        
        # Test high value
        high_adjustments = {param_name: high}
        high_adjusted = calculate_adjusted_forecast(baseline, high_adjustments)
        if high_adjusted:
            high_summary = high_adjusted.get('summary', {})
            high_ebit = high_summary.get('total_ebit', 0)
            high_revenue = high_summary.get('total_revenue', 0)
            high_ebit_change = high_ebit - baseline_ebit
            high_revenue_change = high_revenue - baseline_revenue
        else:
            high_ebit_change = 0
            high_revenue_change = 0
        
        # Calculate range (high - low)
        ebit_range = high_ebit_change - low_ebit_change
        revenue_range = high_revenue_change - low_revenue_change
        
        sensitivity_results['parameters'].append(param_name.replace('_pct', '').title())
        sensitivity_results['ebit_impact'].append(ebit_range)
        sensitivity_results['revenue_impact'].append(revenue_range)
        
        # Calculate margin impact
        baseline_margin = baseline_summary.get('ebit_margin_pct', 0)
        if high_adjusted:
            high_margin = high_adjusted.get('summary', {}).get('ebit_margin_pct', 0)
        else:
            high_margin = baseline_margin
        if low_adjusted:
            low_margin = low_adjusted.get('summary', {}).get('ebit_margin_pct', 0)
        else:
            low_margin = baseline_margin
        
        margin_range = high_margin - low_margin
        sensitivity_results['margin_impact'].append(margin_range)
    
    return sensitivity_results


def display_sensitivity_analysis(results: Dict[str, Any]):
    """Display sensitivity analysis results with tornado diagram."""
    st.markdown("#### 📊 Parameter Sensitivity Ranking")
    
    # Create DataFrame for easier display
    df = pd.DataFrame({
        'Parameter': results['parameters'],
        'EBIT Impact (Range)': [f"R {x:,.0f}M" for x in results['ebit_impact']],
        'Revenue Impact (Range)': [f"R {x:,.0f}M" for x in results['revenue_impact']],
        'Margin Impact (pp)': [f"{x:+.1f}" for x in results['margin_impact']]
    })
    
    # Sort by EBIT impact (absolute value)
    df['EBIT_Impact_Abs'] = [abs(x) for x in results['ebit_impact']]
    df = df.sort_values('EBIT_Impact_Abs', ascending=False)
    df = df.drop('EBIT_Impact_Abs', axis=1)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Tornado Diagram
    st.markdown("#### 🌪️ Tornado Diagram - EBIT Sensitivity")
    
    fig = create_tornado_diagram(results)
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown("#### 💡 Key Insights")
    
    # Find most sensitive parameter
    max_impact_idx = np.argmax([abs(x) for x in results['ebit_impact']])
    most_sensitive = results['parameters'][max_impact_idx]
    max_impact = results['ebit_impact'][max_impact_idx]
    
    st.info(f"""
    **Most Sensitive Parameter:** {most_sensitive}
    
    A ±20% change in {most_sensitive} results in a **R {abs(max_impact):,.0f}M** 
    change in EBIT, making it the highest impact driver of profitability.
    
    **Recommendation:** Focus on accurate forecasting and monitoring of {most_sensitive} 
    as small changes have significant financial impact.
    """)


def create_tornado_diagram(results: Dict[str, Any]) -> go.Figure:
    """Create tornado diagram showing parameter sensitivity."""
    parameters = results['parameters']
    ebit_impacts = results['ebit_impact']
    
    # Sort by absolute impact
    sorted_data = sorted(
        zip(parameters, ebit_impacts),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    sorted_params = [p for p, _ in sorted_data]
    sorted_impacts = [i for _, i in sorted_data]
    
    # Create horizontal bar chart
    colors = ['#ef4444' if x < 0 else '#10b981' for x in sorted_impacts]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sorted_params,
        x=sorted_impacts,
        orientation='h',
        marker_color=colors,
        text=[f"R {abs(x):,.0f}M" for x in sorted_impacts],
        textposition='outside',
        name='EBIT Impact'
    ))
    
    fig.update_layout(
        title="Parameter Sensitivity - EBIT Impact Range",
        xaxis_title="EBIT Impact (R Million)",
        yaxis_title="Parameter",
        height=400,
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
    )
    
    return fig
