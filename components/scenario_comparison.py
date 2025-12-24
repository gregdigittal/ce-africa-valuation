"""
Scenario Comparison for Forecast Methods
=========================================
Allows running both Pipeline and Trend-based forecasts and comparing
results side-by-side.

Phase 4 of Unified Configuration Backlog.
Date: December 20, 2025

Features:
- Run both forecast methods
- Side-by-side metric comparison
- Variance analysis
- Chart overlay
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta


# =============================================================================
# COMPARISON DATA STRUCTURES
# =============================================================================

class ForecastComparison:
    """Holds comparison results between two forecast methods."""
    
    def __init__(
        self,
        pipeline_results: Optional[Dict] = None,
        trend_results: Optional[Dict] = None
    ):
        self.pipeline = pipeline_results
        self.trend = trend_results
        self.timeline = []
        
        if pipeline_results and pipeline_results.get('timeline'):
            self.timeline = pipeline_results['timeline']
        elif trend_results and trend_results.get('timeline'):
            self.timeline = trend_results['timeline']
    
    @property
    def has_both(self) -> bool:
        return self.pipeline is not None and self.trend is not None
    
    def get_metric(self, method: str, metric: str) -> float:
        """Get summary metric from specified method."""
        results = self.pipeline if method == 'pipeline' else self.trend
        if not results:
            return 0.0
        summary = results.get('summary', {})
        return summary.get(metric, 0.0)
    
    def get_monthly(self, method: str, metric: str) -> np.ndarray:
        """Get monthly data from specified method."""
        results = self.pipeline if method == 'pipeline' else self.trend
        if not results:
            return np.array([])
        monthly = results.get('monthly_data', {})
        data = monthly.get(metric, [])
        return np.array(data)
    
    def get_variance(self, metric: str) -> Tuple[float, float]:
        """Get absolute and percentage variance for a metric."""
        pipeline_val = self.get_metric('pipeline', metric)
        trend_val = self.get_metric('trend', metric)
        
        abs_var = trend_val - pipeline_val
        pct_var = (abs_var / pipeline_val * 100) if pipeline_val != 0 else 0.0
        
        return abs_var, pct_var


# =============================================================================
# RUN COMPARISON
# =============================================================================

def run_forecast_comparison(
    db,
    scenario_id: str,
    user_id: str,
    data: Dict[str, Any],
    progress_callback: Optional[callable] = None
) -> ForecastComparison:
    """
    Run both forecast methods and return comparison.
    
    Args:
        db: Database connector
        scenario_id: Scenario ID
        user_id: User ID
        data: Forecast input data (machines, prospects, expenses, etc.)
        progress_callback: Optional progress callback
        
    Returns:
        ForecastComparison with both results
    """
    from forecast_engine import ForecastEngine
    
    comparison = ForecastComparison()
    engine = ForecastEngine()
    
    # Make copy of data with method set to pipeline
    pipeline_data = data.copy()
    if 'assumptions' not in pipeline_data:
        pipeline_data['assumptions'] = {}
    pipeline_data['assumptions'] = pipeline_data['assumptions'].copy()
    pipeline_data['assumptions']['forecast_method'] = 'pipeline'
    pipeline_data['assumptions']['use_trend_forecast'] = False
    
    # Run pipeline forecast
    if progress_callback:
        progress_callback(0.1, "Running Pipeline-Based forecast...")
    
    try:
        pipeline_results = engine.run_forecast(
            pipeline_data,
            progress_callback=lambda p, m: progress_callback(0.1 + p * 0.35, f"[Pipeline] {m}") if progress_callback else None
        )
        if pipeline_results.get('success'):
            comparison.pipeline = pipeline_results
    except Exception as e:
        st.warning(f"Pipeline forecast failed: {e}")
    
    # Make copy of data with method set to trend
    trend_data = data.copy()
    if 'assumptions' not in trend_data:
        trend_data['assumptions'] = {}
    trend_data['assumptions'] = trend_data['assumptions'].copy()
    trend_data['assumptions']['forecast_method'] = 'trend'
    trend_data['assumptions']['use_trend_forecast'] = True
    
    # Run trend forecast
    if progress_callback:
        progress_callback(0.5, "Running Trend-Based forecast...")
    
    try:
        trend_results = engine.run_forecast(
            trend_data,
            progress_callback=lambda p, m: progress_callback(0.5 + p * 0.45, f"[Trend] {m}") if progress_callback else None
        )
        if trend_results.get('success'):
            comparison.trend = trend_results
    except Exception as e:
        st.warning(f"Trend forecast failed: {e}")
    
    if progress_callback:
        progress_callback(1.0, "Comparison complete")
    
    return comparison


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comparison_chart(
    comparison: ForecastComparison,
    metric: str,
    title: str
) -> go.Figure:
    """Create overlay chart comparing both methods."""
    fig = go.Figure()
    
    # Pipeline data
    pipeline_data = comparison.get_monthly('pipeline', metric)
    if len(pipeline_data) > 0:
        fig.add_trace(go.Scatter(
            x=comparison.timeline,
            y=pipeline_data,
            name='Pipeline-Based',
            mode='lines',
            line=dict(color='#2196F3', width=2),
        ))
    
    # Trend data
    trend_data = comparison.get_monthly('trend', metric)
    if len(trend_data) > 0:
        fig.add_trace(go.Scatter(
            x=comparison.timeline,
            y=trend_data,
            name='Trend-Based',
            mode='lines',
            line=dict(color='#FF9800', width=2, dash='dash'),
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Period',
        yaxis_title='Value (R)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=350,
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )
    
    # Format y-axis
    fig.update_yaxes(tickformat=',.0f')
    
    return fig


def create_variance_waterfall(
    comparison: ForecastComparison,
    metrics: List[str],
    metric_names: List[str]
) -> go.Figure:
    """Create waterfall chart showing variances."""
    variances = []
    colors = []
    
    for metric in metrics:
        abs_var, _ = comparison.get_variance(metric)
        variances.append(abs_var)
        colors.append('#4CAF50' if abs_var >= 0 else '#F44336')
    
    fig = go.Figure(go.Bar(
        x=metric_names,
        y=variances,
        marker_color=colors,
        text=[f"R {v:,.0f}" for v in variances],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Variance: Trend vs Pipeline',
        xaxis_title='Metric',
        yaxis_title='Variance (R)',
        height=350,
        margin=dict(l=50, r=20, t=60, b=80)
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    return fig


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_comparison_tab(db, scenario_id: str, user_id: str):
    """Render the scenario comparison tab."""
    st.markdown("### 🔄 Forecast Method Comparison")
    st.caption("Run both Pipeline and Trend-based forecasts and compare results")
    
    # Check if we have a comparison in session state
    comparison_key = f'forecast_comparison_{scenario_id}'
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        run_clicked = st.button(
            "▶️ Run Comparison",
            type="primary",
            use_container_width=True,
            help="Run both forecast methods and compare"
        )
    
    with col1:
        st.info("""
This runs **both** forecast methods:
- **Pipeline-Based:** Fleet + Prospects revenue model
- **Trend-Based:** Historical trend projection
        """)
    
    if run_clicked:
        # Load data for forecast
        from components.forecast_section import load_forecast_data
        try:
            data = load_forecast_data(db, scenario_id, user_id)
        except Exception as e:
            st.error(f"Failed to load forecast data: {e}")
            return
        
        # Run comparison
        progress = st.progress(0, text="Starting comparison...")
        
        def update_progress(pct, msg):
            progress.progress(min(pct, 1.0), text=msg)
        
        comparison = run_forecast_comparison(
            db, scenario_id, user_id, data,
            progress_callback=update_progress
        )
        
        progress.empty()
        
        # Store in session state
        st.session_state[comparison_key] = comparison
    
    # Display comparison if available
    comparison = st.session_state.get(comparison_key)
    
    if not comparison:
        st.info("Click 'Run Comparison' to compare forecast methods.")
        return
    
    if not comparison.has_both:
        st.warning("Could not run both forecasts. Check configuration for each method.")
        
        if comparison.pipeline:
            st.success("✅ Pipeline forecast succeeded")
        else:
            st.error("❌ Pipeline forecast failed")
        
        if comparison.trend:
            st.success("✅ Trend forecast succeeded")
        else:
            st.error("❌ Trend forecast failed - Configure line items in AI Assumptions → Configure Assumptions")
        
        return
    
    st.success("✅ Both forecasts completed successfully")
    
    st.markdown("---")
    
    # ==========================================================================
    # SUMMARY COMPARISON
    # ==========================================================================
    st.markdown("#### 📊 Summary Comparison")
    
    metrics = [
        ('total_revenue', 'Total Revenue'),
        ('total_cogs', 'COGS'),
        ('total_gross_profit', 'Gross Profit'),
        ('total_opex', 'OPEX'),
        ('total_ebit', 'EBIT'),
    ]
    
    # Create comparison table
    comparison_data = []
    for metric_key, metric_name in metrics:
        pipeline_val = comparison.get_metric('pipeline', metric_key)
        trend_val = comparison.get_metric('trend', metric_key)
        abs_var, pct_var = comparison.get_variance(metric_key)
        
        comparison_data.append({
            'Metric': metric_name,
            'Pipeline': pipeline_val,
            'Trend': trend_val,
            'Variance': abs_var,
            'Var %': pct_var
        })
    
    df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        df.style.format({
            'Pipeline': 'R {:,.0f}',
            'Trend': 'R {:,.0f}',
            'Variance': 'R {:,.0f}',
            'Var %': '{:+.1f}%'
        }).applymap(
            lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else ('color: red' if isinstance(x, (int, float)) and x < 0 else ''),
            subset=['Variance', 'Var %']
        ),
        hide_index=True,
        use_container_width=True
    )
    
    st.markdown("---")
    
    # ==========================================================================
    # CHARTS
    # ==========================================================================
    st.markdown("#### 📈 Trend Comparison Charts")
    
    chart_metric = st.selectbox(
        "Select Metric",
        options=['revenue', 'cogs', 'gross_profit', 'opex', 'ebit', 'net_income'],
        format_func=lambda x: x.replace('_', ' ').title(),
        key='comparison_chart_metric'
    )
    
    col1, col2 = st.columns(2)
    with col1:
        fig = create_comparison_chart(
            comparison,
            chart_metric,
            f"{chart_metric.replace('_', ' ').title()} Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Variance waterfall
        fig2 = create_variance_waterfall(
            comparison,
            [m[0] for m in metrics],
            [m[1] for m in metrics]
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # ==========================================================================
    # RECOMMENDATIONS
    # ==========================================================================
    st.markdown("---")
    st.markdown("#### 💡 Recommendations")
    
    rev_var, rev_pct = comparison.get_variance('total_revenue')
    ebit_var, ebit_pct = comparison.get_variance('total_ebit')
    
    if abs(rev_pct) < 10:
        st.success("✅ Revenue forecasts are aligned (within 10%). Either method should produce similar results.")
    elif rev_pct > 0:
        st.info(f"📈 Trend-based forecast is {rev_pct:.1f}% higher than Pipeline. Consider if historical growth trends will continue.")
    else:
        st.warning(f"📉 Trend-based forecast is {abs(rev_pct):.1f}% lower than Pipeline. Pipeline may be more optimistic about new sales.")
    
    if abs(ebit_pct) > 20:
        st.warning(f"⚠️ EBIT variance is significant ({ebit_pct:+.1f}%). Review cost assumptions for consistency.")
    
    # Method recommendation
    st.markdown("**Suggested Method:**")
    
    pipeline_has_fleet = comparison.pipeline.get('summary', {}).get('fleet_count', 0) > 0
    trend_has_items = bool(comparison.trend.get('line_items'))
    
    if pipeline_has_fleet and trend_has_items:
        st.info("""
Both methods are well-configured. Consider:
- **Pipeline** if you have strong fleet data and active sales pipeline
- **Trend** if historical financials are reliable predictors of future performance
- For valuation, you might run **Monte Carlo on both** and compare confidence intervals
        """)
    elif pipeline_has_fleet:
        st.info("**Recommend Pipeline-Based** - You have fleet data but limited line-item configuration for trends.")
    elif trend_has_items:
        st.info("**Recommend Trend-Based** - Line items are configured but fleet/pipeline data may be limited.")
    else:
        st.warning("Both methods need more configuration. Complete setup in AI Assumptions and ensure machine data is loaded.")


def format_currency(value: float) -> str:
    """Format value as currency."""
    if abs(value) >= 1_000_000:
        return f"R {value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"R {value/1_000:.0f}K"
    else:
        return f"R {value:.0f}"
