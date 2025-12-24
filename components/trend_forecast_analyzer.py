"""
Trend Forecast Analyzer
======================
Analyzes historical financial data and provides trend-based forecasting with:
1. Distribution function fitting to historical trends
2. Parameter adjustment with visual preview
3. Integration into forecast engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json


class TrendFunction(Enum):
    """Available trend functions for forecasting."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    LOGARITHMIC = "logarithmic"
    POWER = "power"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"


@dataclass
class TrendParams:
    """Parameters for a trend function."""
    function_type: TrendFunction
    params: Dict[str, float]
    fit_score: float = 0.0
    r_squared: float = 0.0
    description: str = ""


class TrendForecastAnalyzer:
    """Analyzes historical trends and generates forecast functions."""
    
    def __init__(self):
        """Initialize the trend analyzer."""
        pass
    
    def fit_trend_function(
        self,
        historical_data: pd.Series,
        function_type: TrendFunction,
        forecast_periods: int = 60
    ) -> Tuple[TrendParams, np.ndarray]:
        """
        Fit a trend function to historical data.
        
        Args:
            historical_data: Series with historical values (index should be time-based)
            function_type: Type of trend function to fit
            forecast_periods: Number of periods to forecast ahead
        
        Returns:
            Tuple of (TrendParams, forecast_array)
        """
        if len(historical_data) < 3:
            # Not enough data - return flat forecast
            last_value = historical_data.iloc[-1] if len(historical_data) > 0 else 0
            params = TrendParams(
                function_type=function_type,
                params={'constant': last_value},
                fit_score=0.0,
                r_squared=0.0,
                description="Insufficient data"
            )
            forecast = np.full(forecast_periods, last_value)
            return params, forecast
        
        # Prepare data
        y = historical_data.values
        x = np.arange(len(y))
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(y_valid) < 3:
            last_value = y_valid[-1] if len(y_valid) > 0 else 0
            params = TrendParams(
                function_type=function_type,
                params={'constant': last_value},
                fit_score=0.0,
                r_squared=0.0,
                description="Insufficient valid data"
            )
            forecast = np.full(forecast_periods, last_value)
            return params, forecast
        
        # Fit function based on type
        try:
            if function_type == TrendFunction.LINEAR:
                params, forecast = self._fit_linear(x_valid, y_valid, forecast_periods)
            elif function_type == TrendFunction.EXPONENTIAL:
                params, forecast = self._fit_exponential(x_valid, y_valid, forecast_periods)
            elif function_type == TrendFunction.POLYNOMIAL:
                params, forecast = self._fit_polynomial(x_valid, y_valid, forecast_periods)
            elif function_type == TrendFunction.LOGARITHMIC:
                params, forecast = self._fit_logarithmic(x_valid, y_valid, forecast_periods)
            elif function_type == TrendFunction.POWER:
                params, forecast = self._fit_power(x_valid, y_valid, forecast_periods)
            elif function_type == TrendFunction.MOVING_AVERAGE:
                params, forecast = self._fit_moving_average(x_valid, y_valid, forecast_periods)
            elif function_type == TrendFunction.EXPONENTIAL_SMOOTHING:
                params, forecast = self._fit_exponential_smoothing(x_valid, y_valid, forecast_periods)
            else:
                # Default to linear
                params, forecast = self._fit_linear(x_valid, y_valid, forecast_periods)
            
            params.function_type = function_type
            return params, forecast
            
        except Exception as e:
            # Fallback to simple average
            last_value = y_valid[-1]
            avg_value = np.mean(y_valid)
            params = TrendParams(
                function_type=function_type,
                params={'constant': avg_value, 'trend': 0},
                fit_score=0.0,
                r_squared=0.0,
                description=f"Fit failed: {str(e)}"
            )
            forecast = np.full(forecast_periods, last_value)
            return params, forecast
    
    def _fit_linear(self, x: np.ndarray, y: np.ndarray, forecast_periods: int) -> Tuple[TrendParams, np.ndarray]:
        """Fit linear trend: y = a*x + b"""
        coeffs = np.polyfit(x, y, 1)
        a, b = coeffs[0], coeffs[1]
        
        # Calculate R-squared
        y_pred = a * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        params = TrendParams(
            function_type=TrendFunction.LINEAR,
            params={'slope': float(a), 'intercept': float(b)},
            fit_score=r_squared,
            r_squared=r_squared,
            description=f"Linear: y = {a:.2f}*x + {b:.2f}"
        )
        
        # Forecast
        x_forecast = np.arange(len(x), len(x) + forecast_periods)
        forecast = a * x_forecast + b
        
        return params, forecast
    
    def _fit_exponential(self, x: np.ndarray, y: np.ndarray, forecast_periods: int) -> Tuple[TrendParams, np.ndarray]:
        """Fit exponential trend: y = a * exp(b*x)"""
        # Ensure positive values for exponential
        y_positive = np.maximum(y, 0.01)
        log_y = np.log(y_positive)
        
        coeffs = np.polyfit(x, log_y, 1)
        b, log_a = coeffs[0], coeffs[1]
        a = np.exp(log_a)
        
        # Calculate R-squared
        y_pred = a * np.exp(b * x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        params = TrendParams(
            function_type=TrendFunction.EXPONENTIAL,
            params={'base': float(a), 'growth_rate': float(b)},
            fit_score=r_squared,
            r_squared=r_squared,
            description=f"Exponential: y = {a:.2f} * exp({b:.4f}*x)"
        )
        
        # Forecast
        x_forecast = np.arange(len(x), len(x) + forecast_periods)
        forecast = a * np.exp(b * x_forecast)
        
        return params, forecast
    
    def _fit_polynomial(self, x: np.ndarray, y: np.ndarray, forecast_periods: int, degree: int = 2) -> Tuple[TrendParams, np.ndarray]:
        """Fit polynomial trend: y = a*x^2 + b*x + c (or higher degree)"""
        max_degree = min(degree, len(x) - 1)
        coeffs = np.polyfit(x, y, max_degree)
        
        # Calculate R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        params = TrendParams(
            function_type=TrendFunction.POLYNOMIAL,
            params={'coefficients': coeffs.tolist(), 'degree': max_degree},
            fit_score=r_squared,
            r_squared=r_squared,
            description=f"Polynomial (degree {max_degree})"
        )
        
        # Forecast
        x_forecast = np.arange(len(x), len(x) + forecast_periods)
        forecast = np.polyval(coeffs, x_forecast)
        
        return params, forecast
    
    def _fit_logarithmic(self, x: np.ndarray, y: np.ndarray, forecast_periods: int) -> Tuple[TrendParams, np.ndarray]:
        """Fit logarithmic trend: y = a * ln(x + 1) + b"""
        x_shifted = x + 1  # Avoid log(0)
        log_x = np.log(x_shifted)
        
        coeffs = np.polyfit(log_x, y, 1)
        a, b = coeffs[0], coeffs[1]
        
        # Calculate R-squared
        y_pred = a * np.log(x_shifted) + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        params = TrendParams(
            function_type=TrendFunction.LOGARITHMIC,
            params={'coefficient': float(a), 'intercept': float(b)},
            fit_score=r_squared,
            r_squared=r_squared,
            description=f"Logarithmic: y = {a:.2f} * ln(x+1) + {b:.2f}"
        )
        
        # Forecast
        x_forecast = np.arange(len(x), len(x) + forecast_periods)
        forecast = a * np.log(x_forecast + 1) + b
        
        return params, forecast
    
    def _fit_power(self, x: np.ndarray, y: np.ndarray, forecast_periods: int) -> Tuple[TrendParams, np.ndarray]:
        """Fit power trend: y = a * x^b"""
        # Ensure positive values
        x_positive = np.maximum(x, 0.01)
        y_positive = np.maximum(y, 0.01)
        
        log_x = np.log(x_positive)
        log_y = np.log(y_positive)
        
        coeffs = np.polyfit(log_x, log_y, 1)
        b, log_a = coeffs[0], coeffs[1]
        a = np.exp(log_a)
        
        # Calculate R-squared
        y_pred = a * (x_positive ** b)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        params = TrendParams(
            function_type=TrendFunction.POWER,
            params={'coefficient': float(a), 'exponent': float(b)},
            fit_score=r_squared,
            r_squared=r_squared,
            description=f"Power: y = {a:.2f} * x^{b:.4f}"
        )
        
        # Forecast
        x_forecast = np.arange(len(x), len(x) + forecast_periods)
        forecast = a * (np.maximum(x_forecast, 0.01) ** b)
        
        return params, forecast
    
    def _fit_moving_average(self, x: np.ndarray, y: np.ndarray, forecast_periods: int, window: int = 3) -> Tuple[TrendParams, np.ndarray]:
        """Fit moving average trend."""
        window = min(window, len(y))
        ma_values = []
        
        for i in range(len(y)):
            start = max(0, i - window + 1)
            ma_values.append(np.mean(y[start:i+1]))
        
        # Calculate trend from moving average
        ma_array = np.array(ma_values)
        trend = (ma_array[-1] - ma_array[0]) / len(ma_array) if len(ma_array) > 1 else 0
        
        # Calculate R-squared
        y_pred = ma_array
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        params = TrendParams(
            function_type=TrendFunction.MOVING_AVERAGE,
            params={'window': window, 'trend': float(trend), 'last_value': float(ma_array[-1])},
            fit_score=r_squared,
            r_squared=r_squared,
            description=f"Moving Average (window={window})"
        )
        
        # Forecast: extend with trend
        last_value = ma_array[-1]
        forecast = np.array([last_value + trend * (i + 1) for i in range(forecast_periods)])
        
        return params, forecast
    
    def _fit_exponential_smoothing(self, x: np.ndarray, y: np.ndarray, forecast_periods: int, alpha: float = 0.3) -> Tuple[TrendParams, np.ndarray]:
        """Fit exponential smoothing trend."""
        smoothed = [y[0]]
        for i in range(1, len(y)):
            smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[-1])
        
        smoothed_array = np.array(smoothed)
        trend = (smoothed_array[-1] - smoothed_array[0]) / len(smoothed_array) if len(smoothed_array) > 1 else 0
        
        # Calculate R-squared
        y_pred = smoothed_array
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        params = TrendParams(
            function_type=TrendFunction.EXPONENTIAL_SMOOTHING,
            params={'alpha': alpha, 'trend': float(trend), 'last_value': float(smoothed_array[-1])},
            fit_score=r_squared,
            r_squared=r_squared,
            description=f"Exponential Smoothing (α={alpha})"
        )
        
        # Forecast: extend with trend
        last_value = smoothed_array[-1]
        forecast = np.array([last_value + trend * (i + 1) for i in range(forecast_periods)])
        
        return params, forecast
    
    def generate_forecast_with_params(
        self,
        historical_data: pd.Series,
        function_type: TrendFunction,
        adjusted_params: Dict[str, float],
        forecast_periods: int = 60
    ) -> np.ndarray:
        """
        Generate forecast using adjusted parameters.
        
        Args:
            historical_data: Historical data series
            function_type: Type of trend function
            adjusted_params: User-adjusted parameters
            forecast_periods: Number of periods to forecast
        
        Returns:
            Forecast array
        """
        x = np.arange(len(historical_data))
        y = historical_data.values
        x_forecast = np.arange(len(historical_data), len(historical_data) + forecast_periods)
        
        if function_type == TrendFunction.LINEAR:
            slope = adjusted_params.get('slope', 0)
            intercept = adjusted_params.get('intercept', y[-1] if len(y) > 0 else 0)
            forecast = slope * x_forecast + intercept
        
        elif function_type == TrendFunction.EXPONENTIAL:
            base = adjusted_params.get('base', y[-1] if len(y) > 0 else 1)
            growth_rate = adjusted_params.get('growth_rate', 0)
            forecast = base * np.exp(growth_rate * x_forecast)
        
        elif function_type == TrendFunction.POLYNOMIAL:
            coeffs = adjusted_params.get('coefficients', [0, y[-1] if len(y) > 0 else 0])
            forecast = np.polyval(coeffs, x_forecast)
        
        elif function_type == TrendFunction.LOGARITHMIC:
            coefficient = adjusted_params.get('coefficient', 1)
            intercept = adjusted_params.get('intercept', y[-1] if len(y) > 0 else 0)
            forecast = coefficient * np.log(x_forecast + 1) + intercept
        
        elif function_type == TrendFunction.POWER:
            coefficient = adjusted_params.get('coefficient', y[-1] if len(y) > 0 else 1)
            exponent = adjusted_params.get('exponent', 1)
            forecast = coefficient * (np.maximum(x_forecast, 0.01) ** exponent)
        
        elif function_type == TrendFunction.MOVING_AVERAGE:
            last_value = adjusted_params.get('last_value', y[-1] if len(y) > 0 else 0)
            trend = adjusted_params.get('trend', 0)
            forecast = np.array([last_value + trend * (i + 1) for i in range(forecast_periods)])
        
        elif function_type == TrendFunction.EXPONENTIAL_SMOOTHING:
            last_value = adjusted_params.get('last_value', y[-1] if len(y) > 0 else 0)
            trend = adjusted_params.get('trend', 0)
            forecast = np.array([last_value + trend * (i + 1) for i in range(forecast_periods)])
        
        else:
            # Default: flat forecast
            last_value = y[-1] if len(y) > 0 else 0
            forecast = np.full(forecast_periods, last_value)
        
        return forecast


# =============================================================================
# UI RENDERING FUNCTIONS
# =============================================================================

def render_trend_forecast_ui(
    db,
    scenario_id: str,
    user_id: str,
    historical_data: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    Render UI for trend-based forecast analysis and parameter adjustment.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        user_id: User ID
        historical_data: DataFrame with historical financial data (must have 'month' or 'period_date' and financial columns)
    
    Returns:
        Dictionary with selected trend functions and parameters, or None if cancelled
    """
    st.markdown("### 📈 Trend-Based Forecast Analysis")
    st.caption("Analyze historical trends and configure forecast functions with visual preview")
    
    if historical_data.empty:
        st.warning("⚠️ No historical data available. Import historical financial statements first.")
        return None
    
    # Identify available metrics
    financial_metrics = []
    if 'revenue' in historical_data.columns:
        financial_metrics.append(('revenue', 'Revenue'))
    if 'cogs' in historical_data.columns:
        financial_metrics.append(('cogs', 'COGS'))
    if 'opex' in historical_data.columns:
        financial_metrics.append(('opex', 'Operating Expenses'))
    if 'gross_profit' in historical_data.columns:
        financial_metrics.append(('gross_profit', 'Gross Profit'))
    if 'ebit' in historical_data.columns:
        financial_metrics.append(('ebit', 'EBIT'))
    
    if not financial_metrics:
        st.warning("⚠️ No recognized financial metrics found in historical data.")
        return None
    
    # Metric selection
    selected_metric = st.selectbox(
        "Select Metric to Forecast",
        options=[m[0] for m in financial_metrics],
        format_func=lambda x: next(m[1] for m in financial_metrics if m[0] == x),
        key="trend_metric_select"
    )
    
    # Prepare data series
    if 'month' in historical_data.columns:
        data_series = historical_data.set_index('month')[selected_metric].sort_index()
    elif 'period_date' in historical_data.columns:
        data_series = historical_data.set_index('period_date')[selected_metric].sort_index()
    else:
        data_series = historical_data[selected_metric]
    
    # Remove NaN values
    data_series = data_series.dropna()
    
    if len(data_series) < 3:
        st.warning(f"⚠️ Insufficient data points ({len(data_series)}) for trend analysis. Need at least 3.")
        return None
    
    st.markdown("---")
    
    # Function selection
    st.markdown("#### Select Forecast Function")
    
    function_options = {
        TrendFunction.LINEAR: "Linear Growth",
        TrendFunction.EXPONENTIAL: "Exponential Growth",
        TrendFunction.POLYNOMIAL: "Polynomial (Curved)",
        TrendFunction.LOGARITHMIC: "Logarithmic (Diminishing)",
        TrendFunction.POWER: "Power Law",
        TrendFunction.MOVING_AVERAGE: "Moving Average",
        TrendFunction.EXPONENTIAL_SMOOTHING: "Exponential Smoothing"
    }
    
    selected_function_str = st.selectbox(
        "Forecast Function Type",
        options=list(function_options.keys()),
        format_func=lambda x: function_options[x],
        key="trend_function_select"
    )
    
    # Forecast periods
    forecast_periods = st.slider(
        "Forecast Periods (months)",
        min_value=12,
        max_value=120,
        value=60,
        step=12,
        key="trend_forecast_periods"
    )
    
    # Fit function
    analyzer = TrendForecastAnalyzer()
    trend_params, forecast_values = analyzer.fit_trend_function(
        data_series,
        selected_function_str,
        forecast_periods
    )
    
    # Display fit quality
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{trend_params.r_squared:.3f}")
    with col2:
        st.metric("Fit Score", f"{trend_params.fit_score:.3f}")
    with col3:
        st.metric("Function", trend_params.function_type.value)
    
    st.caption(f"📊 {trend_params.description}")
    
    st.markdown("---")
    
    # Parameter adjustment
    st.markdown("#### Adjust Parameters")
    st.caption("Modify parameters to see impact on forecast curve")
    
    adjusted_params = trend_params.params.copy()
    
    # Show parameter sliders based on function type
    if selected_function_str == TrendFunction.LINEAR:
        adjusted_params['slope'] = st.slider(
            "Slope (Growth Rate per Period)",
            min_value=float(adjusted_params.get('slope', 0)) - abs(adjusted_params.get('slope', 0)) * 2,
            max_value=float(adjusted_params.get('slope', 0)) + abs(adjusted_params.get('slope', 0)) * 2,
            value=float(adjusted_params.get('slope', 0)),
            step=float(adjusted_params.get('slope', 1)) / 100 if adjusted_params.get('slope', 0) != 0 else 1.0,
            key="trend_slope"
        )
        adjusted_params['intercept'] = st.slider(
            "Intercept (Starting Value)",
            min_value=float(adjusted_params.get('intercept', 0)) * 0.5,
            max_value=float(adjusted_params.get('intercept', 0)) * 1.5,
            value=float(adjusted_params.get('intercept', 0)),
            step=float(adjusted_params.get('intercept', 1)) / 100 if adjusted_params.get('intercept', 0) != 0 else 1.0,
            key="trend_intercept"
        )
    
    elif selected_function_str == TrendFunction.EXPONENTIAL:
        adjusted_params['base'] = st.slider(
            "Base Value",
            min_value=float(adjusted_params.get('base', 1)) * 0.5,
            max_value=float(adjusted_params.get('base', 1)) * 2.0,
            value=float(adjusted_params.get('base', 1)),
            step=float(adjusted_params.get('base', 1)) / 100 if adjusted_params.get('base', 0) != 0 else 1.0,
            key="trend_base"
        )
        adjusted_params['growth_rate'] = st.slider(
            "Growth Rate (per period)",
            min_value=float(adjusted_params.get('growth_rate', 0)) - 0.1,
            max_value=float(adjusted_params.get('growth_rate', 0)) + 0.1,
            value=float(adjusted_params.get('growth_rate', 0)),
            step=0.001,
            format="%.4f",
            key="trend_growth_rate"
        )
    
    elif selected_function_str == TrendFunction.MOVING_AVERAGE:
        adjusted_params['trend'] = st.slider(
            "Trend (Change per Period)",
            min_value=float(adjusted_params.get('trend', 0)) - abs(adjusted_params.get('trend', 0)) * 2,
            max_value=float(adjusted_params.get('trend', 0)) + abs(adjusted_params.get('trend', 0)) * 2,
            value=float(adjusted_params.get('trend', 0)),
            step=float(adjusted_params.get('trend', 1)) / 100 if adjusted_params.get('trend', 0) != 0 else 1.0,
            key="trend_ma_trend"
        )
        adjusted_params['last_value'] = st.slider(
            "Last Value (Starting Point)",
            min_value=float(adjusted_params.get('last_value', 0)) * 0.5,
            max_value=float(adjusted_params.get('last_value', 0)) * 1.5,
            value=float(adjusted_params.get('last_value', 0)),
            step=float(adjusted_params.get('last_value', 1)) / 100 if adjusted_params.get('last_value', 0) != 0 else 1.0,
            key="trend_ma_last"
        )
    
    # Generate forecast with adjusted parameters
    adjusted_forecast = analyzer.generate_forecast_with_params(
        data_series,
        selected_function_str,
        adjusted_params,
        forecast_periods
    )
    
    # Visual preview
    st.markdown("---")
    st.markdown("#### 📊 Forecast Preview")
    
    fig = create_trend_forecast_chart(
        data_series,
        adjusted_forecast,
        selected_metric,
        trend_params,
        forecast_periods
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Historical Avg", f"{data_series.mean():,.0f}")
    with col2:
        st.metric("Forecast Start", f"{adjusted_forecast[0]:,.0f}")
    with col3:
        st.metric("Forecast End", f"{adjusted_forecast[-1]:,.0f}")
    with col4:
        growth_pct = ((adjusted_forecast[-1] - adjusted_forecast[0]) / adjusted_forecast[0] * 100) if adjusted_forecast[0] != 0 else 0
        st.metric("Total Growth", f"{growth_pct:.1f}%")
    
    st.markdown("---")
    
    # Save/Apply buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💾 Save Trend Configuration", type="primary", use_container_width=True, key="save_trend_config"):
            # Save to assumptions
            trend_config = {
                'metric': selected_metric,
                'function_type': selected_function_str.value,
                'parameters': adjusted_params,
                'forecast_periods': forecast_periods,
                'fit_score': trend_params.fit_score,
                'r_squared': trend_params.r_squared
            }
            
            # Load existing assumptions
            assumptions = db.get_scenario_assumptions(scenario_id, user_id)
            if 'trend_forecasts' not in assumptions:
                assumptions['trend_forecasts'] = {}
            assumptions['trend_forecasts'][selected_metric] = trend_config
            
            # Save
            if db.update_assumptions(scenario_id, user_id, assumptions):
                st.success(f"✅ Trend configuration saved for {next(m[1] for m in financial_metrics if m[0] == selected_metric)}")
                st.rerun()
            else:
                st.error("❌ Failed to save trend configuration")
    
    with col2:
        if st.button("🔄 Reset to Fitted Values", use_container_width=True, key="reset_trend_params"):
            st.rerun()
    
    return {
        'metric': selected_metric,
        'function_type': selected_function_str,
        'parameters': adjusted_params,
        'forecast': adjusted_forecast,
        'forecast_periods': forecast_periods
    }


def create_trend_forecast_chart(
    historical_data: pd.Series,
    forecast_values: np.ndarray,
    metric_name: str,
    trend_params: TrendParams,
    forecast_periods: int
) -> go.Figure:
    """Create interactive chart showing historical data and forecast."""
    
    fig = go.Figure()
    
    # Historical data
    hist_x = list(range(len(historical_data)))
    hist_y = historical_data.values
    
    fig.add_trace(go.Scatter(
        x=hist_x,
        y=hist_y,
        mode='lines+markers',
        name='Historical',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate='Period: %{x}<br>Value: %{y:,.0f}<extra></extra>'
    ))
    
    # Forecast data
    forecast_x = list(range(len(historical_data), len(historical_data) + forecast_periods))
    
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=6),
        hovertemplate='Period: %{x}<br>Forecast: %{y:,.0f}<extra></extra>'
    ))
    
    # Connection line
    if len(hist_y) > 0 and len(forecast_values) > 0:
        fig.add_trace(go.Scatter(
            x=[hist_x[-1], forecast_x[0]],
            y=[hist_y[-1], forecast_values[0]],
            mode='lines',
            name='Transition',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Vertical line at forecast start
    fig.add_vline(
        x=len(historical_data) - 0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="Forecast Start",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=f"{metric_name.replace('_', ' ').title()} - Historical Trend & Forecast",
        xaxis_title="Period",
        yaxis_title="Value",
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig
