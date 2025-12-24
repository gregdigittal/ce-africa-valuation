"""
Forecast Configuration UI
========================
Comprehensive UI for configuring forecast elements with multiple methods,
correlations, and period overrides.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from components.forecast_correlation_engine import (
    ForecastCorrelationEngine,
    ForecastMethod,
    CorrelationCurveType,
    ForecastElementConfig,
    FORECAST_ELEMENTS,
    validate_forecast_config,
    get_calculated_dependencies
)
from components.trend_forecast_analyzer import TrendForecastAnalyzer, TrendFunction


def render_forecast_config_ui(
    db,
    scenario_id: str,
    user_id: str,
    historical_data: pd.DataFrame,
    forecast_periods: int = 60
) -> Optional[Dict[str, ForecastElementConfig]]:
    """
    Render comprehensive forecast configuration UI.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        user_id: User ID
        historical_data: Historical financial data
        forecast_periods: Number of periods to forecast
    
    Returns:
        Dictionary of element_name -> ForecastElementConfig, or None
    """
    st.markdown("### 📊 Comprehensive Forecast Configuration")
    st.caption("Configure forecast methods for each element with correlations, trends, and period overrides")
    
    if historical_data.empty:
        st.warning("⚠️ No historical data available. Import historical financial statements first.")
        return None
    
    # Initialize correlation engine
    correlation_engine = ForecastCorrelationEngine()
    
    # Load existing configurations
    assumptions_data = db.get_scenario_assumptions(scenario_id, user_id) if hasattr(db, 'get_scenario_assumptions') else {}
    existing_configs = assumptions_data.get('forecast_configs', {})
    
    # Initialize configurations for all elements
    configs = {}
    for element_name, element_def in FORECAST_ELEMENTS.items():
        is_calculated = element_def.get('is_calculated', False)
        
        if element_name in existing_configs:
            # Load from saved config
            config_data = existing_configs[element_name]
            
            # Force calculated elements to use CALCULATED method (fix for old saved configs)
            if is_calculated:
                method = ForecastMethod.CALCULATED
            else:
                method = ForecastMethod(config_data.get('method', 'trend_fit'))
            
            config = ForecastElementConfig(
                element_name=element_name,
                display_name=element_def['display_name'],
                is_calculated=is_calculated,
                calculation_formula=element_def.get('calculation_formula'),
                method=method,
                trend_function_type=config_data.get('trend_function_type') if not is_calculated else None,
                trend_parameters=config_data.get('trend_parameters', {}) if not is_calculated else {},
                correlation_source=config_data.get('correlation_source') if not is_calculated else None,
                correlation_type=CorrelationCurveType(config_data.get('correlation_type', 'linear')) if not is_calculated else CorrelationCurveType.LINEAR,
                correlation_fixed_pct=config_data.get('correlation_fixed_pct') if not is_calculated else None,
                correlation_curve_params=config_data.get('correlation_curve_params', {}) if not is_calculated else {},
                fixed_value=config_data.get('fixed_value') if not is_calculated else None,
                period_overrides=config_data.get('period_overrides', {}) if not is_calculated else {}
            )
        else:
            # Create new config
            config = ForecastElementConfig(
                element_name=element_name,
                display_name=element_def['display_name'],
                is_calculated=is_calculated,
                calculation_formula=element_def.get('calculation_formula'),
                method=ForecastMethod.CALCULATED if is_calculated else ForecastMethod.TREND_FIT
            )
        configs[element_name] = config
    
    # Filter out calculated elements from tabs (they're read-only)
    configurable_elements = {name: config for name, config in configs.items() 
                            if not config.is_calculated}
    
    # Tabs for each configurable forecast element (exclude calculated ones)
    if configurable_elements:
        element_tabs = st.tabs([f"📈 {FORECAST_ELEMENTS[name]['display_name']}" 
                              for name in configurable_elements.keys()])
        
        for tab_idx, (element_name, config) in enumerate(zip(configurable_elements.keys(), configurable_elements.values())):
            with element_tabs[tab_idx]:
                render_element_config_ui(
                    config,
                    historical_data,
                    correlation_engine,
                    configs,
                    forecast_periods
                )
    else:
        st.info("ℹ️ All forecast elements are calculated. No configuration needed.")
    
    # Show calculated elements in an info section
    calculated_elements = {name: config for name, config in configs.items() 
                          if config.is_calculated}
    if calculated_elements:
        st.markdown("---")
        st.markdown("### 📊 Calculated Elements (Read-Only)")
        for element_name, config in calculated_elements.items():
            with st.expander(f"📈 {config.display_name}"):
                st.info(f"**Formula:** {config.calculation_formula}")
                dependencies = get_calculated_dependencies(element_name)
                if dependencies:
                    st.caption(f"**Depends on:** {', '.join([FORECAST_ELEMENTS.get(d, {}).get('display_name', d) for d in dependencies])}")
    
    st.markdown("---")
    
    # Validation and preview
    st.markdown("### ✅ Validation & Preview")
    
    is_valid, errors = validate_forecast_config(configs)
    
    if not is_valid:
        st.error("❌ **Configuration Errors Detected:**")
        for error in errors:
            st.error(f"  - {error}")
        return None
    
    st.success("✅ **Configuration is valid!**")
    
    # Preview forecast
    if st.button("📊 Preview Forecast", type="primary", use_container_width=True, key="preview_forecast_config"):
        preview_forecast = generate_forecast_preview(configs, historical_data, forecast_periods)
        
        if preview_forecast:
            st.markdown("#### Forecast Preview")
            # Create DataFrame with Period column
            preview_df = pd.DataFrame(preview_forecast)
            # Add Period column (1-indexed)
            preview_df.insert(0, 'Period', range(1, len(preview_df) + 1))
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            
            # Chart
            fig = create_forecast_preview_chart(preview_df, historical_data)
            st.plotly_chart(fig, use_container_width=True)
    
    # Save button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True, key="save_forecast_config"):
            # Convert configs to serializable format
            configs_dict = {}
            for name, config in configs.items():
                configs_dict[name] = {
                    'method': config.method.value,
                    'trend_function_type': config.trend_function_type,
                    'trend_parameters': config.trend_parameters,
                    'correlation_source': config.correlation_source,
                    'correlation_type': config.correlation_type.value if config.correlation_type else None,
                    'correlation_fixed_pct': config.correlation_fixed_pct,
                    'correlation_curve_params': config.correlation_curve_params,
                    'fixed_value': config.fixed_value,
                    'period_overrides': config.period_overrides
                }
            
            # Save to assumptions
            assumptions_data = db.get_scenario_assumptions(scenario_id, user_id) if hasattr(db, 'get_scenario_assumptions') else {}
            assumptions_data['forecast_configs'] = configs_dict
            
            if db.update_assumptions(scenario_id, user_id, assumptions_data):
                st.success("✅ Forecast configuration saved to database!")
                
                # Invalidate cache to ensure fresh load on next access
                cache_key = f'assumptions_{scenario_id}'
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
                
                st.rerun()
            else:
                st.error("❌ Failed to save configuration to database. Please try again.")
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True, key="reset_forecast_config"):
            # Clear saved configs
            assumptions_data = db.get_scenario_assumptions(scenario_id, user_id) if hasattr(db, 'get_scenario_assumptions') else {}
            if 'forecast_configs' in assumptions_data:
                del assumptions_data['forecast_configs']
                db.update_assumptions(scenario_id, user_id, assumptions_data)
            st.rerun()
    
    return configs


def render_element_config_ui(
    config: ForecastElementConfig,
    historical_data: pd.DataFrame,
    correlation_engine: ForecastCorrelationEngine,
    all_configs: Dict[str, ForecastElementConfig],
    forecast_periods: int
):
    """Render configuration UI for a single forecast element."""
    
    element_def = FORECAST_ELEMENTS.get(config.element_name, {})
    if not element_def:
        st.warning(f"⚠️ Unknown forecast element: {config.element_name}")
        return
    
    # Show calculated status
    if config.is_calculated:
        st.info(f"📊 **Calculated Element**: {config.calculation_formula}")
        st.caption("This element is automatically calculated from other elements and cannot be configured.")
        
        # Show dependencies
        dependencies = get_calculated_dependencies(config.element_name)
        if dependencies:
            st.markdown("**Depends on:** " + ", ".join([FORECAST_ELEMENTS.get(d, {}).get('display_name', d) for d in dependencies]))
        
        return
    
    st.markdown(f"### {config.display_name}")
    
    # Check if historical data exists for this element
    has_historical = config.element_name in historical_data.columns if not historical_data.empty else False
    if not has_historical:
        st.warning(f"⚠️ No historical data available for {config.display_name}. Some features may be limited.")
    
    # Analyze correlations
    suggested_sources = element_def.get('suggested_correlations', [])
    try:
        correlations = correlation_engine.analyze_correlations(
            historical_data,
            config.element_name,
            suggested_sources
        ) if not historical_data.empty else []
    except Exception as e:
        st.warning(f"⚠️ Error analyzing correlations: {str(e)}")
        correlations = []
    
    # Method selection
    st.markdown("#### Forecast Method")
    
    method_options = {
        ForecastMethod.TREND_FIT: "📈 Trend Fit (Fitted Function)",
        ForecastMethod.CORRELATION_FIXED: "🔗 Correlation - Fixed %",
        ForecastMethod.CORRELATION_CURVE: "📊 Correlation - Curve",
        ForecastMethod.FIXED_VALUE: "🔒 Fixed Value",
        ForecastMethod.PERIOD_OVERRIDES: "📅 Period-by-Period Overrides"
    }
    
    selected_method = st.radio(
        "Select Forecast Method",
        options=list(method_options.keys()),
        format_func=lambda x: method_options[x],
        index=list(method_options.keys()).index(config.method) if config.method in method_options else 0,
        key=f"method_{config.element_name}"
    )
    
    config.method = selected_method
    
    st.markdown("---")
    
    # Render method-specific configuration
    if selected_method == ForecastMethod.TREND_FIT:
        render_trend_fit_config(config, historical_data, forecast_periods)
    
    elif selected_method == ForecastMethod.CORRELATION_FIXED:
        render_correlation_fixed_config(config, correlations, all_configs, historical_data, forecast_periods)
    
    elif selected_method == ForecastMethod.CORRELATION_CURVE:
        render_correlation_curve_config(config, correlations, all_configs, historical_data, forecast_periods)
    
    elif selected_method == ForecastMethod.FIXED_VALUE:
        render_fixed_value_config(config, historical_data)
    
    elif selected_method == ForecastMethod.PERIOD_OVERRIDES:
        render_period_overrides_config(config, historical_data, forecast_periods)


def render_trend_fit_config(
    config: ForecastElementConfig,
    historical_data: pd.DataFrame,
    forecast_periods: int
):
    """Render trend fit configuration."""
    st.markdown("#### Trend Function Configuration")
    
    # Check if element exists in historical data
    if config.element_name not in historical_data.columns:
        st.warning(f"⚠️ No historical data found for {config.display_name}. Using default trend function.")
        st.info("💡 Import historical financial statements that include this metric to enable trend fitting.")
        
        # Provide default configuration option
        st.markdown("**Configure Default Trend:**")
        function_options = {
            TrendFunction.LINEAR: "Linear Growth",
            TrendFunction.EXPONENTIAL: "Exponential Growth",
            TrendFunction.POLYNOMIAL: "Polynomial (Curved)",
            TrendFunction.LOGARITHMIC: "Logarithmic (Diminishing)",
            TrendFunction.POWER: "Power Law",
            TrendFunction.MOVING_AVERAGE: "Moving Average",
            TrendFunction.EXPONENTIAL_SMOOTHING: "Exponential Smoothing"
        }
        
        selected_function = st.selectbox(
            "Trend Function",
            options=list(function_options.keys()),
            format_func=lambda x: function_options[x],
            key=f"trend_func_{config.element_name}"
        )
        
        # Only update function type, preserve existing parameters if they exist
        config.trend_function_type = selected_function.value
        
        # Only set default parameters if they don't already exist (preserve user's saved parameters)
        if not config.trend_parameters:
            if selected_function == TrendFunction.LINEAR:
                config.trend_parameters = {'slope': 0, 'intercept': 0}
            elif selected_function == TrendFunction.EXPONENTIAL:
                config.trend_parameters = {'base': 1, 'growth_rate': 0}
            elif selected_function == TrendFunction.MOVING_AVERAGE:
                config.trend_parameters = {'window': 3, 'trend': 0, 'last_value': 0}
        
        return
    
    # Get historical series
    try:
        if 'month' in historical_data.columns:
            data_series = historical_data.set_index('month')[config.element_name].sort_index()
        elif 'period_date' in historical_data.columns:
            data_series = historical_data.set_index('period_date')[config.element_name].sort_index()
        else:
            data_series = historical_data[config.element_name]
        
        data_series = data_series.dropna()
        
        if len(data_series) < 3:
            st.warning("⚠️ Insufficient historical data for trend fitting.")
            return
    except KeyError:
        st.warning(f"⚠️ Column '{config.element_name}' not found in historical data.")
        return
    
    # Function selection
    function_options = {
        TrendFunction.LINEAR: "Linear Growth",
        TrendFunction.EXPONENTIAL: "Exponential Growth",
        TrendFunction.POLYNOMIAL: "Polynomial (Curved)",
        TrendFunction.LOGARITHMIC: "Logarithmic (Diminishing)",
        TrendFunction.POWER: "Power Law",
        TrendFunction.MOVING_AVERAGE: "Moving Average",
        TrendFunction.EXPONENTIAL_SMOOTHING: "Exponential Smoothing"
    }
    
    # Check if user has saved parameters for this function type
    has_saved_params = (
        config.trend_parameters and 
        len(config.trend_parameters) > 0 and 
        config.trend_function_type
    )
    
    # Default to saved function type if available
    default_function = None
    if config.trend_function_type:
        try:
            default_function = TrendFunction(config.trend_function_type)
        except:
            default_function = selected_function
    else:
        default_function = selected_function
    
    # Get index for selectbox
    function_list = list(function_options.keys())
    try:
        default_index = function_list.index(default_function) if default_function in function_list else 0
    except:
        default_index = 0
    
    selected_function = st.selectbox(
        "Trend Function",
        options=function_list,
        index=default_index,
        format_func=lambda x: function_options[x],
        key=f"trend_func_{config.element_name}"
    )
    
    # Check if function type changed
    function_changed = config.trend_function_type != selected_function.value
    
    # Initialize trend_params variable
    trend_params = None
    
    # Fit function - but only if user doesn't have saved parameters OR function type changed
    # If user has saved parameters and function type unchanged, use those instead of refitting
    if function_changed or not has_saved_params:
        # User changed function type or has no parameters - fit new ones
        if function_changed and has_saved_params:
            st.info(f"ℹ️ Function type changed from {config.trend_function_type} to {selected_function.value}. Fitting new parameters.")
        
        analyzer = TrendForecastAnalyzer()
        trend_params, forecast_values = analyzer.fit_trend_function(
            data_series,
            selected_function,
            forecast_periods
        )
        
        config.trend_function_type = selected_function.value
        config.trend_parameters = trend_params.params
        
        # Show fit quality
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{trend_params.r_squared:.3f}")
        with col2:
            st.metric("Fit Score", f"{trend_params.fit_score:.3f}")
    else:
        # User has saved parameters - use them, but still show preview
        st.success(f"✅ Using saved parameters for {selected_function.value} function")
        analyzer = TrendForecastAnalyzer()
        # Generate forecast with saved parameters for preview
        forecast_values = analyzer.generate_forecast_with_params(
            data_series,
            TrendFunction(config.trend_function_type),
            config.trend_parameters,
            forecast_periods
        )
        
        # Show that we're using saved parameters
        st.info(f"📊 Using saved parameters: {config.trend_parameters}")
        
        # Calculate fit quality for saved parameters (for display)
        try:
            import numpy as np
            from scipy import stats
            x = np.arange(len(data_series))
            y = data_series.values
            if selected_function == TrendFunction.LINEAR:
                slope = config.trend_parameters.get('slope', 0)
                intercept = config.trend_parameters.get('intercept', 0)
                y_pred = slope * x + intercept
            else:
                # For other functions, use the forecast values
                y_pred = forecast_values[:len(y)]
            
            ss_res = np.sum((y - y_pred[:len(y)]) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score (Saved)", f"{r_squared:.3f}")
            with col2:
                st.metric("Status", "Using Saved")
        except Exception as e:
            st.warning(f"Could not calculate fit quality: {e}")
    
    # Parameter adjustment
    st.markdown("#### Adjust Parameters")
    
    # Use saved parameters if available, otherwise use fitted parameters
    if has_saved_params and not function_changed:
        # Use saved parameters as starting point
        adjusted_params = config.trend_parameters.copy()
    else:
        # Use newly fitted parameters
        if trend_params is not None:
            adjusted_params = trend_params.params.copy()
        else:
            # Fallback: use saved parameters or defaults
            adjusted_params = config.trend_parameters.copy() if config.trend_parameters else {}
    
    if selected_function == TrendFunction.LINEAR:
        slope_val = float(adjusted_params.get('slope', 0))
        intercept_val = float(adjusted_params.get('intercept', 0))
        
        slope_range = max(abs(slope_val) * 3, 1000) if slope_val != 0 else 1000
        adjusted_params['slope'] = st.slider(
            "Slope (Growth Rate per Period)",
            min_value=slope_val - slope_range,
            max_value=slope_val + slope_range,
            value=slope_val,
            step=slope_range / 100 if slope_range > 0 else 1.0,
            key=f"slope_{config.element_name}"
        )
        
        intercept_range = max(abs(intercept_val) * 0.5, 1000) if intercept_val != 0 else 1000
        adjusted_params['intercept'] = st.slider(
            "Intercept (Starting Value)",
            min_value=intercept_val - intercept_range,
            max_value=intercept_val + intercept_range,
            value=intercept_val,
            step=intercept_range / 100 if intercept_range > 0 else 1.0,
            key=f"intercept_{config.element_name}"
        )
    
    elif selected_function == TrendFunction.EXPONENTIAL:
        base_val = float(adjusted_params.get('base', data_series.iloc[-1] if len(data_series) > 0 else 1))
        growth_val = float(adjusted_params.get('growth_rate', 0))
        
        adjusted_params['base'] = st.slider(
            "Base Value",
            min_value=max(0.01, base_val * 0.1),
            max_value=base_val * 10.0,
            value=base_val,
            step=base_val / 100 if base_val > 0 else 0.01,
            key=f"base_{config.element_name}"
        )
        adjusted_params['growth_rate'] = st.slider(
            "Growth Rate (per period)",
            min_value=-0.2,
            max_value=0.2,
            value=growth_val,
            step=0.001,
            format="%.4f",
            key=f"growth_{config.element_name}"
        )
    
    elif selected_function == TrendFunction.POLYNOMIAL:
        coeffs = adjusted_params.get('coefficients', [0, 0, data_series.iloc[-1] if len(data_series) > 0 else 0])
        if not isinstance(coeffs, list):
            coeffs = [0, 0, data_series.iloc[-1] if len(data_series) > 0 else 0]
        
        st.markdown("**Polynomial Coefficients:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            coeffs[0] = st.number_input(
                "a (x² coefficient)",
                value=float(coeffs[0]) if len(coeffs) > 0 else 0.0,
                step=0.0001,
                format="%.6f",
                key=f"poly_a_{config.element_name}"
            )
        with col2:
            coeffs[1] = st.number_input(
                "b (x coefficient)",
                value=float(coeffs[1]) if len(coeffs) > 1 else 0.0,
                step=1.0,
                key=f"poly_b_{config.element_name}"
            )
        with col3:
            coeffs[2] = st.number_input(
                "c (constant)",
                value=float(coeffs[2]) if len(coeffs) > 2 else (data_series.iloc[-1] if len(data_series) > 0 else 0.0),
                step=100.0,
                key=f"poly_c_{config.element_name}"
            )
        
        adjusted_params['coefficients'] = coeffs
        adjusted_params['degree'] = 2
    
    elif selected_function == TrendFunction.LOGARITHMIC:
        coeff_val = float(adjusted_params.get('coefficient', 1))
        intercept_val = float(adjusted_params.get('intercept', data_series.iloc[-1] if len(data_series) > 0 else 0))
        
        adjusted_params['coefficient'] = st.slider(
            "Coefficient (a)",
            min_value=coeff_val * 0.1,
            max_value=coeff_val * 10.0,
            value=coeff_val,
            step=coeff_val / 100 if coeff_val != 0 else 0.01,
            key=f"log_coeff_{config.element_name}"
        )
        adjusted_params['intercept'] = st.slider(
            "Intercept (b)",
            min_value=intercept_val - abs(intercept_val) * 2,
            max_value=intercept_val + abs(intercept_val) * 2,
            value=intercept_val,
            step=abs(intercept_val) / 100 if intercept_val != 0 else 1.0,
            key=f"log_intercept_{config.element_name}"
        )
    
    elif selected_function == TrendFunction.POWER:
        coeff_val = float(adjusted_params.get('coefficient', data_series.iloc[-1] if len(data_series) > 0 else 1))
        exp_val = float(adjusted_params.get('exponent', 1))
        
        adjusted_params['coefficient'] = st.slider(
            "Coefficient (a)",
            min_value=max(0.01, coeff_val * 0.1),
            max_value=coeff_val * 10.0,
            value=coeff_val,
            step=coeff_val / 100 if coeff_val > 0 else 0.01,
            key=f"power_coeff_{config.element_name}"
        )
        adjusted_params['exponent'] = st.slider(
            "Exponent (b)",
            min_value=-2.0,
            max_value=2.0,
            value=exp_val,
            step=0.01,
            format="%.3f",
            key=f"power_exp_{config.element_name}"
        )
    
    elif selected_function == TrendFunction.MOVING_AVERAGE:
        trend_val = float(adjusted_params.get('trend', 0))
        last_val = float(adjusted_params.get('last_value', data_series.iloc[-1] if len(data_series) > 0 else 0))
        window_val = int(adjusted_params.get('window', 3))
        
        adjusted_params['window'] = st.slider(
            "Moving Average Window",
            min_value=2,
            max_value=min(12, len(data_series)),
            value=window_val,
            step=1,
            key=f"ma_window_{config.element_name}"
        )
        adjusted_params['trend'] = st.slider(
            "Trend (Change per Period)",
            min_value=trend_val - abs(trend_val) * 3,
            max_value=trend_val + abs(trend_val) * 3,
            value=trend_val,
            step=abs(trend_val) / 100 if trend_val != 0 else 1.0,
            key=f"ma_trend_{config.element_name}"
        )
        adjusted_params['last_value'] = st.slider(
            "Last Value (Starting Point)",
            min_value=max(0, last_val * 0.5),
            max_value=last_val * 2.0,
            value=last_val,
            step=last_val / 100 if last_val > 0 else 1.0,
            key=f"ma_last_{config.element_name}"
        )
    
    elif selected_function == TrendFunction.EXPONENTIAL_SMOOTHING:
        alpha_val = float(adjusted_params.get('alpha', 0.3))
        trend_val = float(adjusted_params.get('trend', 0))
        last_val = float(adjusted_params.get('last_value', data_series.iloc[-1] if len(data_series) > 0 else 0))
        
        adjusted_params['alpha'] = st.slider(
            "Alpha (Smoothing Factor)",
            min_value=0.0,
            max_value=1.0,
            value=alpha_val,
            step=0.01,
            format="%.2f",
            key=f"es_alpha_{config.element_name}"
        )
        adjusted_params['trend'] = st.slider(
            "Trend (Change per Period)",
            min_value=trend_val - abs(trend_val) * 3,
            max_value=trend_val + abs(trend_val) * 3,
            value=trend_val,
            step=abs(trend_val) / 100 if trend_val != 0 else 1.0,
            key=f"es_trend_{config.element_name}"
        )
        adjusted_params['last_value'] = st.slider(
            "Last Value (Starting Point)",
            min_value=max(0, last_val * 0.5),
            max_value=last_val * 2.0,
            value=last_val,
            step=last_val / 100 if last_val > 0 else 1.0,
            key=f"es_last_{config.element_name}"
        )
    
    config.trend_parameters = adjusted_params
    config.trend_function_type = selected_function.value
    
    # Preview
    adjusted_forecast = analyzer.generate_forecast_with_params(
        data_series,
        selected_function,
        adjusted_params,
        forecast_periods
    )
    
    fig = create_trend_preview_chart(data_series, adjusted_forecast, config.element_name)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show forecast summary
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


def render_correlation_fixed_config(
    config: ForecastElementConfig,
    correlations: List,
    all_configs: Dict[str, ForecastElementConfig],
    historical_data: pd.DataFrame,
    forecast_periods: int
):
    """Render fixed percentage correlation configuration."""
    st.markdown("#### Fixed Percentage Correlation")
    
    # Available source metrics
    available_sources = [name for name, elem_def in FORECAST_ELEMENTS.items() 
                        if not elem_def.get('is_calculated', False) and name != config.element_name]
    
    if not available_sources:
        st.warning("⚠️ No available source metrics for correlation.")
        return
    
    # Show suggested correlations
    if correlations:
        st.markdown("**💡 Suggested Correlations (from historical analysis):**")
        for i, corr in enumerate(correlations[:3]):  # Top 3
            st.caption(f"{i+1}. {corr.description}")
    else:
        st.info("💡 Import historical data for both metrics to see correlation suggestions.")
    
    # Source selection
    source_metric = st.selectbox(
        "Correlate with",
        options=available_sources,
        format_func=lambda x: FORECAST_ELEMENTS[x]['display_name'],
        index=available_sources.index(config.correlation_source) if config.correlation_source in available_sources else 0,
        key=f"corr_source_{config.element_name}"
    )
    
    config.correlation_source = source_metric
    
    # Find suggested percentage from correlations
    suggested_pct = None
    for corr in correlations:
        if corr.source_metric == source_metric:
            suggested_pct = corr.avg_percentage
            break
    
    # Percentage input
    default_pct = config.correlation_fixed_pct if config.correlation_fixed_pct else (suggested_pct if suggested_pct else 0)
    
    fixed_pct = st.number_input(
        f"Percentage of {FORECAST_ELEMENTS[source_metric]['display_name']} (%)",
        min_value=0.0,
        max_value=200.0,
        value=float(default_pct),
        step=0.1,
        help=f"Historical average: {suggested_pct:.1f}%" if suggested_pct else None,
        key=f"corr_pct_{config.element_name}"
    )
    
    config.correlation_fixed_pct = fixed_pct
    config.correlation_type = CorrelationCurveType.LINEAR
    
    # Preview
    if source_metric in historical_data.columns:
        source_series = historical_data[source_metric].dropna()
        if len(source_series) > 0:
            # Generate preview forecast for source (simplified - use last value)
            last_source = source_series.iloc[-1]
            preview_values = [last_source * (fixed_pct / 100)] * forecast_periods
            
            fig = create_correlation_preview_chart(
                historical_data,
                config.element_name,
                source_metric,
                fixed_pct,
                preview_values
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"💡 Historical data for {FORECAST_ELEMENTS.get(source_metric, {}).get('display_name', source_metric)} not available for preview.")


def render_correlation_curve_config(
    config: ForecastElementConfig,
    correlations: List,
    all_configs: Dict[str, ForecastElementConfig],
    historical_data: pd.DataFrame,
    forecast_periods: int
):
    """Render curve-based correlation configuration."""
    st.markdown("#### Curve-Based Correlation")
    
    # Available source metrics
    available_sources = [name for name, elem_def in FORECAST_ELEMENTS.items() 
                        if not elem_def['is_calculated'] and name != config.element_name]
    
    if not available_sources:
        st.warning("⚠️ No available source metrics for correlation.")
        return
    
    # Source selection
    source_metric = st.selectbox(
        "Correlate with",
        options=available_sources,
        format_func=lambda x: FORECAST_ELEMENTS[x]['display_name'],
        index=available_sources.index(config.correlation_source) if config.correlation_source in available_sources else 0,
        key=f"curve_source_{config.element_name}"
    )
    
    config.correlation_source = source_metric
    
    # Curve type selection
    curve_options = {
        CorrelationCurveType.LINEAR: "Linear (Fixed %)",
        CorrelationCurveType.LOGARITHMIC: "Logarithmic",
        CorrelationCurveType.POLYNOMIAL: "Polynomial",
        CorrelationCurveType.EXPONENTIAL: "Exponential"
    }
    
    selected_curve = st.selectbox(
        "Curve Type",
        options=list(curve_options.keys()),
        format_func=lambda x: curve_options[x],
        index=list(curve_options.keys()).index(config.correlation_type) if config.correlation_type in curve_options else 0,
        key=f"curve_type_{config.element_name}"
    )
    
    config.correlation_type = selected_curve
    
    # Fit curve to historical data
    correlation_engine = ForecastCorrelationEngine()
    curve_params = correlation_engine.fit_correlation_curve(
        historical_data,
        source_metric,
        config.element_name,
        selected_curve
    )
    
    if curve_params:
        config.correlation_curve_params = curve_params
        
        # Show fitted parameters
        st.markdown("**Fitted Parameters:**")
        for param_name, param_value in curve_params.items():
            st.caption(f"{param_name} = {param_value:.4f}")
        
        # Allow adjustment
        st.markdown("#### Adjust Curve Parameters")
        adjusted_params = curve_params.copy()
        
        # Show current fitted values
        st.caption("**Fitted Parameters:**")
        for param_name, param_value in curve_params.items():
            st.caption(f"{param_name} = {param_value:.6f}")
        
        st.markdown("**Adjust:**")
        
        # Adjust parameters based on curve type
        if selected_curve == CorrelationCurveType.LINEAR:
            # Linear: y = a * x + b
            a_val = float(curve_params.get('a', 0))
            b_val = float(curve_params.get('b', 0))
            
            adjusted_params['a'] = st.slider(
                "a (Slope)",
                min_value=a_val * 0.1 if a_val != 0 else -1.0,
                max_value=a_val * 10.0 if a_val != 0 else 1.0,
                value=a_val,
                step=abs(a_val) / 100 if a_val != 0 else 0.01,
                format="%.6f",
                key=f"curve_a_{config.element_name}"
            )
            adjusted_params['b'] = st.slider(
                "b (Intercept)",
                min_value=b_val - abs(b_val) * 2,
                max_value=b_val + abs(b_val) * 2,
                value=b_val,
                step=abs(b_val) / 100 if b_val != 0 else 1.0,
                key=f"curve_b_{config.element_name}"
            )
        
        elif selected_curve == CorrelationCurveType.LOGARITHMIC:
            # Logarithmic: y = a * ln(x) + b
            a_val = float(curve_params.get('a', 1))
            b_val = float(curve_params.get('b', 0))
            
            adjusted_params['a'] = st.slider(
                "a (Coefficient)",
                min_value=a_val * 0.1 if a_val != 0 else -100.0,
                max_value=a_val * 10.0 if a_val != 0 else 100.0,
                value=a_val,
                step=abs(a_val) / 100 if a_val != 0 else 0.1,
                format="%.4f",
                key=f"curve_a_{config.element_name}"
            )
            adjusted_params['b'] = st.slider(
                "b (Intercept)",
                min_value=b_val - abs(b_val) * 2,
                max_value=b_val + abs(b_val) * 2,
                value=b_val,
                step=abs(b_val) / 100 if b_val != 0 else 1.0,
                key=f"curve_b_{config.element_name}"
            )
        
        elif selected_curve == CorrelationCurveType.POLYNOMIAL:
            # Polynomial: y = a * x² + b * x + c
            a_val = float(curve_params.get('a', 0))
            b_val = float(curve_params.get('b', 0))
            c_val = float(curve_params.get('c', 0))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                adjusted_params['a'] = st.number_input(
                    "a (x² coefficient)",
                    value=a_val,
                    step=0.000001,
                    format="%.8f",
                    key=f"curve_a_{config.element_name}"
                )
            with col2:
                adjusted_params['b'] = st.number_input(
                    "b (x coefficient)",
                    value=b_val,
                    step=0.01,
                    format="%.4f",
                    key=f"curve_b_{config.element_name}"
                )
            with col3:
                adjusted_params['c'] = st.number_input(
                    "c (constant)",
                    value=c_val,
                    step=1.0,
                    key=f"curve_c_{config.element_name}"
                )
        
        elif selected_curve == CorrelationCurveType.EXPONENTIAL:
            # Exponential: y = a * exp(b * x)
            a_val = float(curve_params.get('a', 1))
            b_val = float(curve_params.get('b', 0))
            
            adjusted_params['a'] = st.slider(
                "a (Base)",
                min_value=max(0.01, a_val * 0.1),
                max_value=a_val * 10.0,
                value=a_val,
                step=a_val / 100 if a_val > 0 else 0.01,
                format="%.4f",
                key=f"curve_a_{config.element_name}"
            )
            adjusted_params['b'] = st.slider(
                "b (Exponent)",
                min_value=-0.1,
                max_value=0.1,
                value=b_val,
                step=0.0001,
                format="%.6f",
                key=f"curve_b_{config.element_name}"
            )
        
        config.correlation_curve_params = adjusted_params
    
    # Preview
    if source_metric in historical_data.columns:
        source_series = historical_data[source_metric].dropna()
        if len(source_series) > 0:
            last_source = source_series.iloc[-1]
            # Generate preview using curve
            preview_values = []
            for i in range(forecast_periods):
                # Simple growth assumption for preview
                source_val = last_source * (1.02 ** i)  # 2% growth
                target_val = correlation_engine.calculate_correlated_value(
                    source_val,
                    selected_curve,
                    config.correlation_curve_params
                )
                preview_values.append(target_val)
            
            fig = create_correlation_preview_chart(
                historical_data,
                config.element_name,
                source_metric,
                None,  # No fixed pct for curve
                preview_values
            )
            st.plotly_chart(fig, use_container_width=True)


def render_fixed_value_config(
    config: ForecastElementConfig,
    historical_data: pd.DataFrame
):
    """Render fixed value configuration."""
    st.markdown("#### Fixed Value")
    
    # Get last historical value as default
    if config.element_name in historical_data.columns:
        last_value = historical_data[config.element_name].dropna().iloc[-1] if len(historical_data[config.element_name].dropna()) > 0 else 0
    else:
        last_value = 0
    
    default_value = config.fixed_value if config.fixed_value else last_value
    
    fixed_value = st.number_input(
        "Fixed Value",
        min_value=0.0,
        value=float(default_value),
        step=1000.0,
        format="%.0f",
        key=f"fixed_{config.element_name}"
    )
    
    config.fixed_value = fixed_value


def render_period_overrides_config(
    config: ForecastElementConfig,
    historical_data: pd.DataFrame,
    forecast_periods: int
):
    """Render period-by-period override configuration."""
    st.markdown("#### Period-by-Period Growth Overrides")
    st.caption("Override fitted forecast with growth percentages for specific periods")
    
    # Get base forecast (from trend fit)
    base_forecast = None
    if config.element_name in historical_data.columns:
        if 'month' in historical_data.columns:
            data_series = historical_data.set_index('month')[config.element_name].sort_index()
        else:
            data_series = historical_data[config.element_name]
        data_series = data_series.dropna()
        
        if len(data_series) >= 3:
            analyzer = TrendForecastAnalyzer()
            # Use configured trend function if available, otherwise use LINEAR
            function_type = TrendFunction(config.trend_function_type) if config.trend_function_type else TrendFunction.LINEAR
            trend_params = config.trend_parameters if config.trend_parameters else {}
            
            if trend_params:
                base_forecast = analyzer.generate_forecast_with_params(
                    data_series,
                    function_type,
                    trend_params,
                    forecast_periods
                )
            else:
                _, base_forecast = analyzer.fit_trend_function(
                    data_series,
                    TrendFunction.LINEAR,
                    forecast_periods
                )
        else:
            base_forecast = np.zeros(forecast_periods)
    else:
        # No historical data - use fixed value or zero
        base_value = config.fixed_value if config.fixed_value else 0
        base_forecast = np.full(forecast_periods, base_value)
    
    # Period override editor
    st.markdown("**Override Growth % by Period:**")
    
    # Initialize overrides
    if not config.period_overrides:
        config.period_overrides = {}
    
    # Create editable table
    override_data = []
    for period in range(forecast_periods):
        override_data.append({
            'Period': period + 1,
            'Base Forecast': f"{base_forecast[period]:,.0f}" if period < len(base_forecast) else "0",
            'Growth % Override': config.period_overrides.get(period, 0.0)
        })
    
    override_df = pd.DataFrame(override_data)
    
    edited_df = st.data_editor(
        override_df,
        column_config={
            'Period': st.column_config.NumberColumn('Period', disabled=True),
            'Base Forecast': st.column_config.TextColumn('Base Forecast', disabled=True),
            'Growth % Override': st.column_config.NumberColumn('Growth %', min_value=-100.0, max_value=1000.0, step=0.1)
        },
        hide_index=True,
        use_container_width=True,
        key=f"overrides_{config.element_name}"
    )
    
    # Update config from edited dataframe
    config.period_overrides = {}
    for _, row in edited_df.iterrows():
        period_idx = int(row['Period']) - 1
        growth_pct = float(row['Growth % Override'])
        if growth_pct != 0:
            config.period_overrides[period_idx] = growth_pct
    
    # Preview with overrides
    if len(base_forecast) > 0:
        adjusted_forecast = base_forecast.copy()
        for period_idx, growth_pct in config.period_overrides.items():
            if period_idx < len(adjusted_forecast):
                if period_idx == 0:
                    adjusted_forecast[period_idx] = base_forecast[period_idx] * (1 + growth_pct / 100)
                else:
                    adjusted_forecast[period_idx] = adjusted_forecast[period_idx - 1] * (1 + growth_pct / 100)
        
        fig = create_override_preview_chart(base_forecast, adjusted_forecast, config.element_name)
        st.plotly_chart(fig, use_container_width=True)


def create_trend_preview_chart(
    historical_data: pd.Series,
    forecast_values: np.ndarray,
    element_name: str
) -> go.Figure:
    """Create preview chart for trend fit."""
    fig = go.Figure()
    
    # Historical
    hist_x = list(range(len(historical_data)))
    hist_y = historical_data.values
    
    fig.add_trace(go.Scatter(
        x=hist_x,
        y=hist_y,
        mode='lines+markers',
        name='Historical',
        line=dict(color='#1f77b4', width=3)
    ))
    
    # Forecast
    forecast_x = list(range(len(historical_data), len(historical_data) + len(forecast_values)))
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title=f"{FORECAST_ELEMENTS[element_name]['display_name']} - Trend Forecast",
        xaxis_title="Period",
        yaxis_title="Value",
        height=400
    )
    
    return fig


def create_correlation_preview_chart(
    historical_data: pd.DataFrame,
    target_metric: str,
    source_metric: str,
    fixed_pct: Optional[float],
    preview_values: List[float]
) -> go.Figure:
    """Create preview chart for correlation."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Historical target
    if target_metric in historical_data.columns:
        hist_target = historical_data[target_metric].dropna()
        hist_x = list(range(len(hist_target)))
        fig.add_trace(
            go.Scatter(x=hist_x, y=hist_target.values, name=f'Historical {FORECAST_ELEMENTS[target_metric]["display_name"]}', line=dict(color='blue')),
            secondary_y=False
        )
    
    # Historical source
    if source_metric in historical_data.columns:
        hist_source = historical_data[source_metric].dropna()
        hist_x = list(range(len(hist_source)))
        fig.add_trace(
            go.Scatter(x=hist_x, y=hist_source.values, name=f'Historical {FORECAST_ELEMENTS[source_metric]["display_name"]}', line=dict(color='green', dash='dot')),
            secondary_y=True
        )
    
    # Forecast
    forecast_x = list(range(len(hist_target) if target_metric in historical_data.columns else 0, 
                            (len(hist_target) if target_metric in historical_data.columns else 0) + len(preview_values)))
    fig.add_trace(
        go.Scatter(x=forecast_x, y=preview_values, name='Forecast (Correlated)', line=dict(color='orange', dash='dash')),
        secondary_y=False
    )
    
    fig.update_xaxes(title_text="Period")
    fig.update_yaxes(title_text=f"{FORECAST_ELEMENTS[target_metric]['display_name']}", secondary_y=False)
    fig.update_yaxes(title_text=f"{FORECAST_ELEMENTS[source_metric]['display_name']}", secondary_y=True)
    fig.update_layout(title=f"Correlation: {FORECAST_ELEMENTS[target_metric]['display_name']} vs {FORECAST_ELEMENTS[source_metric]['display_name']}", height=400)
    
    return fig


def create_override_preview_chart(
    base_forecast: np.ndarray,
    adjusted_forecast: np.ndarray,
    element_name: str
) -> go.Figure:
    """Create preview chart for period overrides."""
    fig = go.Figure()
    
    periods = list(range(len(base_forecast)))
    
    fig.add_trace(go.Scatter(
        x=periods,
        y=base_forecast,
        mode='lines',
        name='Base Forecast',
        line=dict(color='blue', dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=periods,
        y=adjusted_forecast,
        mode='lines+markers',
        name='With Overrides',
        line=dict(color='orange', width=3)
    ))
    
    fig.update_layout(
        title=f"{FORECAST_ELEMENTS[element_name]['display_name']} - Period Overrides",
        xaxis_title="Period",
        yaxis_title="Value",
        height=400
    )
    
    return fig


def generate_forecast_preview(
    configs: Dict[str, ForecastElementConfig],
    historical_data: pd.DataFrame,
    forecast_periods: int
) -> Optional[Dict[str, List[float]]]:
    """Generate forecast preview from configurations."""
    preview = {}
    
    # First pass: Calculate input elements
    for element_name, config in configs.items():
        if config.is_calculated:
            continue  # Skip calculated elements for now
        
        if config.method == ForecastMethod.TREND_FIT:
            # Use trend fit
            if not historical_data.empty and element_name in historical_data.columns:
                try:
                    if 'month' in historical_data.columns:
                        data_series = historical_data.set_index('month')[element_name].sort_index()
                    elif 'period_date' in historical_data.columns:
                        data_series = historical_data.set_index('period_date')[element_name].sort_index()
                    else:
                        data_series = historical_data[element_name]
                    data_series = data_series.dropna()
                    
                    if len(data_series) >= 3:
                        analyzer = TrendForecastAnalyzer()
                        function_type = TrendFunction(config.trend_function_type) if config.trend_function_type else TrendFunction.LINEAR
                        _, forecast = analyzer.fit_trend_function(data_series, function_type, forecast_periods)
                        preview[element_name] = forecast.tolist()
                except Exception as e:
                    # Skip this element if there's an error
                    continue
        
        elif config.method == ForecastMethod.CORRELATION_FIXED:
            # Use correlation - need source forecast first
            if config.correlation_source and config.correlation_source in preview:
                source_forecast = preview[config.correlation_source]
                preview[element_name] = [v * (config.correlation_fixed_pct / 100) for v in source_forecast]
        
        elif config.method == ForecastMethod.FIXED_VALUE:
            preview[element_name] = [config.fixed_value] * forecast_periods
    
    # Second pass: Calculate derived elements (in dependency order)
    # 1. Calculate EBT (Earnings Before Tax) = EBIT - Interest
    if 'ebit' in preview and 'interest_expense' in preview:
        preview['ebt'] = [ebit - int_exp for ebit, int_exp in zip(preview['ebit'], preview['interest_expense'])]
    elif 'ebit' in preview:
        # If interest not available, assume 0 for preview
        preview['ebt'] = preview['ebit']
    
    # 2. Calculate Tax = max(EBT × Tax Rate, 0)
    for element_name, config in configs.items():
        if config.is_calculated:
            formula = config.calculation_formula
            if formula == 'revenue - cogs' and 'revenue' in preview and 'cogs' in preview:
                preview[element_name] = [r - c for r, c in zip(preview['revenue'], preview['cogs'])]
            elif formula == 'gross_profit - opex - depreciation':
                if 'gross_profit' in preview and 'opex' in preview and 'depreciation' in preview:
                    preview[element_name] = [gp - op - dep for gp, op, dep in 
                                            zip(preview['gross_profit'], preview['opex'], preview['depreciation'])]
            elif 'max(ebt * tax_rate, 0)' in formula:
                # Tax calculation: EBT × Tax Rate (default 28% if not available)
                if 'ebt' in preview:
                    tax_rate = 0.28  # Default tax rate (can be improved to get from assumptions)
                    preview[element_name] = [max(ebt * tax_rate, 0) for ebt in preview['ebt']]
            elif 'calculated_from_balance_sheet' in formula:
                # Interest is calculated from balance sheet, skip for preview
                # Use 0 as placeholder
                if element_name not in preview:
                    preview[element_name] = [0] * forecast_periods
            elif formula == 'ebit - interest_expense - tax':
                if 'ebit' in preview and 'interest_expense' in preview and 'tax' in preview:
                    preview[element_name] = [ebit - int_exp - t for ebit, int_exp, t in 
                                            zip(preview['ebit'], preview['interest_expense'], preview['tax'])]
    
    return preview


def create_forecast_preview_chart(
    preview_df: pd.DataFrame,
    historical_data: pd.DataFrame
) -> go.Figure:
    """Create comprehensive forecast preview chart."""
    fig = go.Figure()
    
    # Ensure Period column exists
    if 'Period' not in preview_df.columns:
        preview_df = preview_df.copy()
        preview_df.insert(0, 'Period', range(1, len(preview_df) + 1))
    
    # Get Period values for x-axis
    period_values = preview_df['Period'].values if 'Period' in preview_df.columns else range(1, len(preview_df) + 1)
    
    for column in preview_df.columns:
        if column != 'Period':
            # Get display name from FORECAST_ELEMENTS if available
            display_name = FORECAST_ELEMENTS.get(column, {}).get('display_name', column.replace('_', ' ').title())
            
            fig.add_trace(go.Scatter(
                x=period_values,
                y=preview_df[column].values,
                mode='lines+markers',
                name=display_name
            ))
    
    fig.update_layout(
        title="Forecast Preview - All Elements",
        xaxis_title="Period",
        yaxis_title="Value",
        height=500,
        hovermode='x unified'
    )
    
    return fig
