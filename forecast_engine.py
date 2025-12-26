"""
Forecast Engine
===============
Sprint 21: Extracted forecast calculation logic from UI components.

Core business logic for financial forecasting, separated from UI concerns.
This module is unit-testable and can be used independently of Streamlit.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from copy import deepcopy
from typing import Any as _Any

# Streamlit is optional: this module must also run outside Streamlit (API/worker/tests).
try:  # pragma: no cover
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore


def _ui(method: str, *args: _Any, **kwargs: _Any) -> None:
    """Best-effort UI logging. No-ops when not running under Streamlit."""
    if st is None:
        return
    try:
        fn = getattr(st, method, None)
        if callable(fn):
            fn(*args, **kwargs)
    except Exception:
        return


class ForecastEngine:
    """
    Core forecast calculation engine.
    
    Separated from UI to improve testability and maintainability.
    """
    
    def __init__(self):
        """Initialize the forecast engine."""
        # Lightweight diagnostics captured during a run (for UI explainability).
        self._last_opex_diag: Dict[str, Any] = {}
        self._last_cogs_diag: Dict[str, Any] = {}
        self._last_margin_diag: Dict[str, Any] = {}
    
    def run_forecast(
        self,
        data: Dict[str, Any],
        manufacturing_scenario: Optional[Any] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run the complete forecast calculation.
        
        Args:
            data: Dictionary containing:
                - assumptions: Dict of forecast assumptions
                - ai_assumptions: Optional AI-derived assumptions
                - machines: List of machine records
                - prospects: List of prospect records
                - expenses: List of expense assumptions
                - forecast_months: Number of months to forecast
            manufacturing_scenario: Optional manufacturing strategy scenario
            progress_callback: Optional callback function(progress: float, message: str)
        
        Returns:
            Dictionary with forecast results:
                - success: bool
                - error: Optional error message
                - timeline: List of month strings
                - revenue: Dict with consumables, refurb, pipeline, total
                - costs: Dict with cogs, opex, etc.
                - profit: Dict with gross, ebit
                - summary: Dict with summary statistics
        """
        results = self._initialize_results(manufacturing_scenario)
        
        if progress_callback:
            progress_callback(0.1, "Loading data...")
        
        # Validate inputs
        if not data.get('assumptions'):
            results['error'] = "No assumptions configured. Please complete setup first."
            return results
        
        if not data.get('machines'):
            results['error'] = "No machines found in fleet. Please import fleet data first."
            return results
        
        assumptions = data['assumptions']
        ai_assumptions = data.get('ai_assumptions')
        forecast_months = assumptions.get('forecast_duration_months', 60)
        
        # Get margins using AI assumptions with fallback
        margin_consumable, margin_source_c = self._get_effective_margin(
            ai_assumptions, assumptions, 'gross_margin_liner', 'margin_consumable_pct', 0.38
        )
        margin_refurb, margin_source_r = self._get_effective_margin(
            ai_assumptions, assumptions, 'gross_margin_refurb', 'margin_refurb_pct', 0.32
        )
        
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
        
        if progress_callback:
            progress_callback(0.3, f"Calculating fleet revenue ({len(data['machines'])} machines)...")
        
        # Calculate fleet revenue
        consumable_rev, refurb_rev = self._calculate_fleet_revenue(
            data['machines'], n_months, inflation
        )
        
        if progress_callback:
            progress_callback(0.5, f"Adding pipeline revenue ({len(data.get('prospects', []))} prospects)...")
        
        # Calculate revenue using configured forecast methods
        # NEW: Check forecast_method toggle first (from unified config)
        forecast_method = assumptions.get('forecast_method', 'pipeline')
        use_trend_forecast = bool(assumptions.get('use_trend_forecast', False)) or (forecast_method in ('trend', 'hybrid'))
        # Pipeline mode should ignore any stale trend flags
        if forecast_method == 'pipeline':
            use_trend_forecast = False
        forecast_configs = assumptions.get('forecast_configs', {})
        trend_forecast_config = assumptions.get('trend_forecasts', {})  # Legacy support
        
        # Log which method is being used
        if forecast_method == 'trend':
            _ui("info", "📊 Using **Trend-Based** forecast method")
        elif forecast_method == 'hybrid':
            _ui("info", "🧩 Using **Hybrid** forecast method (Trend baseline + Pipeline overlay)")
        else:
            _ui("info", "📈 Using **Pipeline-Based** forecast method (Fleet + Prospects)")
        
        # ==========================================================================
        # NEW: Line-Item Level Forecasting (Phase 2)
        # ==========================================================================
        # If we have unified_line_item_config, use the new line-item forecast engine
        trend_baseline_total_rev: Optional[np.ndarray] = None
        unified_config = assumptions.get('unified_line_item_config')
        if forecast_method in ('trend', 'hybrid') and unified_config and unified_config.get('line_items'):
            try:
                from components.line_item_forecast_engine import (
                    run_line_item_forecast,
                    convert_to_legacy_format
                )
                
                if progress_callback:
                    progress_callback(0.5, "Running line-item level forecast...")
                
                # Create a mock db object for loading config (config already in assumptions)
                class MockDB:
                    def get_scenario_assumptions(self, sid, uid):
                        return assumptions
                
                mock_db = MockDB()
                user_id = data.get('user_id', '')
                scenario_id = data.get('scenario_id', '')
                
                # Run line-item forecast
                line_item_result = run_line_item_forecast(
                    db=mock_db,
                    scenario_id=scenario_id,
                    user_id=user_id,
                    n_months=n_months,
                    start_date=start_date,
                    historical_data=None,  # Will use config's historical means
                    run_monte_carlo=False,  # MC handled separately
                    mc_iterations=1000,
                    assumptions=assumptions
                )
                
                if line_item_result is not None:
                    _ui("success", f"✅ Line-item forecast complete: {len(line_item_result.line_items)} items forecasted")

                    if forecast_method == 'trend':
                        # Trend-only: convert to legacy format and return (existing behavior)
                        legacy_results = convert_to_legacy_format(line_item_result, start_date)
                        results.update(legacy_results)
                        results['forecast_method_used'] = 'line_item'
                        results['line_item_count'] = len(line_item_result.line_items)

                        if progress_callback:
                            progress_callback(1.0, "Forecast complete")

                        return results

                    # Hybrid: use line-item total revenue as the baseline (do not return early)
                    try:
                        trend_baseline_total_rev = np.array(line_item_result.total_revenue, dtype=float)
                        results['trend_baseline_source'] = 'unified_line_item_config'
                    except Exception:
                        trend_baseline_total_rev = None
                else:
                    _ui("warning", "⚠️ Line-item forecast returned no results. Falling back to legacy trend method.")
            except ImportError as e:
                _ui("warning", f"⚠️ Line-item engine not available: {e}. Using legacy method.")
            except Exception as e:
                _ui("error", f"❌ Line-item forecast error: {e}")
                _ui("info", "Falling back to legacy trend method.")

        # ======================================================================
        # HYBRID: Trend baseline (existing) + Prospect pipeline overlay (new)
        # ======================================================================
        if forecast_method == 'hybrid':
            # Pipeline overlay is always computed from prospects
            pipeline_rev = self._calculate_pipeline_revenue(
                data.get('prospects', []), n_months, inflation, start_date
            )

            base_fleet_rev = consumable_rev + refurb_rev

            # Determine trend baseline for existing revenue
            baseline_existing_total = None

            # 1) Prefer unified line-item baseline if available
            if trend_baseline_total_rev is not None and len(trend_baseline_total_rev) == n_months:
                baseline_existing_total = np.maximum(trend_baseline_total_rev, 0)

            # 2) Else try comprehensive forecast configs (legacy trend engine)
            if baseline_existing_total is None and use_trend_forecast and forecast_configs and len(forecast_configs) > 0:
                revenue_forecast = None
                revenue_config_key = None

                if 'revenue' in forecast_configs:
                    revenue_config_key = 'revenue'
                elif 'total_revenue' in forecast_configs:
                    revenue_config_key = 'total_revenue'

                if revenue_config_key:
                    revenue_forecast = self._generate_forecast_from_config(
                        revenue_config_key,
                        forecast_configs,
                        data,
                        n_months,
                        start_date
                    )

                if revenue_forecast is not None and len(revenue_forecast) == n_months:
                    baseline_existing_total = np.maximum(revenue_forecast, 0)
                else:
                    _ui("warning", "⚠️ Hybrid selected but no valid revenue trend forecast found in forecast_configs. Falling back to fleet baseline.")

            # 3) Else try legacy trend_forecasts (Trend Forecast tab)
            if baseline_existing_total is None and use_trend_forecast and isinstance(trend_forecast_config, dict) and ('revenue' in trend_forecast_config):
                try:
                    from components.trend_forecast_analyzer import TrendForecastAnalyzer, TrendFunction

                    trend_config = trend_forecast_config['revenue']
                    analyzer = TrendForecastAnalyzer()

                    historical_revenue = data.get('historical_revenue', pd.Series())
                    if historical_revenue.empty:
                        hist_financials = data.get('historic_financials', pd.DataFrame())
                        if not hist_financials.empty and 'revenue' in hist_financials.columns:
                            if 'month' in hist_financials.columns:
                                historical_revenue = hist_financials.set_index('month')['revenue'].sort_index()
                            else:
                                historical_revenue = hist_financials['revenue']

                    if not historical_revenue.empty:
                        function_type = TrendFunction(trend_config.get('function_type', 'linear'))
                        trend_params = trend_config.get('parameters', {})
                        trend_revenue = analyzer.generate_forecast_with_params(
                            historical_revenue,
                            function_type,
                            trend_params,
                            n_months
                        )
                        if len(trend_revenue) == n_months:
                            baseline_existing_total = np.maximum(np.array(trend_revenue, dtype=float), 0)
                except Exception:
                    baseline_existing_total = None

            # 4) Final fallback: fleet baseline (hybrid degrades gracefully to pipeline)
            if baseline_existing_total is None:
                baseline_existing_total = base_fleet_rev.copy()

            # Ensure we never go below the fleet baseline (installed base model floor)
            baseline_existing_total = np.maximum(baseline_existing_total, base_fleet_rev)

            # Allocate any uplift above fleet baseline back into consumables/refurb to preserve margins logic
            uplift = baseline_existing_total - base_fleet_rev
            if np.any(uplift > 0):
                # Default split when fleet baseline is zero for a month
                default_cons_share = 0.70
                default_ref_share = 0.30

                denom = np.where(base_fleet_rev > 0, base_fleet_rev, 1.0)
                cons_share = np.where(base_fleet_rev > 0, consumable_rev / denom, default_cons_share)
                ref_share = np.where(base_fleet_rev > 0, refurb_rev / denom, default_ref_share)

                consumable_rev = consumable_rev + uplift * cons_share
                refurb_rev = refurb_rev + uplift * ref_share

            total_rev = consumable_rev + refurb_rev + pipeline_rev

        # Use comprehensive forecast configs if available (only if trend method selected)
        elif forecast_method == 'trend' and use_trend_forecast and forecast_configs and len(forecast_configs) > 0:
            if progress_callback:
                progress_callback(0.5, f"Applying configured forecast methods ({len(forecast_configs)} elements)...")
            
            # Debug: Log what configs we have (only if error occurs)
            # Removed verbose logging - will show errors if forecast generation fails
            
            # Generate revenue using configured method
            # Try 'revenue' first, then 'total_revenue' if that's what's configured
            revenue_forecast = None
            revenue_config_key = None
            
            if 'revenue' in forecast_configs:
                revenue_config_key = 'revenue'
            elif 'total_revenue' in forecast_configs:
                revenue_config_key = 'total_revenue'
            
            if revenue_config_key:
                revenue_forecast = self._generate_forecast_from_config(
                    revenue_config_key,
                    forecast_configs,
                    data,
                    n_months,
                    start_date
                )
            
            if revenue_forecast is not None and len(revenue_forecast) == n_months:
                # Use configured revenue forecast
                # For trend-based forecasts, use the trend forecast directly
                total_rev = np.maximum(revenue_forecast, 0)  # Ensure non-negative
                
                # Calculate pipeline as difference (or zero if trend is lower than fleet)
                base_fleet_rev = consumable_rev + refurb_rev
                if len(base_fleet_rev) > 0 and np.sum(base_fleet_rev) > 0:
                    # If trend forecast is less than fleet, keep fleet minimum
                    total_rev = np.maximum(total_rev, base_fleet_rev)
                    pipeline_rev = total_rev - base_fleet_rev
                else:
                    # No fleet revenue, all is pipeline
                    pipeline_rev = total_rev
            else:
                # ==========================================================
                # ERROR: Trend forecast generation failed
                # ==========================================================
                # DO NOT fall back silently - show user exactly what went wrong
                
                error_details = []
                if revenue_config_key:
                    config = forecast_configs.get(revenue_config_key, {})
                    error_details.append(f"**Element:** {revenue_config_key}")
                    error_details.append(f"**Method:** {config.get('method', 'Not set')}")
                    error_details.append(f"**Trend Type:** {config.get('trend_function_type', 'Not set')}")
                    error_details.append(f"**Parameters:** {config.get('trend_parameters', {})}")
                    
                    if revenue_forecast is None:
                        error_details.append("**Error:** Forecast generation returned None")
                        error_details.append("**Possible Causes:**")
                        error_details.append("- No historical data available for trend fitting")
                        error_details.append("- Invalid trend parameters")
                        error_details.append("- Trend function type not recognized")
                    elif len(revenue_forecast) != n_months:
                        error_details.append(f"**Error:** Length mismatch - got {len(revenue_forecast)} periods, expected {n_months}")
                else:
                    error_details.append("**Error:** No revenue configuration found in forecast_configs")
                    error_details.append(f"**Available configs:** {list(forecast_configs.keys())}")
                
                error_msg = "\n".join(error_details)

                _ui("error", "❌ **TREND FORECAST FAILED**")
                _ui("markdown", error_msg)
                _ui("markdown", "---")
                _ui("warning", """
**How to Fix:**
1. Go to **AI Assumptions → Trend Forecast** tab
2. Select the revenue element and configure a valid trend
3. Click **Save Configuration**
4. Return here and run the forecast again

The forecast cannot continue without valid trend parameters when trend-based forecasting is enabled.
                """)
                
                # Return error result instead of falling back
                results['success'] = False
                results['error'] = f"Trend forecast failed for revenue. {error_details[0] if error_details else 'Unknown error'}"
                return results
        
        # Legacy: Use simple trend forecast if no comprehensive config
        elif forecast_method == 'trend' and use_trend_forecast and 'revenue' in trend_forecast_config:
            # Use trend-based forecast for revenue
            if progress_callback:
                progress_callback(0.5, "Applying trend-based revenue forecast...")
            
            from components.trend_forecast_analyzer import TrendForecastAnalyzer, TrendFunction
            
            trend_config = trend_forecast_config['revenue']
            analyzer = TrendForecastAnalyzer()
            
            # Get historical revenue data
            historical_revenue = data.get('historical_revenue', pd.Series())
            if historical_revenue.empty:
                # Try to get from historic financials
                hist_financials = data.get('historic_financials', pd.DataFrame())
                if not hist_financials.empty and 'revenue' in hist_financials.columns:
                    if 'month' in hist_financials.columns:
                        historical_revenue = hist_financials.set_index('month')['revenue'].sort_index()
                    else:
                        historical_revenue = hist_financials['revenue']
            
            if not historical_revenue.empty:
                function_type = TrendFunction(trend_config.get('function_type', 'linear'))
                trend_params = trend_config.get('parameters', {})
                
                trend_revenue = analyzer.generate_forecast_with_params(
                    historical_revenue,
                    function_type,
                    trend_params,
                    n_months
                )
                
                # Combine: fleet revenue (base) + trend growth
                # Trend forecast represents total revenue growth, so we scale fleet revenue proportionally
                if len(trend_revenue) > 0 and trend_revenue[0] > 0:
                    base_fleet_rev = consumable_rev + refurb_rev
                    if len(base_fleet_rev) > 0 and base_fleet_rev[0] > 0:
                        # Scale trend to start from current fleet revenue level
                        scale_factor = base_fleet_rev[0] / trend_revenue[0] if trend_revenue[0] > 0 else 1.0
                        trend_revenue_scaled = trend_revenue * scale_factor
                        
                        # Use trend forecast as total revenue, with fleet as minimum
                        total_rev = np.maximum(trend_revenue_scaled, base_fleet_rev)
                        pipeline_rev = total_rev - base_fleet_rev  # Difference becomes "pipeline-like" growth
                    else:
                        total_rev = trend_revenue
                        pipeline_rev = np.zeros(n_months)
                else:
                    # ERROR: Legacy trend forecast failed
                    _ui("error", "❌ **LEGACY TREND FORECAST FAILED**")
                    _ui("markdown", "**Element:** revenue")
                    _ui("markdown", f"**Trend Config:** {trend_config}")
                    _ui("markdown", f"**Historical Data Length:** {len(historical_revenue)}")
                    _ui("warning", """
**How to Fix:**
Please configure the trend forecast properly in AI Assumptions → Trend Forecast tab.
                    """)
                    
                    results['success'] = False
                    results['error'] = "Legacy trend forecast failed for revenue - insufficient historical data or invalid parameters"
                    return results
            else:
                # ERROR: No historical data for trend-based forecast
                _ui("error", "❌ **NO HISTORICAL DATA FOR TREND FORECAST**")
                _ui("markdown", "Trend-based forecasting is enabled but no historical revenue data is available.")
                _ui("warning", """
**How to Fix:**
1. Go to **Setup** and import historical financial data
2. Ensure the 'revenue' column is mapped correctly
3. Return here and run the forecast again
                """)
                    
                results['success'] = False
                results['error'] = "No historical revenue data available for trend-based forecast"
                return results
        else:
            # Use pipeline-based forecast (original logic)
            pipeline_rev = self._calculate_pipeline_revenue(
                data.get('prospects', []), n_months, inflation, start_date
            )
            total_rev = consumable_rev + refurb_rev + pipeline_rev

        # ======================================================================
        # Guardrail: First forecast-year revenue should not be below latest actual
        # unless user explicitly sets a lower floor.
        # ======================================================================
        try:
            consumable_rev, refurb_rev, pipeline_rev, total_rev, floor_diag = self._apply_first_forecast_year_revenue_floor(
                timeline=timeline,
                consumable_rev=consumable_rev,
                refurb_rev=refurb_rev,
                pipeline_rev=pipeline_rev,
                total_rev=total_rev,
                data=data,
                assumptions=assumptions,
            )
            if isinstance(floor_diag, dict) and floor_diag.get("applied"):
                results["first_forecast_year_revenue_floor"] = floor_diag
        except Exception:
            # Never fail forecasting due to a guardrail calculation.
            pass

        # ======================================================================
        # Guardrail: Existing customer (fleet) revenue should not decline YoY.
        # This prevents the common case where FY1 is uplifted to match actuals
        # but subsequent years revert to an underspecified wear-profile baseline.
        # ======================================================================
        try:
            consumable_rev, refurb_rev, total_rev, ex_diag = self._apply_existing_customer_revenue_growth_floor(
                timeline=timeline,
                consumable_rev=consumable_rev,
                refurb_rev=refurb_rev,
                pipeline_rev=pipeline_rev,
                total_rev=total_rev,
                assumptions=assumptions,
            )
            if isinstance(ex_diag, dict) and ex_diag.get("applied"):
                results["existing_customer_revenue_floor"] = ex_diag
        except Exception:
            pass
        
        if progress_callback:
            progress_callback(0.6, "Calculating costs...")
        
        # Get forecast configs if available
        forecast_configs = assumptions.get('forecast_configs', {}) if use_trend_forecast else {}
        
        # Calculate COGS - use configured method if available
        # Check both 'cogs' and 'total_cogs' in configs
        cogs_config_key = None
        if forecast_configs:
            if 'cogs' in forecast_configs:
                cogs_config_key = 'cogs'
            elif 'total_cogs' in forecast_configs:
                cogs_config_key = 'total_cogs'
        
        if forecast_configs and cogs_config_key:
            try:
                cogs_data = self._calculate_cogs_with_config(
                    consumable_rev, refurb_rev, pipeline_rev,
                    margin_consumable, margin_refurb,
                    manufacturing_scenario, n_months,
                    forecast_configs, data, start_date, cogs_config_key, total_rev
                )
            except Exception as e:
                _ui("error", f"❌ COGS trend failed: {e}")
                results['success'] = False
                results['error'] = f"COGS trend forecast failed: {e}"
                return results
        else:
            # Use default calculation
            cogs_data = self._calculate_cogs(
                consumable_rev, refurb_rev, pipeline_rev,
                margin_consumable, margin_refurb,
                manufacturing_scenario, n_months
            )
        
        # Calculate OPEX - use configured method if available
        # Check both 'opex' and 'total_opex' in configs
        opex_config_key = None
        if forecast_configs:
            if 'opex' in forecast_configs:
                opex_config_key = 'opex'
            elif 'total_opex' in forecast_configs:
                opex_config_key = 'total_opex'
        
        if forecast_configs and opex_config_key:
            try:
                opex = self._calculate_opex_with_config(
                    data.get('expenses', []), total_rev, cogs_data['cogs'], n_months,
                    forecast_configs, data, start_date, opex_config_key
                )
            except Exception as e:
                _ui("error", f"❌ OPEX trend failed: {e}")
                results['success'] = False
                results['error'] = f"OPEX trend forecast failed: {e}"
                return results
        else:
            # Use default calculation
            opex = self._calculate_opex(
                data.get('expenses', []),
                total_rev,
                cogs_data['cogs'],
                n_months,
                assumptions=assumptions,
                data=data,
            )
        
        if progress_callback:
            progress_callback(0.8, "Finalizing results...")
        
        # Calculate profits
        gross_profit = total_rev - cogs_data['cogs']
        ebit = gross_profit - opex

        # Attach diagnostics (best-effort; never fail the run)
        try:
            results.setdefault("calculation_meta", {})
            results["calculation_meta"]["forecast_method"] = forecast_method
            results["calculation_meta"]["use_trend_forecast"] = bool(use_trend_forecast)
            results["calculation_meta"]["inflation_rate_used"] = float(inflation)
            results["calculation_meta"]["margin_consumable"] = float(margin_consumable)
            results["calculation_meta"]["margin_refurb"] = float(margin_refurb)
            if isinstance(margin_source_c, str):
                results["calculation_meta"]["margin_consumable_source"] = margin_source_c
            if isinstance(margin_source_r, str):
                results["calculation_meta"]["margin_refurb_source"] = margin_source_r

            # COGS source
            if forecast_configs and cogs_config_key:
                cfg = forecast_configs.get(cogs_config_key, {}) if isinstance(forecast_configs, dict) else {}
                results["calculation_meta"]["cogs_method"] = f"forecast_config:{cfg.get('method', 'unknown')}"
            else:
                results["calculation_meta"]["cogs_method"] = "margins"

            # OPEX source
            if forecast_configs and opex_config_key:
                cfg = forecast_configs.get(opex_config_key, {}) if isinstance(forecast_configs, dict) else {}
                results["calculation_meta"]["opex_method"] = f"forecast_config:{cfg.get('method', 'unknown')}"
            else:
                # from _calculate_opex diagnostics if present
                if isinstance(getattr(self, "_last_opex_diag", None), dict) and self._last_opex_diag:
                    results["calculation_meta"]["opex_method"] = self._last_opex_diag.get("mode")
                    results["calculation_meta"]["opex_pct_floor"] = self._last_opex_diag.get("pct_floor")
                    results["calculation_meta"]["opex_pct_source"] = self._last_opex_diag.get("pct_source")
                    results["calculation_meta"]["expense_items"] = self._last_opex_diag.get("expense_items")
                    results["calculation_meta"]["looks_like_setup_defaults"] = self._last_opex_diag.get("looks_like_setup_defaults")
        except Exception:
            pass
        
        # Store results
        results = self._store_results(
            results, consumable_rev, refurb_rev, pipeline_rev, total_rev,
            cogs_data, opex, gross_profit, ebit,
            data, margin_consumable, margin_refurb, manufacturing_scenario
        )
        
        results['success'] = True
        results['assumptions'] = assumptions
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return results
    
    def _initialize_results(self, manufacturing_scenario: Optional[Any]) -> Dict[str, Any]:
        """Initialize results dictionary structure."""
        return {
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
                'cogs_buy': [],
                'cogs_make': [],
                'mfg_variable_overhead': [],
                'mfg_fixed_overhead': [],
                'mfg_overhead': [],
                'mfg_depreciation': [],
                'commissioning': [],
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
            'assumptions_source': 'Manual',
            'ai_assumptions_used': [],
            'manufacturing_included': manufacturing_scenario is not None,
            'manufacturing_strategy': manufacturing_scenario.strategy if manufacturing_scenario else None
        }
    
    def _get_effective_margin(
        self,
        ai_assumptions: Optional[Any],
        assumptions: Dict[str, Any],
        ai_key: str,
        manual_key: str,
        default: float
    ) -> tuple:
        """Get effective margin from AI assumptions with fallback."""
        # For now: prioritize manual assumptions, then AI, then default.
        # IMPORTANT: support both legacy/manual keys:
        # - margin_consumable_pct / margin_refurb_pct
        # - gross_margin_liner / gross_margin_refurb (often saved by UI/AI)
        margin = assumptions.get(manual_key)
        source = 'Manual'

        if margin is None or margin == 0:
            # Try alternate manual keys
            alt_keys = [ai_key]
            # If AI produced an overall gross margin, use it when segment-specific key is missing.
            if str(ai_key).startswith("gross_margin"):
                alt_keys.append("gross_margin_pct")
            for k in alt_keys:
                try:
                    v = assumptions.get(k)
                    if v is not None and v != 0:
                        margin = v
                        source = "Manual"
                        break
                except Exception:
                    continue
        
        if margin is None or margin == 0:
            # Try AI assumptions
            if ai_assumptions:
                try:
                    if hasattr(ai_assumptions, 'assumptions'):
                        ai_assum = ai_assumptions.assumptions.get(ai_key)
                        # Fallback to overall margin if present
                        if not ai_assum and str(ai_key).startswith("gross_margin"):
                            ai_assum = ai_assumptions.assumptions.get("gross_margin_pct")
                        if ai_assum:
                            margin = ai_assum.final_static_value if hasattr(ai_assum, 'final_static_value') else ai_assum
                            source = 'AI'
                except:
                    pass
        
        if margin is None or margin == 0:
            margin = default
            source = 'Default'
        
        # Normalize (handle percentage vs decimal)
        if margin > 1:
            margin = margin / 100
        
        return margin, source
    
    def _calculate_fleet_revenue(
        self,
        machines: List[Dict],
        n_months: int,
        inflation: float
    ) -> tuple:
        """Calculate fleet revenue from machines."""
        consumable_rev = np.zeros(n_months)
        refurb_rev = np.zeros(n_months)
        
        for machine in machines:
            profile = machine.get('wear_profiles_v2', {})
            if not profile:
                profile = {
                    'liner_life_months': 6,
                    'avg_consumable_revenue': 50000,
                    'refurb_interval_months': 36,
                    'avg_refurb_revenue': 150000,
                }
            
            liner_life = max(1, profile.get('liner_life_months', 6) or 6)
            consumable_rev_per = profile.get('avg_consumable_revenue', 50000) or 50000
            refurb_interval = max(1, profile.get('refurb_interval_months', 36) or 36)
            refurb_rev_per = profile.get('avg_refurb_revenue', 150000) or 150000
            
            # Calculate monthly revenue (smoothed)
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
        
        return consumable_rev, refurb_rev
    
    def _calculate_pipeline_revenue(
        self,
        prospects: List[Dict],
        n_months: int,
        inflation: float,
        start_date: datetime
    ) -> np.ndarray:
        """Calculate pipeline revenue from prospects."""
        pipeline_rev = np.zeros(n_months)
        
        for prospect in prospects:
            confidence_raw = prospect.get('confidence_pct', 0) or 0
            confidence = confidence_raw / 100 if confidence_raw > 1 else confidence_raw
            
            annual_liner = prospect.get('annual_liner_value', 0) or 0
            refurb_value = prospect.get('refurb_value', 0) or 0
            
            if annual_liner == 0 and refurb_value == 0:
                annual_liner = prospect.get('expected_annual_revenue', 0) or 0
            
            total_annual_rev = annual_liner + refurb_value
            monthly_rev = (total_annual_rev / 12) * confidence
            
            # Determine start month
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
        
        return pipeline_rev

    def _apply_first_forecast_year_revenue_floor(
        self,
        *,
        timeline: pd.DatetimeIndex,
        consumable_rev: np.ndarray,
        refurb_rev: np.ndarray,
        pipeline_rev: np.ndarray,
        total_rev: np.ndarray,
        data: Dict[str, Any],
        assumptions: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Enforce: first forecast-year total revenue should not be less than the latest
        actual year's revenue, unless the user manually sets a lower floor.

        Implementation:
        - Compute latest actual-year revenue from imported history.
        - Determine the first forecast calendar year after that, within the forecast horizon.
        - If forecast total for that year is below the floor, add an "uplift" allocated
          into consumables/refurb (keeps pipeline unchanged) so margins logic still works.
        """
        diag: Dict[str, Any] = {"applied": False}

        enforce = assumptions.get("enforce_first_forecast_year_revenue_floor", True)
        if isinstance(enforce, str):
            enforce = enforce.strip().lower() not in ("0", "false", "no", "off")
        if not bool(enforce):
            diag["reason"] = "disabled"
            return consumable_rev, refurb_rev, pipeline_rev, total_rev, diag

        manual_floor = assumptions.get("first_forecast_year_revenue_floor", None)
        try:
            manual_floor = float(manual_floor) if manual_floor is not None and manual_floor != "" else None
        except Exception:
            manual_floor = None

        # Load historical financials (prefer aggregated loader output)
        hist = data.get("historical_financials")
        if hist is None or not isinstance(hist, pd.DataFrame) or hist.empty:
            hist = data.get("historic_financials")
        if hist is None or not isinstance(hist, pd.DataFrame) or hist.empty:
            diag["reason"] = "no_historical_data"
            return consumable_rev, refurb_rev, pipeline_rev, total_rev, diag

        # Identify date + revenue columns
        date_col = "period_date" if "period_date" in hist.columns else ("month" if "month" in hist.columns else None)
        rev_col = "total_revenue" if "total_revenue" in hist.columns else ("revenue" if "revenue" in hist.columns else None)
        if not date_col or not rev_col:
            diag["reason"] = "missing_columns"
            diag["missing"] = {"date_col": date_col, "rev_col": rev_col}
            return consumable_rev, refurb_rev, pipeline_rev, total_rev, diag

        h = hist.copy()
        h[date_col] = pd.to_datetime(h[date_col], errors="coerce")
        h = h.dropna(subset=[date_col])
        if h.empty:
            diag["reason"] = "no_valid_dates"
            return consumable_rev, refurb_rev, pipeline_rev, total_rev, diag

        # Prefer is_actual rows when available
        if "is_actual" in h.columns:
            try:
                mask = h["is_actual"].astype(bool)
                if mask.any():
                    h = h[mask]
            except Exception:
                pass

        h[rev_col] = pd.to_numeric(h[rev_col], errors="coerce").fillna(0.0)
        # Keep only non-trivial revenue rows for determining "latest" year
        # (still allows 0-revenue years to be excluded from floor)
        nonzero = h[h[rev_col].abs() > 0]
        if nonzero.empty:
            diag["reason"] = "no_nonzero_revenue"
            return consumable_rev, refurb_rev, pipeline_rev, total_rev, diag

        last_actual_year = int(nonzero[date_col].dt.year.max())
        year_df = h[h[date_col].dt.year == last_actual_year]
        months_in_year = int(year_df[date_col].dt.month.nunique()) if not year_df.empty else 0
        last_year_sum = float(year_df[rev_col].sum()) if not year_df.empty else 0.0

        # Annualize when partial year (common when latest actuals are YTD)
        if months_in_year and months_in_year < 12:
            last_year_annualized = last_year_sum * (12.0 / float(months_in_year))
        else:
            last_year_annualized = last_year_sum

        floor_value = float(manual_floor) if manual_floor is not None else float(last_year_annualized)
        if floor_value < 0:
            # Negative floors don't make sense for revenue; treat as disabled.
            diag["reason"] = "invalid_floor_value"
            diag["floor_value"] = floor_value
            return consumable_rev, refurb_rev, pipeline_rev, total_rev, diag

        # Determine the first forecast year after the latest actual year within this horizon
        forecast_years = sorted(set(pd.to_datetime(timeline).year.tolist()))
        future_years = [y for y in forecast_years if int(y) > int(last_actual_year)]
        if not future_years:
            diag["reason"] = "no_forecast_year_after_actual"
            diag["last_actual_year"] = last_actual_year
            diag["forecast_years"] = forecast_years
            return consumable_rev, refurb_rev, pipeline_rev, total_rev, diag

        first_forecast_year = int(future_years[0])
        year_mask = (pd.to_datetime(timeline).year == first_forecast_year)
        idxs = np.where(year_mask)[0]
        if idxs.size == 0:
            diag["reason"] = "no_months_in_first_forecast_year"
            return consumable_rev, refurb_rev, pipeline_rev, total_rev, diag

        current_year_total = float(np.sum(total_rev[idxs]))

        # If horizon includes only part of the year, floor proportionally.
        target_total = float(floor_value) * (float(idxs.size) / 12.0) if idxs.size < 12 else float(floor_value)

        if current_year_total >= target_total:
            diag.update(
                {
                    "applied": False,
                    "last_actual_year": last_actual_year,
                    "last_actual_year_sum": last_year_sum,
                    "last_actual_year_months": months_in_year,
                    "last_actual_year_annualized": float(last_year_annualized),
                    "first_forecast_year": first_forecast_year,
                    "current_forecast_year_total": current_year_total,
                    "target_floor_total": target_total,
                    "manual_floor": manual_floor,
                }
            )
            return consumable_rev, refurb_rev, pipeline_rev, total_rev, diag

        uplift_total = target_total - current_year_total

        # Allocate uplift across months based on existing (fleet) pattern; fall back to equal weights.
        base_existing = (consumable_rev + refurb_rev)[idxs]
        base_sum = float(np.sum(base_existing))
        if base_sum > 0:
            weights = base_existing / (base_sum + 1e-12)
        else:
            weights = np.ones(idxs.size, dtype=float) / float(idxs.size)

        # Default split when existing is zero for a month
        default_cons_share = 0.70
        default_ref_share = 0.30

        for j, idx in enumerate(idxs.tolist()):
            uplift_m = float(uplift_total) * float(weights[j])
            denom = float(consumable_rev[idx] + refurb_rev[idx])
            if denom > 0:
                cons_share = float(consumable_rev[idx] / denom)
                ref_share = float(refurb_rev[idx] / denom)
            else:
                cons_share = default_cons_share
                ref_share = default_ref_share

            consumable_rev[idx] = float(consumable_rev[idx]) + uplift_m * cons_share
            refurb_rev[idx] = float(refurb_rev[idx]) + uplift_m * ref_share

        total_rev = consumable_rev + refurb_rev + pipeline_rev

        diag.update(
            {
                "applied": True,
                "last_actual_year": last_actual_year,
                "last_actual_year_sum": last_year_sum,
                "last_actual_year_months": months_in_year,
                "last_actual_year_annualized": float(last_year_annualized),
                "first_forecast_year": first_forecast_year,
                "current_forecast_year_total": current_year_total,
                "target_floor_total": target_total,
                "uplift_added_total": float(uplift_total),
                "manual_floor": manual_floor,
            }
        )
        return consumable_rev, refurb_rev, pipeline_rev, total_rev, diag

    def _apply_existing_customer_revenue_growth_floor(
        self,
        *,
        timeline: pd.DatetimeIndex,
        consumable_rev: np.ndarray,
        refurb_rev: np.ndarray,
        pipeline_rev: np.ndarray,
        total_rev: np.ndarray,
        assumptions: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Enforce a minimum year-on-year growth (or non-decline) for existing customer revenue.

        We treat "existing customers revenue" as the installed-base/fleet component:
          existing = consumables + refurb

        Default behavior:
        - Enabled by default
        - Uses inflation as the minimum growth rate (so existing revenue is at least inflationary)

        This helps when wear profiles are incomplete/understated and would otherwise cause
        implausible step-downs after applying first-year floor constraints.
        """
        diag: Dict[str, Any] = {"applied": False}

        enabled = assumptions.get("enforce_existing_customer_revenue_floor", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in ("0", "false", "no", "off")
        if not bool(enabled):
            diag["reason"] = "disabled"
            return consumable_rev, refurb_rev, total_rev, diag

        # Minimum YoY growth rate (percent or decimal). Default = inflation.
        growth_raw = assumptions.get("existing_customer_min_growth_pct", None)
        if growth_raw is None or growth_raw == "":
            growth_raw = assumptions.get("inflation_rate", 0) or 0
        try:
            g = float(growth_raw)
        except Exception:
            g = 0.0
        if g > 1.01:
            g = g / 100.0
        g = float(max(g, 0.0))

        years = np.array(pd.to_datetime(timeline).year, dtype=int)
        uniq_years = sorted(set(int(y) for y in years.tolist()))
        if len(uniq_years) < 2:
            diag["reason"] = "single_year_horizon"
            return consumable_rev, refurb_rev, total_rev, diag

        applied_steps = []
        existing = consumable_rev + refurb_rev

        for i in range(1, len(uniq_years)):
            y_prev = int(uniq_years[i - 1])
            y_cur = int(uniq_years[i])
            idx_prev = np.where(years == y_prev)[0]
            idx_cur = np.where(years == y_cur)[0]
            if idx_prev.size == 0 or idx_cur.size == 0:
                continue

            prev_sum = float(np.sum(existing[idx_prev]))
            cur_sum = float(np.sum(existing[idx_cur]))
            prev_months = float(idx_prev.size)
            cur_months = float(idx_cur.size)
            if prev_months <= 0 or cur_months <= 0:
                continue

            prev_annual = prev_sum * (12.0 / prev_months)
            cur_annual = cur_sum * (12.0 / cur_months)
            min_annual = prev_annual * (1.0 + g)
            if cur_annual >= min_annual:
                continue

            target_sum = min_annual * (cur_months / 12.0)

            # Scale consumables+refurb within the year to meet target.
            if cur_sum > 0:
                factor = target_sum / cur_sum
                consumable_rev[idx_cur] = consumable_rev[idx_cur] * factor
                refurb_rev[idx_cur] = refurb_rev[idx_cur] * factor
            else:
                # No base existing revenue: allocate target evenly using prior-year mix.
                prev_cons = float(np.sum(consumable_rev[idx_prev]))
                prev_ref = float(np.sum(refurb_rev[idx_prev]))
                denom = prev_cons + prev_ref
                cons_share = (prev_cons / denom) if denom > 0 else 0.70
                ref_share = 1.0 - cons_share
                per_month = target_sum / cur_months
                consumable_rev[idx_cur] = per_month * cons_share
                refurb_rev[idx_cur] = per_month * ref_share

            # Update existing for subsequent comparisons
            existing = consumable_rev + refurb_rev
            applied_steps.append(
                {
                    "year": y_cur,
                    "prev_year": y_prev,
                    "prev_annual": prev_annual,
                    "cur_annual_before": cur_annual,
                    "target_annual": min_annual,
                    "growth_floor": g,
                }
            )

        total_rev = consumable_rev + refurb_rev + pipeline_rev

        if applied_steps:
            diag["applied"] = True
            diag["min_growth_rate"] = g
            diag["steps"] = applied_steps[:10]
        else:
            diag["reason"] = "already_meets_floor"

        return consumable_rev, refurb_rev, total_rev, diag
    
    def _generate_forecast_from_config(
        self,
        element_name: str,
        forecast_configs: Dict[str, Any],
        data: Dict[str, Any],
        n_months: int,
        start_date: datetime
    ) -> Optional[np.ndarray]:
        """
        Generate forecast for an element using its configured method.
        
        Args:
            element_name: Name of element to forecast
            forecast_configs: Dictionary of forecast configurations
            data: Forecast data dictionary
            n_months: Number of months to forecast
            start_date: Start date for forecast
        
        Returns:
            Forecast array or None if method not supported
        """
        # Check if element_name is in configs, or try alternative names
        config = None
        actual_element_name = element_name
        
        if element_name in forecast_configs:
            config = forecast_configs[element_name]
            actual_element_name = element_name
        else:
            # Try alternative names (e.g., 'total_revenue' if looking for 'revenue')
            name_alternatives = {
                'revenue': ['total_revenue'],
                'total_revenue': ['revenue'],
                'cogs': ['total_cogs'],
                'total_cogs': ['cogs'],
                'opex': ['total_opex'],
                'total_opex': ['opex']
            }
            
            if element_name in name_alternatives:
                for alt_name in name_alternatives[element_name]:
                    if alt_name in forecast_configs:
                        config = forecast_configs[alt_name]
                        actual_element_name = alt_name
                        break
        
        if config is None:
            # Debug: Log why config wasn't found
            _ui("warning", f"⚠️ No config found for '{element_name}'. Available configs: {list(forecast_configs.keys())}")
            return None
        
        method = config.get('method', 'trend_fit')
        
        try:
            from components.forecast_correlation_engine import (
                ForecastMethod,
                CorrelationCurveType,
                ForecastCorrelationEngine
            )
            from components.trend_forecast_analyzer import TrendForecastAnalyzer, TrendFunction
            
            # Get historical data (support both keys for compatibility)
            historical_data = data.get('historic_financials', None)
            if historical_data is None or (isinstance(historical_data, pd.DataFrame) and historical_data.empty):
                historical_data = data.get('historical_financials', pd.DataFrame())

            if isinstance(historical_data, pd.DataFrame) and historical_data.empty:
                _ui("error", f"❌ No historical data found for {element_name}. Available data keys: {list(data.keys())}")
                return None
            
            # Map element names to possible column names in historical data
            column_name_map = {
                'revenue': ['revenue', 'total_revenue'],
                'cogs': ['cogs', 'total_cogs'],
                'opex': ['opex', 'total_opex'],
                'gross_profit': ['gross_profit', 'total_gross_profit'],
                'depreciation': ['depreciation'],
                'ebit': ['ebit']
            }
            
            # Find the actual column name in historical data
            # Try the element_name first, then mapped names, then the actual_element_name from config
            actual_column = None
            
            # Priority 1: Try element_name directly
            if element_name in historical_data.columns:
                actual_column = element_name
            # Priority 2: Try actual_element_name (from config key)
            elif actual_element_name in historical_data.columns:
                actual_column = actual_element_name
            # Priority 3: Try mapped column names
            elif element_name in column_name_map:
                for possible_name in column_name_map[element_name]:
                    if possible_name in historical_data.columns:
                        actual_column = possible_name
                        break
            # Priority 4: Try mapped names for actual_element_name
            elif actual_element_name in column_name_map:
                for possible_name in column_name_map[actual_element_name]:
                    if possible_name in historical_data.columns:
                        actual_column = possible_name
                        break
            
            if actual_column is None:
                # Try to provide helpful error message
                try:
                    available_cols = list(historical_data.columns)
                    _ui("warning", f"⚠️ Could not find column for {element_name} in historical data. Available columns: {available_cols}")
                except:
                    pass
                return None
            
            # Prepare historical series using the actual column name found
            if 'month' in historical_data.columns:
                data_series = historical_data.set_index('month')[actual_column].sort_index()
            elif 'period_date' in historical_data.columns:
                data_series = historical_data.set_index('period_date')[actual_column].sort_index()
            else:
                data_series = historical_data[actual_column]
            
            data_series = data_series.dropna()
            if len(data_series) < 3:
                _ui("warning", f"⚠️ Insufficient historical data for {element_name}: {len(data_series)} periods (need at least 3)")
                return None
            
            if method == 'trend_fit':
                # Use trend fit
                analyzer = TrendForecastAnalyzer()
                function_type_str = config.get('trend_function_type', 'linear')
                
                # Validate function type
                try:
                    function_type = TrendFunction(function_type_str)
                except ValueError:
                    # Invalid function type, try linear as fallback
                    function_type = TrendFunction.LINEAR
                    _ui("warning", f"Invalid trend function type '{function_type_str}' for {element_name}, using linear")
                
                trend_params = config.get('trend_parameters', {})
                
                # Parameters loaded from config - will be used for forecast generation
                
                # Validate trend parameters are not empty
                if not trend_params:
                    _ui("warning", f"⚠️ No trend parameters found for {element_name}. Using default parameters calculated from historical data.")
                    # Use default parameters based on function type
                    if function_type == TrendFunction.LINEAR:
                        # Default linear: use last value as intercept, calculate slope from data
                        if len(data_series) > 1:
                            slope = (data_series.iloc[-1] - data_series.iloc[0]) / (len(data_series) - 1)
                            intercept = data_series.iloc[-1] - slope * (len(data_series) - 1)
                            trend_params = {'slope': float(slope), 'intercept': float(intercept)}
                        else:
                            trend_params = {'slope': 0, 'intercept': float(data_series.iloc[-1]) if len(data_series) > 0 else 0}
                    elif function_type == TrendFunction.EXPONENTIAL:
                        trend_params = {'base': float(data_series.iloc[-1]) if len(data_series) > 0 else 1, 'growth_rate': 0}
                    else:
                        trend_params = {}
                    
                    _ui("info", f"📊 Calculated default parameters for {element_name}: {trend_params}")
                
                try:
                    forecast = analyzer.generate_forecast_with_params(
                        data_series,
                        function_type,
                        trend_params,
                        n_months
                    )
                    
                    return forecast
                except Exception as forecast_error:
                    # More specific error handling for forecast generation
                    try:
                        import traceback
                        _ui("error", f"❌ Failed to generate {function_type_str} forecast for {element_name}")
                        _ui("error", f"Error: {str(forecast_error)}")
                        _ui("info", f"Parameters used: {trend_params}")
                        _ui("info", f"Historical data: {len(data_series)} periods, range=[{data_series.min():,.0f}, {data_series.max():,.0f}]")
                        _ui("code", traceback.format_exc())
                    except:
                        pass
                    raise  # Re-raise to be caught by outer exception handler
            
            elif method == 'correlation_fixed':
                # Use fixed percentage correlation
                source_metric = config.get('correlation_source')
                fixed_pct = config.get('correlation_fixed_pct', 0)
                
                if source_metric:
                    # Recursively get source forecast
                    source_forecast = self._generate_forecast_from_config(
                        source_metric,
                        forecast_configs,
                        data,
                        n_months,
                        start_date
                    )
                    
                    if source_forecast is not None:
                        return source_forecast * (fixed_pct / 100)
            
            elif method == 'correlation_curve':
                # Use curve-based correlation
                source_metric = config.get('correlation_source')
                curve_type_str = config.get('correlation_type', 'linear')
                curve_params = config.get('correlation_curve_params', {})
                
                if source_metric:
                    correlation_engine = ForecastCorrelationEngine()
                    curve_type = CorrelationCurveType(curve_type_str)
                    
                    # Get source forecast
                    source_forecast = self._generate_forecast_from_config(
                        source_metric,
                        forecast_configs,
                        data,
                        n_months,
                        start_date
                    )
                    
                    if source_forecast is not None:
                        forecast = np.array([
                            correlation_engine.calculate_correlated_value(
                                source_val,
                                curve_type,
                                curve_params
                            )
                            for source_val in source_forecast
                        ])
                        return forecast
            
            elif method == 'fixed_value':
                # Fixed value
                fixed_value = config.get('fixed_value', 0)
                return np.full(n_months, fixed_value)
            
            elif method == 'period_overrides':
                # Period overrides - start with base trend, then apply overrides
                analyzer = TrendForecastAnalyzer()
                function_type = TrendFunction(config.get('trend_function_type', 'linear'))
                trend_params = config.get('trend_parameters', {})
                
                base_forecast = analyzer.generate_forecast_with_params(
                    data_series,
                    function_type,
                    trend_params,
                    n_months
                )
                
                # Apply overrides
                overrides = config.get('period_overrides', {})
                adjusted_forecast = base_forecast.copy()
                
                for period_idx, growth_pct in overrides.items():
                    if 0 <= period_idx < len(adjusted_forecast):
                        if period_idx == 0:
                            adjusted_forecast[period_idx] = base_forecast[period_idx] * (1 + growth_pct / 100)
                        else:
                            adjusted_forecast[period_idx] = adjusted_forecast[period_idx - 1] * (1 + growth_pct / 100)
                
                return adjusted_forecast
        
        except Exception as e:
            # Log detailed error for debugging
            error_msg = f"Error generating forecast for {element_name}: {str(e)}"
            try:
                import traceback
                error_details = traceback.format_exc()
                _ui("error", f"❌ {error_msg}\n\n**Details:**\n```\n{error_details}\n```")
                _ui("warning", f"⚠️ Falling back to default calculation for {element_name}. Please check your trend configuration.")
            except:
                # If st is not available, log to console
                import traceback
                print(f"ERROR: {error_msg}")
                print(traceback.format_exc())
            return None
        
        return None
    
    def _calculate_cogs(
        self,
        consumable_rev: np.ndarray,
        refurb_rev: np.ndarray,
        pipeline_rev: np.ndarray,
        margin_consumable: float,
        margin_refurb: float,
        manufacturing_scenario: Optional[Any],
        n_months: int
    ) -> Dict[str, np.ndarray]:
        """Calculate COGS with manufacturing strategy support."""
        # Base COGS calculation
        cogs_consumable = consumable_rev * (1 - margin_consumable)
        cogs_refurb = refurb_rev * (1 - margin_refurb)
        # Pipeline margin: use the *same* blended margin as the existing (fleet) mix by month.
        # This prevents unintended gross margin drift as pipeline share changes.
        base_fleet_rev = consumable_rev + refurb_rev
        denom = np.where(base_fleet_rev > 0, base_fleet_rev, 1.0)
        cons_share = np.where(base_fleet_rev > 0, consumable_rev / denom, 0.70)
        ref_share = np.where(base_fleet_rev > 0, refurb_rev / denom, 0.30)
        blended_margin = cons_share * float(margin_consumable) + ref_share * float(margin_refurb)
        cogs_pipeline = pipeline_rev * (1 - blended_margin)
        base_cogs = cogs_consumable + cogs_refurb + cogs_pipeline
        
        # Initialize manufacturing arrays
        cogs = np.zeros(n_months)
        cogs_buy = np.zeros(n_months)
        cogs_make = np.zeros(n_months)
        mfg_variable_overhead = np.zeros(n_months)
        mfg_fixed_overhead = np.zeros(n_months)
        mfg_overhead = np.zeros(n_months)
        mfg_depreciation = np.zeros(n_months)
        commissioning_costs = np.zeros(n_months)
        
        if manufacturing_scenario and manufacturing_scenario.strategy != 'buy':
            # Apply manufacturing strategy
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
                        manufacturing_scenario, month_idx, base_cogs[month_idx]
                    )
                    cogs[month_idx] = mfg_impact['total_cogs']
                    cogs_buy[month_idx] = mfg_impact['buy_cogs']
                    cogs_make[month_idx] = mfg_impact['mfg_cogs']
                    mfg_variable_overhead[month_idx] = mfg_impact.get('mfg_variable_overhead', 0)
                    mfg_fixed_overhead[month_idx] = mfg_impact.get('mfg_fixed_overhead', 0)
                    mfg_overhead[month_idx] = mfg_impact.get('mfg_overhead', 0)
                    mfg_depreciation[month_idx] = mfg_impact['mfg_depreciation']
                    commissioning_costs[month_idx] = mfg_impact['commissioning_cost']
            else:
                cogs = base_cogs.copy()
                cogs_buy = base_cogs.copy()
        else:
            # No manufacturing - all COGS is buy
            cogs = base_cogs.copy()
            cogs_buy = base_cogs.copy()
        
        return {
            'cogs': cogs,
            'cogs_buy': cogs_buy,
            'cogs_make': cogs_make,
            'mfg_variable_overhead': mfg_variable_overhead,
            'mfg_fixed_overhead': mfg_fixed_overhead,
            'mfg_overhead': mfg_overhead,
            'mfg_depreciation': mfg_depreciation,
            'commissioning': commissioning_costs
        }
    
    def _calculate_cogs_with_config(
        self,
        consumable_rev: np.ndarray,
        refurb_rev: np.ndarray,
        pipeline_rev: np.ndarray,
        margin_consumable: float,
        margin_refurb: float,
        manufacturing_scenario: Optional[Any],
        n_months: int,
        forecast_configs: Dict[str, Any],
        data: Dict[str, Any],
        start_date: datetime,
        cogs_config_key: str,
        total_rev: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate COGS using configured forecast method.
        
        If config uses correlation with revenue, apply that.
        If config uses trend fit, use historical COGS data.
        Otherwise raise an error (no silent fallback when trend is enabled).
        """
        config = forecast_configs.get(cogs_config_key, {})
        method = config.get('method', 'trend_fit')
        
        # Try to generate COGS forecast from config
        cogs_forecast = self._generate_forecast_from_config(
            cogs_config_key,
            forecast_configs,
            data,
            n_months,
            start_date
        )
        
        if cogs_forecast is not None and len(cogs_forecast) == n_months:
            # Use trend-based COGS forecast
            cogs = np.maximum(cogs_forecast, 0)  # Ensure non-negative
            
            # For manufacturing, we still need to split buy/make
            # Use proportional split based on default calculation
            base_cogs_data = self._calculate_cogs(
                consumable_rev, refurb_rev, pipeline_rev,
                margin_consumable, margin_refurb,
                manufacturing_scenario, n_months
            )
            
            # Scale the buy/make components proportionally
            total_base_cogs = base_cogs_data['cogs']
            if np.sum(total_base_cogs) > 0:
                scale_factor = cogs / (total_base_cogs + 1e-10)  # Avoid division by zero
                cogs_buy = base_cogs_data['cogs_buy'] * scale_factor
                cogs_make = base_cogs_data['cogs_make'] * scale_factor
            else:
                cogs_buy = cogs * 0.8  # Default 80% buy
                cogs_make = cogs * 0.2  # Default 20% make
            
            return {
                'cogs': cogs,
                'cogs_buy': cogs_buy,
                'cogs_make': cogs_make,
                'mfg_variable_overhead': base_cogs_data.get('mfg_variable_overhead', np.zeros(n_months)),
                'mfg_fixed_overhead': base_cogs_data.get('mfg_fixed_overhead', np.zeros(n_months)),
                'mfg_overhead': base_cogs_data.get('mfg_overhead', np.zeros(n_months)),
                'mfg_depreciation': base_cogs_data.get('mfg_depreciation', np.zeros(n_months)),
                'commissioning': base_cogs_data.get('commissioning', np.zeros(n_months))
            }
        else:
            # Trend failed for COGS - raise to stop forecast (no silent fallback)
            error_msg = f"COGS trend forecast failed (method={method}, element={cogs_config_key})"
            _ui("error", f"❌ {error_msg}")
            _ui("caption", "Please fix the COGS trend configuration in AI Assumptions → Trend Forecast.")
            raise RuntimeError(error_msg)
    
    def _calculate_opex(
        self,
        expenses: List[Dict],
        total_rev: np.ndarray,
        cogs: np.ndarray,
        n_months: int,
        *,
        assumptions: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Calculate operating expenses."""
        opex = np.zeros(n_months)
        diag: Dict[str, Any] = {
            "mode": "",
            "expense_items": 0,
            "used_pct_floor": False,
            "pct_floor": None,
            "pct_source": None,
            "baseline_total_opex": None,
            "expense_total_opex": None,
            "looks_like_setup_defaults": False,
        }
        
        # Prefer explicit expense assumptions if present
        active_expenses = [e for e in (expenses or []) if e and e.get('is_active', True)]
        diag["expense_items"] = int(len(active_expenses))
        try:
            default_codes = {"personnel", "logistics", "professional", "facilities", "marketing"}
            codes = {str(e.get("expense_code", "") or "").strip().lower() for e in active_expenses}
            diag["looks_like_setup_defaults"] = bool(default_codes.issubset(codes)) if codes else False
        except Exception:
            pass

        if active_expenses:
            for expense in active_expenses:
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
                        opex[month_idx] += base * inflation_factor + (steps * step_amount)

        # Baseline OPEX as % of revenue (used as fallback AND as a floor when expense assumptions are incomplete).
        pct = None
        pct_source = None
        try:
            if isinstance(assumptions, dict):
                pct = (
                    assumptions.get('opex_as_pct_revenue')
                    or assumptions.get('opex_pct_revenue')
                    or assumptions.get('opex_pct')
                    or assumptions.get('opex_pct_of_revenue')  # some AI/legacy paths use this name
                )
            try:
                pct = float(pct) if pct is not None and pct != "" else None
                pct_source = "manual" if pct is not None else None
            except Exception:
                pct = None
                pct_source = None

            if pct is not None and pct > 1.01:
                pct = pct / 100.0

            # Derive from AI assumptions if missing (common: opex_pct_of_revenue)
            if (pct is None or pct <= 0) and isinstance(data, dict):
                ai = data.get("ai_assumptions")
                try:
                    if ai and hasattr(ai, "assumptions_saved") and ai.assumptions_saved and hasattr(ai, "assumptions"):
                        for k in ["opex_pct_of_revenue", "opex_pct_revenue", "opex_as_pct_revenue"]:
                            if k in ai.assumptions:
                                a = ai.assumptions[k]
                                v = a.final_static_value if hasattr(a, "final_static_value") else a
                                v = float(v)
                                if v > 1.01:
                                    v = v / 100.0
                                if v > 0:
                                    pct = float(np.clip(v, 0.0, 1.0))
                                    pct_source = f"ai:{k}"
                                    break
                except Exception:
                    pass

            # Derive from historicals if still missing
            if pct is None or pct <= 0:
                hist = None
                if isinstance(data, dict):
                    hist = data.get('historic_financials')
                    if hist is None or (isinstance(hist, pd.DataFrame) and hist.empty):
                        hist = data.get('historical_financials')
                if isinstance(hist, pd.DataFrame) and not hist.empty:
                    rev_col = 'revenue' if 'revenue' in hist.columns else ('total_revenue' if 'total_revenue' in hist.columns else None)
                    opx_col = 'opex' if 'opex' in hist.columns else ('total_opex' if 'total_opex' in hist.columns else None)
                    if rev_col and opx_col:
                        h = hist.copy()
                        h[rev_col] = pd.to_numeric(h[rev_col], errors='coerce').fillna(0.0)
                        h[opx_col] = pd.to_numeric(h[opx_col], errors='coerce').fillna(0.0).abs()
                        h = h[h[rev_col].abs() > 0]
                        if not h.empty:
                            # Use latest 12 months for stability
                            if 'period_date' in h.columns:
                                h['period_date'] = pd.to_datetime(h['period_date'], errors='coerce')
                                h = h.dropna(subset=['period_date']).sort_values('period_date')
                            elif 'month' in h.columns:
                                h['month'] = pd.to_datetime(h['month'], errors='coerce')
                                h = h.dropna(subset=['month']).sort_values('month')
                            h_tail = h.tail(12)
                            denom = float(h_tail[rev_col].sum())
                            if denom != 0:
                                pct = float(np.clip(float(h_tail[opx_col].sum()) / denom, 0.0, 1.0))
                                pct_source = "historical"
        except Exception:
            pct = None
            pct_source = None

        if pct is None or pct <= 0:
            pct = 0.27
            pct_source = "default"

        baseline_opex = np.maximum(total_rev, 0) * float(pct)

        # Decide: expense assumptions vs baseline % (and whether to enforce a floor).
        try:
            expense_total = float(np.sum(opex))
        except Exception:
            expense_total = 0.0
        try:
            baseline_total = float(np.sum(baseline_opex))
        except Exception:
            baseline_total = 0.0

        enforce_floor = True
        if isinstance(assumptions, dict):
            v = assumptions.get("enforce_opex_pct_floor", True)
            if isinstance(v, str):
                enforce_floor = v.strip().lower() not in ("0", "false", "no", "off")
            else:
                enforce_floor = bool(v) if v is not None else True

        if expense_total <= 0:
            # No expense assumptions (or they netted to zero): use baseline
            opex = baseline_opex
            diag["mode"] = "pct_of_revenue"
        else:
            if enforce_floor:
                opex = np.maximum(opex, baseline_opex)
                diag["mode"] = "expenses_with_pct_floor"
                diag["used_pct_floor"] = bool(np.sum(baseline_opex > (opex + 1e-9)) > 0)
            else:
                diag["mode"] = "expenses_only"

        diag["pct_floor"] = float(pct)
        diag["pct_source"] = pct_source
        diag["baseline_total_opex"] = baseline_total
        diag["expense_total_opex"] = expense_total

        # stash diagnostics for UI
        try:
            self._last_opex_diag = diag
        except Exception:
            pass

        return opex
    
    def _calculate_opex_with_config(
        self,
        expenses: List[Dict],
        total_rev: np.ndarray,
        cogs: np.ndarray,
        n_months: int,
        forecast_configs: Dict[str, Any],
        data: Dict[str, Any],
        start_date: datetime,
        opex_config_key: str
    ) -> np.ndarray:
        """
        Calculate OPEX using configured forecast method.
        
        If config uses correlation with revenue, apply that.
        If config uses trend fit, use historical OPEX data.
        Otherwise raise an error (no silent fallback when trend is enabled).
        """
        config = forecast_configs.get(opex_config_key, {})
        method = config.get('method', 'trend_fit')
        
        # Try to generate OPEX forecast from config
        opex_forecast = self._generate_forecast_from_config(
            opex_config_key,
            forecast_configs,
            data,
            n_months,
            start_date
        )
        
        if opex_forecast is not None and len(opex_forecast) == n_months:
            # Use trend-based OPEX forecast
            return np.maximum(opex_forecast, 0)  # Ensure non-negative
        else:
            # Trend failed for OPEX - raise to stop forecast (no silent fallback)
            error_msg = f"OPEX trend forecast failed (method={method}, element={opex_config_key})"
            _ui("error", f"❌ {error_msg}")
            _ui("caption", "Please fix the OPEX trend configuration in AI Assumptions → Trend Forecast.")
            raise RuntimeError(error_msg)
    
    def _store_results(
        self,
        results: Dict[str, Any],
        consumable_rev: np.ndarray,
        refurb_rev: np.ndarray,
        pipeline_rev: np.ndarray,
        total_rev: np.ndarray,
        cogs_data: Dict[str, np.ndarray],
        opex: np.ndarray,
        gross_profit: np.ndarray,
        ebit: np.ndarray,
        data: Dict[str, Any],
        margin_consumable: float,
        margin_refurb: float,
        manufacturing_scenario: Optional[Any]
    ) -> Dict[str, Any]:
        """Store calculated results in results dictionary."""
        # Store revenue
        results['revenue']['consumables'] = consumable_rev.tolist()
        results['revenue']['refurb'] = refurb_rev.tolist()
        results['revenue']['pipeline'] = pipeline_rev.tolist()
        results['revenue']['total'] = total_rev.tolist()
        
        # Store costs
        results['costs']['cogs'] = cogs_data['cogs'].tolist()
        results['costs']['cogs_buy'] = cogs_data['cogs_buy'].tolist()
        results['costs']['cogs_make'] = cogs_data['cogs_make'].tolist()
        results['costs']['mfg_variable_overhead'] = cogs_data['mfg_variable_overhead'].tolist()
        results['costs']['mfg_fixed_overhead'] = cogs_data['mfg_fixed_overhead'].tolist()
        results['costs']['mfg_overhead'] = cogs_data['mfg_overhead'].tolist()
        results['costs']['mfg_depreciation'] = cogs_data['mfg_depreciation'].tolist()
        results['costs']['commissioning'] = cogs_data['commissioning'].tolist()
        results['costs']['opex'] = opex.tolist()
        results['costs']['total'] = (cogs_data['cogs'] + opex).tolist()
        
        # Store profit
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
            'total_cogs': float(np.sum(cogs_data['cogs'])),
            'total_cogs_buy': float(np.sum(cogs_data['cogs_buy'])),
            'total_cogs_make': float(np.sum(cogs_data['cogs_make'])),
            'total_mfg_variable_overhead': float(np.sum(cogs_data['mfg_variable_overhead'])),
            'total_mfg_fixed_overhead': float(np.sum(cogs_data['mfg_fixed_overhead'])),
            'total_mfg_overhead': float(np.sum(cogs_data['mfg_overhead'])),
            'total_mfg_depreciation': float(np.sum(cogs_data['mfg_depreciation'])),
            'total_commissioning': float(np.sum(cogs_data['commissioning'])),
            'total_opex': float(np.sum(opex)),
            'total_gross_profit': total_gp_sum,
            'total_ebit': total_ebit_sum,
            'avg_gross_margin': float(total_gp_sum / max(total_revenue_sum, 1)),
            'avg_ebit_margin': float(total_ebit_sum / max(total_revenue_sum, 1)),
            'forecast_months': len(results['timeline']),
            'machine_count': len(data.get('machines', [])),
            'prospect_count': len(data.get('prospects', [])),
            'data_source': data.get('data_source', 'unknown'),
            'margin_consumable_used': margin_consumable,
            'margin_refurb_used': margin_refurb,
            'assumptions_source': results['assumptions_source'],
            'manufacturing_included': manufacturing_scenario is not None,
            'manufacturing_strategy': manufacturing_scenario.strategy if manufacturing_scenario else None
        }
        
        return results


# Convenience function for backward compatibility
def run_forecast_engine(
    data: Dict[str, Any],
    manufacturing_scenario: Optional[Any] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Convenience function to run forecast using the engine.
    
    This maintains backward compatibility while using the new engine.
    """
    engine = ForecastEngine()
    return engine.run_forecast(data, manufacturing_scenario, progress_callback)
