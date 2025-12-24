"""
Line-Item Level Forecast Engine
================================
Forecasts each line item individually using its configured trend,
then sums to get aggregates.

Phase 2 of Unified Configuration Backlog.
Date: December 20, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass

try:
    from components.unified_assumptions_config import (
        UnifiedAssumptionsConfig,
        LineItemConfig,
        load_unified_config,
        CALCULATED_ELEMENTS
    )
except ImportError:
    # Fallback for direct testing
    pass


# =============================================================================
# TREND FUNCTIONS
# =============================================================================

def apply_trend(
    base_value: float,
    trend_type: str,
    growth_rate: float,
    n_months: int,
    monthly: bool = True,
    manual_intercept: float = None,
    manual_slope: float = None
) -> np.ndarray:
    """
    Apply a trend function to generate forecast values.
    
    Args:
        base_value: Starting value (historical mean or last actual)
        trend_type: 'flat', 'linear', 'exponential', 'logarithmic', 'manual'
        growth_rate: Annual growth rate as percentage (e.g., 5.0 for 5%)
        n_months: Number of months to forecast
        monthly: If True, growth_rate is annual and converted to monthly
        manual_intercept: Starting value for manual trend (overrides base_value)
        manual_slope: Monthly change for manual trend
        
    Returns:
        Array of forecasted values
    """
    # Time indices (0-indexed for intercept at t=0)
    t = np.arange(0, n_months)
    
    # Handle manual trend first (user-specified intercept and slope)
    if trend_type == 'manual':
        intercept = manual_intercept if manual_intercept is not None else base_value
        slope = manual_slope if manual_slope is not None else 0.0
        # Linear: y = intercept + slope * t
        return intercept + slope * t
    
    if base_value == 0:
        return np.zeros(n_months)
    
    # Convert annual growth to monthly
    if monthly:
        monthly_rate = (1 + growth_rate / 100) ** (1/12) - 1
    else:
        monthly_rate = growth_rate / 100
    
    # Time indices (1-indexed for proper growth in non-manual modes)
    t = np.arange(1, n_months + 1)
    
    if trend_type == 'flat':
        # No growth - constant value
        return np.full(n_months, base_value)
    
    elif trend_type == 'linear':
        # Linear growth: value increases by fixed amount each period
        monthly_increment = base_value * monthly_rate
        return base_value + monthly_increment * t
    
    elif trend_type == 'exponential':
        # Exponential growth: value grows by percentage each period
        return base_value * (1 + monthly_rate) ** t
    
    elif trend_type == 'logarithmic':
        # Logarithmic growth: fast initial growth that slows
        # y = base * (1 + growth_rate * log(1 + t/12))
        if growth_rate >= 0:
            return base_value * (1 + (growth_rate / 100) * np.log(1 + t / 12))
        else:
            # For negative growth, use inverse log
            return base_value * (1 - abs(growth_rate / 100) * np.log(1 + t / 12))
    
    else:
        # Default to flat
        return np.full(n_months, base_value)


def apply_distribution(
    values: np.ndarray,
    distribution_type: str,
    cv: float,
    n_samples: int = 1000
) -> np.ndarray:
    """
    Apply Monte Carlo distribution to forecast values.
    
    Args:
        values: Base forecast values (one per period)
        distribution_type: 'normal', 'lognormal', 'triangular', 'uniform'
        cv: Coefficient of variation (std/mean)
        n_samples: Number of MC samples per period
        
    Returns:
        Array of shape (n_samples, n_periods) with sampled values
    """
    n_periods = len(values)
    samples = np.zeros((n_samples, n_periods))
    
    for i, mean_value in enumerate(values):
        if mean_value == 0:
            samples[:, i] = 0
            continue
        
        std = abs(mean_value) * cv
        
        if distribution_type == 'normal':
            samples[:, i] = np.random.normal(mean_value, std, n_samples)
        
        elif distribution_type == 'lognormal':
            # Convert to log parameters
            if mean_value > 0:
                sigma = np.sqrt(np.log(1 + (cv ** 2)))
                mu = np.log(mean_value) - sigma ** 2 / 2
                samples[:, i] = np.random.lognormal(mu, sigma, n_samples)
            else:
                samples[:, i] = mean_value
        
        elif distribution_type == 'triangular':
            # Triangular with mode at mean, ±2*std
            low = mean_value - 2 * std
            high = mean_value + 2 * std
            samples[:, i] = np.random.triangular(low, mean_value, high, n_samples)
        
        elif distribution_type == 'uniform':
            low = mean_value - std * np.sqrt(3)
            high = mean_value + std * np.sqrt(3)
            samples[:, i] = np.random.uniform(low, high, n_samples)
        
        else:
            samples[:, i] = mean_value
    
    return samples


# =============================================================================
# LINE-ITEM FORECAST ENGINE
# =============================================================================

@dataclass
class LineItemForecast:
    """Forecast result for a single line item."""
    key: str
    name: str
    category: str
    statement_type: str
    values: np.ndarray  # Forecast values per period
    samples: Optional[np.ndarray] = None  # MC samples if enabled


@dataclass
class AggregatedForecast:
    """Aggregated forecast from line items."""
    total_revenue: np.ndarray
    total_cogs: np.ndarray
    gross_profit: np.ndarray
    total_opex: np.ndarray
    ebit: np.ndarray
    depreciation: np.ndarray
    interest_expense: np.ndarray
    tax: np.ndarray
    net_income: np.ndarray
    
    # Line-item detail
    line_items: Dict[str, LineItemForecast]
    
    # MC results (if enabled)
    mc_enabled: bool = False
    mc_samples: Optional[Dict[str, np.ndarray]] = None


class LineItemForecastEngine:
    """
    Forecasts each line item individually, then aggregates.
    """
    
    def __init__(self, config: UnifiedAssumptionsConfig, correlation_config: Optional[Dict] = None):
        self.config = config
        self.line_items = config.line_items
        self.correlation_config = correlation_config
    
    def run_forecast(
        self,
        n_months: int,
        start_date: datetime,
        historical_data: Optional[pd.DataFrame] = None,
        run_monte_carlo: bool = False,
        mc_iterations: int = 1000,
        tax_rate: float = 0.28,
        interest_rate: float = 0.10,
        depreciation_pct: float = 0.05,
    ) -> AggregatedForecast:
        """
        Run the line-item level forecast.
        
        Args:
            n_months: Number of months to forecast
            start_date: Forecast start date
            historical_data: Optional historical data for base values
            run_monte_carlo: Whether to run MC simulation
            mc_iterations: Number of MC iterations
            tax_rate: Tax rate for net income calculation
            interest_rate: Annual interest rate
            depreciation_pct: Depreciation as % of OPEX
            
        Returns:
            AggregatedForecast with line-item detail
        """
        line_item_forecasts: Dict[str, LineItemForecast] = {}
        
        # Group line items by category for aggregation
        revenue_items = []
        cogs_items = []
        opex_items = []
        other_items = []
        
        # Forecast each line item
        for key, item in self.line_items.items():
            if not item.is_configurable:
                continue
            
            # Get base value (historical mean or from data)
            base_value = self._get_base_value(key, item, historical_data)
            
            # Check for manual trend override
            use_manual = getattr(item, 'use_manual_trend', False)
            trend_type = 'manual' if use_manual else item.trend_type
            manual_intercept = getattr(item, 'manual_intercept', None) if use_manual else None
            manual_slope = getattr(item, 'manual_slope', None) if use_manual else None
            
            # Apply trend
            forecast_values = apply_trend(
                base_value=base_value,
                trend_type=trend_type,
                growth_rate=item.trend_growth_rate,
                n_months=n_months,
                monthly=True,
                manual_intercept=manual_intercept,
                manual_slope=manual_slope
            )
            
            # Create forecast object
            forecast = LineItemForecast(
                key=key,
                name=item.line_item_name,
                category=item.category,
                statement_type=item.statement_type,
                values=forecast_values
            )
            
            # MC samples will be generated after all forecasts (to support correlations)
            # For now, just store the config
            
            line_item_forecasts[key] = forecast
            
            # Categorize for aggregation
            category_lower = item.category.lower()
            if 'revenue' in category_lower or 'sales' in category_lower or 'income' in category_lower:
                revenue_items.append(forecast)
            elif 'cogs' in category_lower or 'cost of' in category_lower or 'direct cost' in category_lower:
                cogs_items.append(forecast)
            elif 'opex' in category_lower or 'operating' in category_lower or 'expense' in category_lower or 'overhead' in category_lower:
                opex_items.append(forecast)
            else:
                other_items.append(forecast)
        
        # Aggregate by category
        total_revenue = self._sum_forecasts(revenue_items, n_months)
        total_cogs = self._sum_forecasts(cogs_items, n_months)
        total_opex = self._sum_forecasts(opex_items, n_months)
        
        # Calculate derived values
        gross_profit = total_revenue - total_cogs
        depreciation = total_opex * depreciation_pct
        ebit = gross_profit - total_opex
        
        # Interest and tax (simplified)
        # For more complex models, these would come from balance sheet
        interest_expense = np.zeros(n_months)  # Placeholder
        tax = np.maximum(ebit - interest_expense, 0) * tax_rate
        net_income = ebit - interest_expense - tax
        
        # Build MC aggregates if enabled
        mc_samples = None
        if run_monte_carlo:
            # Generate samples for each line item (with correlations if configured)
            mc_samples = self._generate_correlated_mc_samples(
                line_item_forecasts,
                revenue_items,
                cogs_items,
                opex_items,
                n_months,
                mc_iterations,
                tax_rate
            )
        
        return AggregatedForecast(
            total_revenue=total_revenue,
            total_cogs=total_cogs,
            gross_profit=gross_profit,
            total_opex=total_opex,
            ebit=ebit,
            depreciation=depreciation,
            interest_expense=interest_expense,
            tax=tax,
            net_income=net_income,
            line_items=line_item_forecasts,
            mc_enabled=run_monte_carlo,
            mc_samples=mc_samples
        )
    
    def _get_base_value(
        self,
        key: str,
        item: LineItemConfig,
        historical_data: Optional[pd.DataFrame]
    ) -> float:
        """Get base value for forecast (last actual or historical mean)."""
        # First, try historical data if provided
        if historical_data is not None and not historical_data.empty:
            # Try to find matching column
            for col in [item.line_item_name, key, key.replace('_', ' ')]:
                if col in historical_data.columns:
                    values = historical_data[col].dropna().values
                    if len(values) > 0:
                        return float(values[-1])  # Use last value
        
        # Fall back to stored historical mean
        return item.historical_mean
    
    def _sum_forecasts(
        self,
        forecasts: List[LineItemForecast],
        n_months: int
    ) -> np.ndarray:
        """Sum forecast values from multiple line items."""
        if not forecasts:
            return np.zeros(n_months)
        
        total = np.zeros(n_months)
        for f in forecasts:
            if len(f.values) == n_months:
                total += f.values
        
        return total
    
    def _generate_correlated_mc_samples(
        self,
        line_items: Dict[str, LineItemForecast],
        revenue_items: List[LineItemForecast],
        cogs_items: List[LineItemForecast],
        opex_items: List[LineItemForecast],
        n_months: int,
        n_samples: int,
        tax_rate: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate MC samples with correlation support.
        
        If correlations are configured, uses Cholesky decomposition for
        correlated sampling. Otherwise, independent sampling.
        """
        # Check if we have correlation config
        correlation_matrix = None
        if hasattr(self, 'correlation_config') and self.correlation_config:
            try:
                from components.correlation_config import CorrelationConfig
                if isinstance(self.correlation_config, dict):
                    corr_cfg = CorrelationConfig.from_dict(self.correlation_config)
                else:
                    corr_cfg = self.correlation_config
                
                mc_item_keys = [k for k, item in self.line_items.items() 
                               if item.use_distribution]
                
                if mc_item_keys and corr_cfg.correlations:
                    correlation_matrix = corr_cfg.get_correlation_matrix(mc_item_keys)
            except ImportError:
                pass
            except Exception:
                pass
        
        # Generate samples for each MC-enabled line item
        for key, forecast in line_items.items():
            item = self.line_items.get(key)
            if item and item.use_distribution:
                forecast.samples = apply_distribution(
                    values=forecast.values,
                    distribution_type=item.distribution_type,
                    cv=item.distribution_cv,
                    n_samples=n_samples
                )
        
        # TODO: Apply correlation structure using correlation_matrix
        # For now, aggregate independently (correlation to be added)
        
        return self._aggregate_mc_samples(
            line_items, revenue_items, cogs_items, opex_items,
            n_months, n_samples, tax_rate
        )
    
    def _aggregate_mc_samples(
        self,
        line_items: Dict[str, LineItemForecast],
        revenue_items: List[LineItemForecast],
        cogs_items: List[LineItemForecast],
        opex_items: List[LineItemForecast],
        n_months: int,
        n_samples: int,
        tax_rate: float
    ) -> Dict[str, np.ndarray]:
        """Aggregate MC samples from line items to category totals."""
        samples = {}
        
        # Revenue samples
        rev_samples = np.zeros((n_samples, n_months))
        for f in revenue_items:
            if f.samples is not None:
                rev_samples += f.samples
            else:
                rev_samples += f.values  # Use deterministic if no MC
        samples['total_revenue'] = rev_samples
        
        # COGS samples
        cogs_samples = np.zeros((n_samples, n_months))
        for f in cogs_items:
            if f.samples is not None:
                cogs_samples += f.samples
            else:
                cogs_samples += f.values
        samples['total_cogs'] = cogs_samples
        
        # OPEX samples
        opex_samples = np.zeros((n_samples, n_months))
        for f in opex_items:
            if f.samples is not None:
                opex_samples += f.samples
            else:
                opex_samples += f.values
        samples['total_opex'] = opex_samples
        
        # Derived
        samples['gross_profit'] = rev_samples - cogs_samples
        samples['ebit'] = samples['gross_profit'] - opex_samples
        samples['tax'] = np.maximum(samples['ebit'], 0) * tax_rate
        samples['net_income'] = samples['ebit'] - samples['tax']
        
        return samples


# =============================================================================
# INTEGRATION WITH MAIN FORECAST ENGINE
# =============================================================================

def run_line_item_forecast(
    db,
    scenario_id: str,
    user_id: str,
    n_months: int,
    start_date: datetime,
    historical_data: Optional[pd.DataFrame] = None,
    run_monte_carlo: bool = False,
    mc_iterations: int = 1000,
    assumptions: Optional[Dict] = None
) -> Optional[AggregatedForecast]:
    """
    Run a line-item level forecast using unified configuration.
    
    This is the main entry point for trend-based forecasting.
    
    Args:
        db: Database connector
        scenario_id: Scenario ID
        user_id: User ID
        n_months: Forecast horizon in months
        start_date: Forecast start date
        historical_data: Optional historical data DataFrame
        run_monte_carlo: Whether to run MC simulation
        mc_iterations: Number of MC iterations
        assumptions: Optional assumptions dict (for tax rate, etc.)
        
    Returns:
        AggregatedForecast or None if no config found
    """
    # Load unified config
    config = load_unified_config(db, scenario_id, user_id)
    
    if not config.line_items:
        return None
    
    # Load correlation config (Phase 3)
    correlation_config = None
    if assumptions and 'correlation_config' in assumptions:
        correlation_config = assumptions['correlation_config']
    else:
        try:
            from components.correlation_config import load_correlation_config
            corr_cfg = load_correlation_config(db, scenario_id, user_id)
            if corr_cfg.correlations:
                correlation_config = corr_cfg.to_dict()
        except ImportError:
            pass
        except Exception:
            pass
    
    # Get rates from assumptions
    tax_rate = 0.28
    interest_rate = 0.10
    depreciation_pct = 0.05
    
    if assumptions:
        tax_rate = assumptions.get('tax_rate', 28) / 100 if assumptions.get('tax_rate', 28) > 1 else assumptions.get('tax_rate', 0.28)
        interest_rate = assumptions.get('interest_rate', 10) / 100 if assumptions.get('interest_rate', 10) > 1 else assumptions.get('interest_rate', 0.10)
        depreciation_pct = assumptions.get('depreciation_pct', 5) / 100 if assumptions.get('depreciation_pct', 5) > 1 else assumptions.get('depreciation_pct', 0.05)
    
    # Create engine and run
    engine = LineItemForecastEngine(config, correlation_config=correlation_config)
    
    result = engine.run_forecast(
        n_months=n_months,
        start_date=start_date,
        historical_data=historical_data,
        run_monte_carlo=run_monte_carlo,
        mc_iterations=mc_iterations,
        tax_rate=tax_rate,
        interest_rate=interest_rate,
        depreciation_pct=depreciation_pct
    )
    
    return result


def convert_to_legacy_format(forecast: AggregatedForecast, start_date: datetime) -> Dict[str, Any]:
    """
    Convert AggregatedForecast to legacy format expected by UI.
    
    This allows gradual migration - new engine produces new format,
    but we convert to old format for compatibility.
    """
    n_months = len(forecast.total_revenue)
    
    # Build timeline
    timeline = []
    for i in range(n_months):
        date = start_date + relativedelta(months=i)
        timeline.append(date.strftime('%Y-%m'))
    
    # Build results dict
    results = {
        'success': True,
        'timeline': timeline,
        'total_months': n_months,
        'summary': {
            'total_revenue': float(np.sum(forecast.total_revenue)),
            'total_cogs': float(np.sum(forecast.total_cogs)),
            'total_gross_profit': float(np.sum(forecast.gross_profit)),
            'total_opex': float(np.sum(forecast.total_opex)),
            'total_ebit': float(np.sum(forecast.ebit)),
            'total_net_income': float(np.sum(forecast.net_income)),
        },
        'monthly_data': {
            'revenue': forecast.total_revenue.tolist(),
            'cogs': forecast.total_cogs.tolist(),
            'gross_profit': forecast.gross_profit.tolist(),
            'opex': forecast.total_opex.tolist(),
            'ebit': forecast.ebit.tolist(),
            'depreciation': forecast.depreciation.tolist(),
            'interest': forecast.interest_expense.tolist(),
            'tax': forecast.tax.tolist(),
            'net_income': forecast.net_income.tolist(),
        },
        # Line-item detail for drill-down
        'line_items': {}
    }
    
    # Add line-item detail
    for key, item_forecast in forecast.line_items.items():
        results['line_items'][key] = {
            'name': item_forecast.name,
            'category': item_forecast.category,
            'values': item_forecast.values.tolist()
        }
    
    # Add MC results if available
    if forecast.mc_enabled and forecast.mc_samples:
        mc_results = {}
        for metric, samples in forecast.mc_samples.items():
            # Calculate percentiles
            mc_results[metric] = {
                'mean': np.mean(samples, axis=0).tolist(),
                'p10': np.percentile(samples, 10, axis=0).tolist(),
                'p25': np.percentile(samples, 25, axis=0).tolist(),
                'p50': np.percentile(samples, 50, axis=0).tolist(),
                'p75': np.percentile(samples, 75, axis=0).tolist(),
                'p90': np.percentile(samples, 90, axis=0).tolist(),
            }
        results['monte_carlo'] = mc_results
    
    return results
