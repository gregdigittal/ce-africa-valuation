"""
Forecast Correlation Engine
===========================
Analyzes historical correlations between financial metrics and provides
flexible forecast configuration with multiple methods per element.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class ForecastMethod(Enum):
    """Available forecast methods for each element."""
    TREND_FIT = "trend_fit"  # Fitted trend function
    CORRELATION_FIXED = "correlation_fixed"  # Fixed % of another variable
    CORRELATION_CURVE = "correlation_curve"  # Curve-based % of another variable
    FIXED_VALUE = "fixed_value"  # Fixed absolute value
    PERIOD_OVERRIDES = "period_overrides"  # Period-by-period growth % overrides
    CALCULATED = "calculated"  # Calculated from other elements (read-only)


class CorrelationCurveType(Enum):
    """Types of correlation curves."""
    LINEAR = "linear"  # Fixed percentage
    LOGARITHMIC = "logarithmic"  # Logarithmic curve
    POLYNOMIAL = "polynomial"  # Polynomial curve
    EXPONENTIAL = "exponential"  # Exponential curve


@dataclass
class CorrelationAnalysis:
    """Results of correlation analysis between two metrics."""
    source_metric: str
    target_metric: str
    correlation_coefficient: float
    r_squared: float
    slope: float  # For linear: target = slope * source + intercept
    intercept: float
    avg_percentage: float  # Average target/source ratio
    std_percentage: float  # Std dev of percentage
    description: str = ""


@dataclass
class ForecastElementConfig:
    """Configuration for a single forecast element."""
    element_name: str
    display_name: str
    is_calculated: bool  # True if calculated from other elements
    calculation_formula: Optional[str] = None  # Formula if calculated
    method: ForecastMethod = ForecastMethod.TREND_FIT
    
    # Trend fit parameters
    trend_function_type: Optional[str] = None
    trend_parameters: Dict[str, float] = field(default_factory=dict)
    
    # Correlation parameters
    correlation_source: Optional[str] = None  # Which metric to correlate with
    correlation_type: CorrelationCurveType = CorrelationCurveType.LINEAR
    correlation_fixed_pct: Optional[float] = None  # Fixed percentage
    correlation_curve_params: Dict[str, float] = field(default_factory=dict)  # Curve parameters
    
    # Fixed value
    fixed_value: Optional[float] = None
    
    # Period overrides
    period_overrides: Dict[int, float] = field(default_factory=dict)  # {period_index: growth_pct}
    
    # Historical correlation analysis (suggested starting point)
    suggested_correlations: List[CorrelationAnalysis] = field(default_factory=list)


class ForecastCorrelationEngine:
    """Analyzes correlations and manages forecast element configurations."""
    
    def __init__(self):
        """Initialize the correlation engine."""
        pass
    
    def analyze_correlations(
        self,
        historical_data: pd.DataFrame,
        target_metric: str,
        source_metrics: List[str]
    ) -> List[CorrelationAnalysis]:
        """
        Analyze correlations between target metric and potential source metrics.
        
        Args:
            historical_data: DataFrame with historical financial data
            target_metric: Metric to forecast (e.g., 'cogs')
            source_metrics: List of potential source metrics (e.g., ['revenue', 'gross_profit'])
        
        Returns:
            List of correlation analyses, sorted by correlation strength
        """
        correlations = []
        
        if target_metric not in historical_data.columns:
            return correlations
        
        target_values = historical_data[target_metric].dropna()
        
        for source_metric in source_metrics:
            if source_metric not in historical_data.columns:
                continue
            
            source_values = historical_data[source_metric].dropna()
            
            # Align data
            common_index = target_values.index.intersection(source_values.index)
            if len(common_index) < 3:
                continue
            
            target_aligned = target_values.loc[common_index]
            source_aligned = source_values.loc[common_index]
            
            # Remove zeros and negatives for percentage calculations
            valid_mask = (source_aligned > 0) & (target_aligned >= 0)
            if valid_mask.sum() < 3:
                continue
            
            target_valid = target_aligned[valid_mask]
            source_valid = source_aligned[valid_mask]
            
            # Calculate correlation
            if len(target_valid) >= 3:
                corr_coef, p_value = stats.pearsonr(source_valid, target_valid)
                
                # Linear regression
                slope, intercept, r_value, p_value_reg, std_err = stats.linregress(
                    source_valid, target_valid
                )
                r_squared = r_value ** 2
                
                # Calculate average percentage
                percentages = (target_valid / source_valid) * 100
                avg_pct = percentages.mean()
                std_pct = percentages.std()
                
                # Create description
                if abs(corr_coef) > 0.7:
                    strength = "Strong"
                elif abs(corr_coef) > 0.4:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                
                description = f"{strength} correlation (r={corr_coef:.3f}), avg {target_metric}/{source_metric} = {avg_pct:.1f}%"
                
                analysis = CorrelationAnalysis(
                    source_metric=source_metric,
                    target_metric=target_metric,
                    correlation_coefficient=float(corr_coef),
                    r_squared=float(r_squared),
                    slope=float(slope),
                    intercept=float(intercept),
                    avg_percentage=float(avg_pct),
                    std_percentage=float(std_pct),
                    description=description
                )
                
                correlations.append(analysis)
        
        # Sort by absolute correlation coefficient
        correlations.sort(key=lambda x: abs(x.correlation_coefficient), reverse=True)
        
        return correlations
    
    def fit_correlation_curve(
        self,
        historical_data: pd.DataFrame,
        source_metric: str,
        target_metric: str,
        curve_type: CorrelationCurveType
    ) -> Dict[str, float]:
        """
        Fit a correlation curve between two metrics.
        
        Args:
            historical_data: Historical data DataFrame
            source_metric: Source metric (e.g., 'revenue')
            target_metric: Target metric (e.g., 'cogs')
            curve_type: Type of curve to fit
        
        Returns:
            Dictionary of curve parameters
        """
        if source_metric not in historical_data.columns or target_metric not in historical_data.columns:
            return {}
        
        source_values = historical_data[source_metric].dropna()
        target_values = historical_data[target_metric].dropna()
        
        # Align data
        common_index = source_values.index.intersection(target_values.index)
        if len(common_index) < 3:
            return {}
        
        source_aligned = source_values.loc[common_index].values
        target_aligned = target_values.loc[common_index].values
        
        # Remove zeros and negatives
        valid_mask = (source_aligned > 0) & (target_aligned >= 0)
        if valid_mask.sum() < 3:
            return {}
        
        x = source_aligned[valid_mask]
        y = target_aligned[valid_mask]
        
        try:
            if curve_type == CorrelationCurveType.LINEAR:
                # Linear: y = a * x + b
                coeffs = np.polyfit(x, y, 1)
                return {'a': float(coeffs[0]), 'b': float(coeffs[1])}
            
            elif curve_type == CorrelationCurveType.LOGARITHMIC:
                # Logarithmic: y = a * ln(x) + b
                log_x = np.log(x)
                coeffs = np.polyfit(log_x, y, 1)
                return {'a': float(coeffs[0]), 'b': float(coeffs[1])}
            
            elif curve_type == CorrelationCurveType.POLYNOMIAL:
                # Polynomial: y = a * x^2 + b * x + c
                coeffs = np.polyfit(x, y, 2)
                return {'a': float(coeffs[0]), 'b': float(coeffs[1]), 'c': float(coeffs[2])}
            
            elif curve_type == CorrelationCurveType.EXPONENTIAL:
                # Exponential: y = a * exp(b * x)
                # Ensure positive y values
                y_positive = np.maximum(y, 0.01)
                log_y = np.log(y_positive)
                coeffs = np.polyfit(x, log_y, 1)
                return {'a': float(np.exp(coeffs[1])), 'b': float(coeffs[0])}
            
        except Exception as e:
            st.warning(f"Curve fitting failed: {e}")
            return {}
        
        return {}
    
    def calculate_correlated_value(
        self,
        source_value: float,
        correlation_type: CorrelationCurveType,
        correlation_params: Dict[str, float],
        fixed_pct: Optional[float] = None
    ) -> float:
        """
        Calculate target value based on correlation with source value.
        
        Args:
            source_value: Source metric value
            correlation_type: Type of correlation curve
            correlation_params: Curve parameters
            fixed_pct: Fixed percentage (overrides curve if provided)
        
        Returns:
            Calculated target value
        """
        if fixed_pct is not None:
            return source_value * (fixed_pct / 100)
        
        if correlation_type == CorrelationCurveType.LINEAR:
            a = correlation_params.get('a', 0)
            b = correlation_params.get('b', 0)
            return a * source_value + b
        
        elif correlation_type == CorrelationCurveType.LOGARITHMIC:
            a = correlation_params.get('a', 0)
            b = correlation_params.get('b', 0)
            return a * np.log(max(source_value, 0.01)) + b
        
        elif correlation_type == CorrelationCurveType.POLYNOMIAL:
            a = correlation_params.get('a', 0)
            b = correlation_params.get('b', 0)
            c = correlation_params.get('c', 0)
            return a * (source_value ** 2) + b * source_value + c
        
        elif correlation_type == CorrelationCurveType.EXPONENTIAL:
            a = correlation_params.get('a', 0)
            b = correlation_params.get('b', 0)
            return a * np.exp(b * source_value)
        
        return 0.0


# =============================================================================
# FORECAST ELEMENT DEFINITIONS
# =============================================================================

# Define which elements are calculated vs inputs
FORECAST_ELEMENTS = {
    'revenue': {
        'display_name': 'Revenue',
        'is_calculated': False,
        'calculation_formula': None,
        'suggested_correlations': ['gross_profit', 'ebit']  # Can correlate with these
    },
    'cogs': {
        'display_name': 'Cost of Goods Sold',
        'is_calculated': False,
        'calculation_formula': None,
        'suggested_correlations': ['revenue', 'gross_profit']
    },
    'gross_profit': {
        'display_name': 'Gross Profit',
        'is_calculated': True,
        'calculation_formula': 'revenue - cogs',
        'suggested_correlations': ['revenue', 'cogs']
    },
    'opex': {
        'display_name': 'Operating Expenses',
        'is_calculated': False,
        'calculation_formula': None,
        'suggested_correlations': ['revenue', 'gross_profit', 'ebit']
    },
    'depreciation': {
        'display_name': 'Depreciation',
        'is_calculated': False,
        'calculation_formula': None,
        'suggested_correlations': ['revenue', 'opex']
    },
    'ebit': {
        'display_name': 'EBIT',
        'is_calculated': True,
        'calculation_formula': 'gross_profit - opex - depreciation',
        'suggested_correlations': ['revenue', 'gross_profit', 'opex']
    },
    'interest_expense': {
        'display_name': 'Interest Expense',
        'is_calculated': True,
        'calculation_formula': 'calculated_from_balance_sheet',  # Calculated from debt/cash balances and interest rates
        'suggested_correlations': []  # Not applicable - this is calculated
    },
    'tax': {
        'display_name': 'Tax',
        'is_calculated': True,
        'calculation_formula': 'max(ebt * tax_rate, 0)',  # Tax = EBT × Tax Rate (only if EBT > 0)
        'suggested_correlations': []  # Not applicable - this is calculated
    },
    'net_profit': {
        'display_name': 'Net Profit',
        'is_calculated': True,
        'calculation_formula': 'ebit - interest_expense - tax',
        'suggested_correlations': ['ebit', 'revenue']
    }
}


def get_calculated_dependencies(element_name: str) -> List[str]:
    """Get list of elements that this calculated element depends on."""
    element_def = FORECAST_ELEMENTS.get(element_name, {})
    formula = element_def.get('calculation_formula')
    if not formula:
        return []
    
    # Simple parsing - extract variable names from formula
    dependencies = []
    for dep_name in FORECAST_ELEMENTS.keys():
        if dep_name in formula:
            dependencies.append(dep_name)
    
    return dependencies


def validate_forecast_config(configs: Dict[str, ForecastElementConfig]) -> Tuple[bool, List[str]]:
    """
    Validate forecast configurations to detect conflicts.
    
    Args:
        configs: Dictionary of element_name -> ForecastElementConfig
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    for element_name, config in configs.items():
        element_def = FORECAST_ELEMENTS.get(element_name, {})
        
        # Skip validation for calculated elements - they're auto-calculated
        if element_def.get('is_calculated', False):
            # Auto-correct: Force calculated elements to use CALCULATED method
            if config.method != ForecastMethod.CALCULATED:
                config.method = ForecastMethod.CALCULATED
            continue
        
        # Check: Circular correlation dependencies
        if config.method in [ForecastMethod.CORRELATION_FIXED, ForecastMethod.CORRELATION_CURVE]:
            if config.correlation_source:
                # Check if source depends on this element
                source_config = configs.get(config.correlation_source)
                if source_config:
                    if source_config.correlation_source == element_name:
                        errors.append(
                            f"Circular correlation: {config.display_name} correlates with "
                            f"{source_config.display_name}, which correlates with {config.display_name}"
                        )
                    
                    # Check deeper dependencies
                    checked = set()
                    to_check = [config.correlation_source]
                    while to_check:
                        current = to_check.pop()
                        if current in checked:
                            if current == element_name:
                                errors.append(
                                    f"Circular correlation chain involving {config.display_name}"
                                )
                                break
                            continue
                        checked.add(current)
                        
                        current_config = configs.get(current)
                        if current_config and current_config.correlation_source:
                            to_check.append(current_config.correlation_source)
        
        # Check: Period overrides must have valid growth percentages
        if config.method == ForecastMethod.PERIOD_OVERRIDES:
            for period, growth_pct in config.period_overrides.items():
                if not isinstance(growth_pct, (int, float)):
                    errors.append(
                        f"{config.display_name}: Invalid growth percentage for period {period}"
                    )
    
    return len(errors) == 0, errors
