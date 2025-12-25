"""
AI Assumptions Engine
======================
Date: December 14, 2025
Version: 1.0 - Sprint 14 Enhancement

This component transforms historical data into forecast assumptions by:
1. Analyzing historical trends and patterns
2. Fitting probability distributions to historical data
3. Proposing static and stochastic assumptions
4. Allowing user preview, adjustment, and acceptance
5. Saving assumptions for use in Forecast and Manufacturing Strategy

WORKFLOW INTEGRATION:
- Required step after Setup, before Forecast
- Outputs feed into: Forecast Section, Manufacturing Strategy, MC Simulation
- Assumptions saved per scenario

FEATURES:
1. Historical Data Analysis
2. Distribution Fitting (Normal, Lognormal, Triangular, Beta, Uniform)
3. Visual Distribution Preview with Adjustment
4. Static vs Stochastic Assumption Selection
5. Save Assumptions to Database
6. Export/Import Assumptions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import traceback

# Supabase helpers (avoid silent 1000-row truncation on large selects)
try:
    from supabase_pagination import fetch_all_rows
except Exception:
    fetch_all_rows = None

# Import trend forecast analyzer
try:
    from components.trend_forecast_analyzer import render_trend_forecast_ui
    TREND_ANALYZER_AVAILABLE = True
except ImportError:
    TREND_ANALYZER_AVAILABLE = False
    def render_trend_forecast_ui(*args, **kwargs):
        st.error("Trend Forecast Analyzer not available. Please ensure trend_forecast_analyzer.py is installed.")

# =============================================================================
# JSON SERIALIZATION HELPER (Sprint 19 - Fix numpy type serialization)
# =============================================================================

def convert_to_serializable(obj):
    """
    Convert numpy types and other non-serializable types to JSON-serializable Python types.
    
    Args:
        obj: Object to convert (can be dict, list, numpy type, etc.)
        
    Returns:
        JSON-serializable version of the object
    """
    # Handle numpy integer types
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.int_)):
        return int(obj)
    # Handle numpy float types (np.float_ removed in NumPy 2.0)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj) if np.isfinite(obj) else None
    # Handle numpy scalar float types (for NumPy 2.0+ compatibility)
    elif isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    # Handle Python float with infinities/NaN
    elif isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    # Handle numpy boolean types
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle pandas Series
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    # Handle dictionaries (recursive)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    # Handle lists and tuples (recursive)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    # Handle datetime types
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    # Handle Enum types
    elif isinstance(obj, Enum):
        return obj.value
    else:
        # Handle pandas NaN (check safely to avoid array ambiguity)
        try:
            is_na = pd.isna(obj)
            # Only use result if it's a scalar boolean
            if isinstance(is_na, (bool, np.bool_)) and is_na:
                return None
        except (ValueError, TypeError):
            pass
        return obj

# Import standardized UI components (Sprint 19)
# Define fallback functions first to ensure they're always available
def _fallback_show_success(msg, details=None, icon="✅"):
    """Fallback success message function."""
    full_msg = f"{icon} **{msg}**"
    if details:
        full_msg += f"\n\n{details}"
    st.success(full_msg)

def _fallback_show_error(msg, details=None, icon="❌"):
    """Fallback error message function."""
    full_msg = f"{icon} **{msg}**"
    if details:
        full_msg += f"\n\n{details}"
    st.error(full_msg)

def _fallback_show_warning(msg, details=None, icon="⚠️"):
    """Fallback warning message function."""
    full_msg = f"{icon} **{msg}**"
    if details:
        full_msg += f"\n\n{details}"
    st.warning(full_msg)

def _fallback_show_info(msg, details=None, icon="ℹ️"):
    """Fallback info message function."""
    full_msg = f"{icon} **{msg}**"
    if details:
        full_msg += f"\n\n{details}"
    st.info(full_msg)

def _fallback_show_loading(msg="Loading...", key=None):
    """Fallback loading spinner function."""
    return st.spinner(msg)

def _fallback_show_progress(current, total, message=None, key=None):
    """Fallback progress bar function."""
    progress = current / total if total > 0 else 0
    if message:
        return st.progress(progress, text=message)
    return st.progress(progress)

# Try to import from ui_components, use fallbacks if import fails
try:
    from components.ui_components import (
        show_success, show_error, show_warning, show_info, show_loading, show_progress
    )
    # Verify functions are actually callable
    if not all(callable(f) for f in [show_success, show_error, show_warning, show_info, show_loading, show_progress]):
        raise AttributeError("Imported functions are not callable")
except (ImportError, AttributeError, ModuleNotFoundError, NameError) as e:
    # Use fallback functions if import fails
    show_success = _fallback_show_success
    show_error = _fallback_show_error
    show_warning = _fallback_show_warning
    show_info = _fallback_show_info
    show_loading = _fallback_show_loading
    show_progress = _fallback_show_progress
except Exception:
    # Ultimate fallback - use fallback functions
    show_success = _fallback_show_success
    show_error = _fallback_show_error
    show_warning = _fallback_show_warning
    show_info = _fallback_show_info
    show_loading = _fallback_show_loading
    show_progress = _fallback_show_progress

# =============================================================================
# COLOR CONSTANTS
# =============================================================================
GOLD = "#D4A537"
GOLD_LIGHT = "rgba(212, 165, 55, 0.1)"
GREEN = "#10b981"
RED = "#ef4444"
BLUE = "#3b82f6"
PURPLE = "#8b5cf6"
ORANGE = "#f59e0b"
TEXT_MUTED = "#888888"
TEXT_WHITE = "#FFFFFF"
BORDER_COLOR = "#404040"
DARK_BG = "#1E1E1E"


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class DistributionType(Enum):
    """Supported probability distributions."""
    STATIC = "static"  # Fixed value (historical average)
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    TRIANGULAR = "triangular"
    BETA = "beta"
    UNIFORM = "uniform"
    PERT = "pert"  # Modified triangular


class AssumptionCategory(Enum):
    """Categories of assumptions."""
    REVENUE = "revenue"
    COST = "cost"
    GROWTH = "growth"
    MARGIN = "margin"
    VOLUME = "volume"
    PRICE = "price"
    MANUFACTURING = "manufacturing"


@dataclass
class DistributionParams:
    """Parameters for a probability distribution."""
    distribution_type: str = "static"
    
    # Static
    static_value: float = 0.0
    
    # Normal / Lognormal
    mean: float = 0.0
    std: float = 0.0
    
    # Triangular / PERT
    min_val: float = 0.0
    mode_val: float = 0.0
    max_val: float = 0.0
    
    # Beta
    alpha: float = 2.0
    beta: float = 2.0
    scale: float = 1.0
    loc: float = 0.0
    
    # Uniform
    low: float = 0.0
    high: float = 1.0
    
    # Fit quality
    fit_score: float = 0.0  # R² or KS statistic
    
    def to_dict(self) -> dict:
        """Convert DistributionParams to dictionary, ensuring JSON serializability."""
        d = asdict(self)
        # Convert numpy types to native Python types
        return convert_to_serializable(d)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DistributionParams':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Assumption:
    """A single forecast assumption."""
    id: str
    name: str
    display_name: str
    category: str
    unit: str = ""
    
    # Historical analysis
    historical_mean: float = 0.0
    historical_std: float = 0.0
    historical_min: float = 0.0
    historical_max: float = 0.0
    historical_median: float = 0.0
    historical_trend: str = "stable"  # increasing, decreasing, stable
    historical_cagr: float = 0.0
    data_points: int = 0
    
    # AI proposed distribution
    proposed_distribution: DistributionParams = field(default_factory=DistributionParams)
    
    # User selection
    use_distribution: bool = False  # False = use static, True = use distribution
    user_distribution: DistributionParams = field(default_factory=DistributionParams)
    user_accepted: bool = False
    user_modified: bool = False
    
    # Final value (for static) or distribution (for stochastic)
    final_static_value: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert Assumption to dictionary, ensuring JSON serializability."""
        d = asdict(self)
        d['proposed_distribution'] = self.proposed_distribution.to_dict()
        d['user_distribution'] = self.user_distribution.to_dict()
        # Convert numpy types to native Python types
        return convert_to_serializable(d)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Assumption':
        data = data.copy()
        if 'proposed_distribution' in data:
            data['proposed_distribution'] = DistributionParams.from_dict(data['proposed_distribution'])
        if 'user_distribution' in data:
            data['user_distribution'] = DistributionParams.from_dict(data['user_distribution'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ManufacturingAssumption:
    """Manufacturing-specific assumption for a part or category."""
    part_id: str
    part_name: str
    category: str = "liner"
    
    # Historical data
    historical_sell_price: float = 0.0
    historical_cogs: float = 0.0
    historical_margin_pct: float = 0.0
    historical_volume: int = 0
    
    # AI proposed
    proposed_manufacture_pct: float = 0.0  # % to manufacture
    proposed_mfg_cost_pct: float = 75.0  # Mfg cost as % of buy
    rationale: str = ""
    
    # User selection
    user_manufacture_pct: float = 0.0
    user_mfg_cost_pct: float = 75.0
    user_accepted: bool = False
    
    def to_dict(self) -> dict:
        """Convert DistributionParams to dictionary, ensuring JSON serializability."""
        d = asdict(self)
        # Convert numpy types to native Python types
        return convert_to_serializable(d)


@dataclass
class AssumptionsSet:
    """Complete set of assumptions for a scenario."""
    scenario_id: str
    created_at: str = ""
    updated_at: str = ""
    
    # Status
    analysis_complete: bool = False
    assumptions_saved: bool = False
    
    # Core assumptions
    assumptions: Dict[str, Assumption] = field(default_factory=dict)
    
    # Detailed line item assumptions (NEW)
    line_item_assumptions: Dict[str, Dict[str, Assumption]] = field(default_factory=dict)
    # Structure: {'income_statement': {'line_item_name': Assumption}, 'balance_sheet': {...}, 'cash_flow': {...}}
    
    # Manufacturing assumptions
    manufacturing_assumptions: Dict[str, ManufacturingAssumption] = field(default_factory=dict)
    manufacturing_mode: str = "average"  # 'average' or 'part_level'
    
    # Summary stats
    historical_periods: int = 0
    data_quality_score: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert AssumptionsSet to dictionary, ensuring JSON serializability."""
        result = {
            'scenario_id': self.scenario_id,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'analysis_complete': self.analysis_complete,
            'assumptions_saved': self.assumptions_saved,
            'assumptions': {k: v.to_dict() for k, v in self.assumptions.items()},
            'line_item_assumptions': {
                stmt_type: {k: v.to_dict() for k, v in stmt_assumptions.items()}
                for stmt_type, stmt_assumptions in self.line_item_assumptions.items()
            },
            'manufacturing_assumptions': {k: v.to_dict() for k, v in self.manufacturing_assumptions.items()},
            'manufacturing_mode': self.manufacturing_mode,
            'historical_periods': int(self.historical_periods) if isinstance(self.historical_periods, (np.integer, np.int64, np.int32)) else self.historical_periods,
            'data_quality_score': float(self.data_quality_score) if isinstance(self.data_quality_score, (np.floating, np.float64, np.float32)) else self.data_quality_score
        }
        # Convert all numpy types to native Python types
        return convert_to_serializable(result)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AssumptionsSet':
        obj = cls(
            scenario_id=data.get('scenario_id', ''),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            analysis_complete=data.get('analysis_complete', False),
            assumptions_saved=data.get('assumptions_saved', False),
            manufacturing_mode=data.get('manufacturing_mode', 'average'),
            historical_periods=data.get('historical_periods', 0),
            data_quality_score=data.get('data_quality_score', 0.0)
        )
        
        for k, v in data.get('assumptions', {}).items():
            obj.assumptions[k] = Assumption.from_dict(v)
        
        for k, v in data.get('manufacturing_assumptions', {}).items():
            obj.manufacturing_assumptions[k] = ManufacturingAssumption(**v)
        
        # Load line item assumptions
        for stmt_type, stmt_assumptions in data.get('line_item_assumptions', {}).items():
            obj.line_item_assumptions[stmt_type] = {
                k: Assumption.from_dict(v) for k, v in stmt_assumptions.items()
            }
        
        return obj


# =============================================================================
# DISTRIBUTION FITTING ENGINE
# =============================================================================

class DistributionFitter:
    """Fits probability distributions to historical data."""
    
    @staticmethod
    def fit_normal(data: np.ndarray) -> Tuple[DistributionParams, float]:
        """Fit normal distribution and return params with fit score."""
        mean, std = np.mean(data), np.std(data)
        params = DistributionParams(
            distribution_type="normal",
            mean=mean,
            std=std,
            static_value=mean
        )
        
        # Calculate fit score using Kolmogorov-Smirnov test
        try:
            ks_stat, p_value = stats.kstest(data, 'norm', args=(mean, std))
            params.fit_score = 1 - ks_stat  # Higher is better
        except:
            params.fit_score = 0.5
        
        return params, params.fit_score
    
    @staticmethod
    def fit_lognormal(data: np.ndarray) -> Tuple[DistributionParams, float]:
        """Fit lognormal distribution."""
        # Filter out non-positive values
        positive_data = data[data > 0]
        if len(positive_data) < 3:
            return DistributionParams(distribution_type="lognormal"), 0.0
        
        log_data = np.log(positive_data)
        mean, std = np.mean(log_data), np.std(log_data)
        
        params = DistributionParams(
            distribution_type="lognormal",
            mean=mean,  # These are log-space parameters
            std=std,
            static_value=np.exp(mean)
        )
        
        try:
            ks_stat, p_value = stats.kstest(positive_data, 'lognorm', args=(std, 0, np.exp(mean)))
            params.fit_score = 1 - ks_stat
        except:
            params.fit_score = 0.5
        
        return params, params.fit_score
    
    @staticmethod
    def fit_triangular(data: np.ndarray) -> Tuple[DistributionParams, float]:
        """Fit triangular distribution using min, mode, max."""
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Estimate mode using kernel density
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_grid = np.linspace(min_val, max_val, 100)
            mode_val = x_grid[np.argmax(kde(x_grid))]
        except:
            mode_val = np.median(data)
        
        params = DistributionParams(
            distribution_type="triangular",
            min_val=min_val,
            mode_val=mode_val,
            max_val=max_val,
            static_value=mode_val
        )
        
        # Calculate fit score
        try:
            # Normalize to [0,1] for scipy triangular
            c = (mode_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            loc = min_val
            scale = max_val - min_val
            ks_stat, _ = stats.kstest(data, 'triang', args=(c, loc, scale))
            params.fit_score = 1 - ks_stat
        except:
            params.fit_score = 0.5
        
        return params, params.fit_score
    
    @staticmethod
    def fit_beta(data: np.ndarray) -> Tuple[DistributionParams, float]:
        """Fit beta distribution (useful for percentages/ratios)."""
        # Normalize data to [0, 1]
        data_min, data_max = np.min(data), np.max(data)
        if data_max == data_min:
            return DistributionParams(distribution_type="beta"), 0.0
        
        normalized = (data - data_min) / (data_max - data_min)
        
        # Clip to avoid 0 and 1 exactly
        normalized = np.clip(normalized, 0.001, 0.999)
        
        try:
            alpha, beta_param, loc, scale = stats.beta.fit(normalized, floc=0, fscale=1)
            
            params = DistributionParams(
                distribution_type="beta",
                alpha=alpha,
                beta=beta_param,
                loc=data_min,
                scale=data_max - data_min,
                static_value=np.mean(data)
            )
            
            ks_stat, _ = stats.kstest(normalized, 'beta', args=(alpha, beta_param))
            params.fit_score = 1 - ks_stat
        except:
            params = DistributionParams(distribution_type="beta")
            params.fit_score = 0.0
        
        return params, params.fit_score
    
    @staticmethod
    def fit_uniform(data: np.ndarray) -> Tuple[DistributionParams, float]:
        """Fit uniform distribution."""
        low = np.min(data)
        high = np.max(data)
        
        params = DistributionParams(
            distribution_type="uniform",
            low=low,
            high=high,
            static_value=(low + high) / 2
        )
        
        try:
            ks_stat, _ = stats.kstest(data, 'uniform', args=(low, high - low))
            params.fit_score = 1 - ks_stat
        except:
            params.fit_score = 0.5
        
        return params, params.fit_score
    
    @classmethod
    def fit_best(cls, data: np.ndarray, metric_type: str = "general") -> Tuple[DistributionParams, str]:
        """
        Fit multiple distributions and return the best one.
        
        Args:
            data: Historical data array
            metric_type: Type of metric to help select appropriate distributions
                - 'growth': Growth rates (can be negative) -> Normal, Triangular
                - 'ratio': Ratios/percentages (0-1 or 0-100) -> Beta, Triangular
                - 'price': Prices (always positive) -> Lognormal, Triangular
                - 'volume': Counts (positive integers) -> Lognormal, Normal
                - 'general': Try all
        """
        if len(data) < 3:
            return DistributionParams(distribution_type="static", static_value=np.mean(data) if len(data) > 0 else 0), "Insufficient data"
        
        data = np.array(data, dtype=float)
        data = data[~np.isnan(data)]
        
        if len(data) < 3:
            return DistributionParams(distribution_type="static", static_value=0), "Insufficient valid data"
        
        candidates = []
        
        # Select distributions based on metric type
        if metric_type in ['growth', 'general']:
            candidates.append(('normal', cls.fit_normal(data)))
            candidates.append(('triangular', cls.fit_triangular(data)))
        
        if metric_type in ['ratio', 'general']:
            candidates.append(('beta', cls.fit_beta(data)))
            candidates.append(('triangular', cls.fit_triangular(data)))
        
        if metric_type in ['price', 'volume', 'general']:
            candidates.append(('lognormal', cls.fit_lognormal(data)))
            candidates.append(('normal', cls.fit_normal(data)))
            candidates.append(('triangular', cls.fit_triangular(data)))
        
        candidates.append(('uniform', cls.fit_uniform(data)))
        
        # Find best fit
        best_name = 'normal'
        best_params = DistributionParams(distribution_type="normal", mean=np.mean(data), std=np.std(data))
        best_score = 0
        
        for name, (params, score) in candidates:
            if score > best_score:
                best_score = score
                best_params = params
                best_name = name
        
        best_params.fit_score = best_score
        
        return best_params, best_name


# =============================================================================
# HISTORICAL ANALYSIS ENGINE
# =============================================================================

class HistoricalAnalyzer:
    """Analyzes historical data to derive assumptions."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.assumptions: Dict[str, Assumption] = {}
    
    def analyze_metric(self, column: str, display_name: str, category: str, 
                      unit: str = "", metric_type: str = "general", values: np.ndarray = None) -> Optional[Assumption]:
        """
        Analyze a single metric and create an assumption.
        
        Args:
            column: Column name in self.data (if None, values must be provided)
            display_name: Display name for the assumption
            category: Category (revenue, cost, margin, etc.)
            unit: Unit (R, %, etc.)
            metric_type: Type for distribution fitting (price, cost, ratio, growth)
            values: Optional numpy array of values (if provided, column is ignored)
        """
        # Use provided values or extract from column
        if values is None:
            if column is None or column not in self.data.columns:
                return None
            values = self.data[column].dropna().values
        else:
            values = np.array(values)
        
        if len(values) < 3:
            return None
        
        # Calculate historical statistics
        assumption = Assumption(
            id=column,
            name=column,
            display_name=display_name,
            category=category,
            unit=unit,
            historical_mean=float(np.mean(values)),
            historical_std=float(np.std(values)),
            historical_min=float(np.min(values)),
            historical_max=float(np.max(values)),
            historical_median=float(np.median(values)),
            data_points=len(values)
        )
        
        # Calculate trend
        if len(values) >= 3:
            x = np.arange(len(values))
            slope, _, r_value, _, _ = stats.linregress(x, values)
            
            # Determine trend direction
            if r_value ** 2 > 0.3:  # Meaningful trend
                if slope > 0:
                    assumption.historical_trend = "increasing"
                else:
                    assumption.historical_trend = "decreasing"
            else:
                assumption.historical_trend = "stable"
            
            # Calculate CAGR for applicable metrics
            if values[0] > 0 and values[-1] > 0 and len(values) >= 12:
                years = len(values) / 12
                assumption.historical_cagr = (values[-1] / values[0]) ** (1 / years) - 1
        
        # Fit distribution
        best_params, dist_name = DistributionFitter.fit_best(values, metric_type)
        assumption.proposed_distribution = best_params
        
        # Set default static value
        assumption.final_static_value = assumption.historical_mean
        
        # Copy proposed to user (default acceptance)
        assumption.user_distribution = DistributionParams.from_dict(best_params.to_dict())
        
        self.assumptions[column] = assumption
        return assumption
    
    def analyze_all_financials(self) -> Dict[str, Assumption]:
        """Analyze all standard financial metrics.
        
        FIXED: Removed duplicates and calculated elements.
        - Only analyzes canonical names (total_revenue, total_cogs, total_opex)
        - Excludes calculated elements (gross_profit, ebit, net_income, interest_expense, tax)
        - COGS and OPEX analyzed as percentages of revenue (best practice)
        """
        from components.forecast_correlation_engine import FORECAST_ELEMENTS
        
        # Revenue metrics - use canonical 'total_revenue' only
        if 'total_revenue' in self.data.columns:
            self.analyze_metric('total_revenue', 'Total Revenue', 'revenue', 'R', 'price')
        elif 'revenue' in self.data.columns:
            # Fallback to 'revenue' if 'total_revenue' not available
            self.analyze_metric('revenue', 'Total Revenue', 'revenue', 'R', 'price')
        
        # Cost metrics - analyze as % of revenue (best practice financial modeling)
        # COGS as % of Revenue
        if 'total_revenue' in self.data.columns and 'total_cogs' in self.data.columns:
            rev = self.data['total_revenue']
            cogs = self.data['total_cogs']
            if len(rev) > 0 and rev.sum() > 0:
                cogs_pct = (cogs / rev * 100).dropna()
                if len(cogs_pct) >= 3:
                    self.analyze_derived_metric(
                        cogs_pct.values, 
                        'cogs_pct_of_revenue', 
                        'COGS % of Revenue', 
                        'cost', 
                        '%', 
                        'ratio'
                    )
        elif 'total_cogs' in self.data.columns:
            # Fallback: analyze as absolute if revenue not available
            self.analyze_metric('total_cogs', 'COGS', 'cost', 'R', 'price')
        elif 'cogs' in self.data.columns:
            self.analyze_metric('cogs', 'COGS', 'cost', 'R', 'price')
        
        # OPEX as % of Revenue
        if 'total_revenue' in self.data.columns and 'total_opex' in self.data.columns:
            rev = self.data['total_revenue']
            opex = self.data['total_opex']
            if len(rev) > 0 and rev.sum() > 0:
                opex_pct = (opex / rev * 100).dropna()
                if len(opex_pct) >= 3:
                    self.analyze_derived_metric(
                        opex_pct.values, 
                        'opex_pct_of_revenue', 
                        'OPEX % of Revenue', 
                        'cost', 
                        '%', 
                        'ratio'
                    )
        elif 'total_opex' in self.data.columns:
            # Fallback: analyze as absolute if revenue not available
            self.analyze_metric('total_opex', 'OPEX', 'cost', 'R', 'price')
        elif 'opex' in self.data.columns:
            self.analyze_metric('opex', 'OPEX', 'cost', 'R', 'price')
        
        # Depreciation (not calculated, can be configured)
        if 'depreciation' in self.data.columns:
            self.analyze_metric('depreciation', 'Depreciation', 'cost', 'R', 'price')
        
        # DO NOT analyze calculated elements:
        # - gross_profit (calculated: revenue - cogs)
        # - ebit (calculated: gross_profit - opex - depreciation)
        # - ebitda (calculated: ebit + depreciation)
        # - interest_expense (calculated from balance sheet)
        # - tax (calculated: max(ebt * tax_rate, 0))
        # - net_income (calculated: ebit - interest_expense - tax)
        
        # Calculate derived metrics if raw data available
        if 'total_revenue' in self.data.columns and 'total_cogs' in self.data.columns:
            rev = self.data['total_revenue']
            cogs = self.data['total_cogs']
            if len(rev) > 0 and rev.sum() > 0:
                margin_pct = ((rev - cogs) / rev * 100).dropna()
                if len(margin_pct) >= 3:
                    self.analyze_derived_metric(margin_pct.values, 'gross_margin_pct', 
                                               'Gross Margin %', 'margin', '%', 'ratio')
        
        # Revenue growth rates
        if 'total_revenue' in self.data.columns:
            rev = self.data['total_revenue'].values
            if len(rev) > 1:
                growth = np.diff(rev) / rev[:-1] * 100
                growth = growth[~np.isnan(growth) & ~np.isinf(growth)]
                if len(growth) >= 3:
                    self.analyze_derived_metric(growth, 'revenue_growth_pct',
                                               'Revenue Growth %', 'growth', '%', 'growth')
        
        return self.assumptions
    
    def analyze_detailed_line_items(self, db, scenario_id: str, user_id: str = None) -> Dict[str, Dict[str, Assumption]]:
        """
        Analyze detailed line items from historical data.
        
        Returns:
            Dict with structure: {'income_statement': {'line_item_name': Assumption}, ...}
        """
        line_item_assumptions = {}
        
        for statement_type in ['income_statement', 'balance_sheet', 'cash_flow']:
            # Load detailed line items (pass user_id)
            line_items_df = load_detailed_line_items(db, scenario_id, statement_type, user_id)
            
            if line_items_df.empty:
                continue
            
            line_item_assumptions[statement_type] = {}
            
            # Group by line_item_name and analyze each
            for line_item_name, group in line_items_df.groupby('line_item_name'):
                # Get time series of amounts
                if 'period_date' in group.columns and 'amount' in group.columns:
                    series = group.set_index('period_date')['amount'].sort_index()
                    values = series.dropna().values
                    
                    if len(values) < 3:
                        continue
                    
                    # Get category (should be same for all rows of same line item)
                    category = group['category'].iloc[0] if 'category' in group.columns else 'Other'
                    
                    # Determine metric type based on category and amount sign
                    if category in ['Revenue', 'Other Income', 'Finance Income']:
                        metric_type = 'price'  # Positive values
                    elif category in ['Operating Expenses', 'Cost of Sales', 'Finance Costs', 'Taxation']:
                        metric_type = 'cost'  # Negative values (expenses)
                    else:
                        metric_type = 'price'  # Default
                    
                    # Create assumption ID (unique per statement type)
                    assumption_id = f"{statement_type}_{line_item_name}"
                    
                    # Analyze this line item
                    assumption = self.analyze_metric(
                        column=None,  # Not from self.data, so pass None
                        display_name=f"{line_item_name} ({statement_type.replace('_', ' ').title()})",
                        category=category,
                        unit='R',
                        metric_type=metric_type,
                        values=values  # Pass values directly
                    )
                    
                    if assumption:
                        assumption.id = assumption_id
                        assumption.name = line_item_name
                        line_item_assumptions[statement_type][line_item_name] = assumption
        
        return line_item_assumptions
    
    def analyze_detailed_line_items(self, db, scenario_id: str, user_id: str = None) -> Dict[str, Dict[str, Assumption]]:
        """
        Analyze detailed line items from historical data.
        
        Returns:
            Dict with structure: {'income_statement': {'line_item_name': Assumption}, ...}
        """
        line_item_assumptions = {}
        
        for statement_type in ['income_statement', 'balance_sheet', 'cash_flow']:
            # Load detailed line items (pass user_id)
            line_items_df = load_detailed_line_items(db, scenario_id, statement_type, user_id)
            
            if line_items_df.empty:
                continue
            
            line_item_assumptions[statement_type] = {}
            
            # Group by line_item_name and analyze each
            for line_item_name, group in line_items_df.groupby('line_item_name'):
                # Get time series of amounts
                if 'period_date' in group.columns and 'amount' in group.columns:
                    series = group.set_index('period_date')['amount'].sort_index()
                    values = series.dropna().values
                    
                    if len(values) < 3:
                        continue
                    
                    # Get category (should be same for all rows of same line item)
                    category = group['category'].iloc[0] if 'category' in group.columns else 'Other'
                    
                    # Determine metric type based on category and amount sign
                    if category in ['Revenue', 'Other Income', 'Finance Income']:
                        metric_type = 'price'  # Positive values
                    elif category in ['Operating Expenses', 'Cost of Sales', 'Finance Costs', 'Taxation']:
                        metric_type = 'cost'  # Negative values (expenses)
                    else:
                        metric_type = 'price'  # Default
                    
                    # Create assumption ID (unique per statement type)
                    assumption_id = f"{statement_type}_{line_item_name}"
                    
                    # Analyze this line item
                    assumption = self.analyze_metric(
                        column=None,  # Not from self.data, so pass None
                        display_name=f"{line_item_name} ({statement_type.replace('_', ' ').title()})",
                        category=category,
                        unit='R',
                        metric_type=metric_type,
                        values=values  # Pass values directly
                    )
                    
                    if assumption:
                        assumption.id = assumption_id
                        assumption.name = line_item_name
                        line_item_assumptions[statement_type][line_item_name] = assumption
        
        return line_item_assumptions
    
    def analyze_derived_metric(self, values: np.ndarray, id: str, display_name: str,
                               category: str, unit: str, metric_type: str) -> Optional[Assumption]:
        """Analyze a derived metric (calculated from other columns)."""
        if len(values) < 3:
            return None
        
        assumption = Assumption(
            id=id,
            name=id,
            display_name=display_name,
            category=category,
            unit=unit,
            historical_mean=float(np.mean(values)),
            historical_std=float(np.std(values)),
            historical_min=float(np.min(values)),
            historical_max=float(np.max(values)),
            historical_median=float(np.median(values)),
            data_points=len(values)
        )
        
        # Fit distribution
        best_params, _ = DistributionFitter.fit_best(values, metric_type)
        assumption.proposed_distribution = best_params
        assumption.final_static_value = assumption.historical_mean
        assumption.user_distribution = DistributionParams.from_dict(best_params.to_dict())
        
        self.assumptions[id] = assumption
        return assumption


# =============================================================================
# MANUFACTURING ASSUMPTIONS ANALYZER
# =============================================================================

class ManufacturingAnalyzer:
    """Analyzes historical data for manufacturing assumptions."""
    
    def __init__(self, historical_data: pd.DataFrame, parts_data: Optional[pd.DataFrame] = None):
        self.historical_data = historical_data
        self.parts_data = parts_data
        self.assumptions: Dict[str, ManufacturingAssumption] = {}
    
    def analyze_average_level(self) -> Dict[str, ManufacturingAssumption]:
        """Analyze at aggregate level when part-level data unavailable."""
        
        if self.historical_data.empty:
            return {}
        
        # Calculate aggregate metrics
        total_revenue = 0
        total_cogs = 0
        
        if 'total_revenue' in self.historical_data.columns:
            total_revenue = self.historical_data['total_revenue'].sum()
        elif 'revenue' in self.historical_data.columns:
            total_revenue = self.historical_data['revenue'].sum()
        
        if 'total_cogs' in self.historical_data.columns:
            total_cogs = self.historical_data['total_cogs'].sum()
        elif 'cogs' in self.historical_data.columns:
            total_cogs = self.historical_data['cogs'].sum()
        
        periods = len(self.historical_data)
        avg_revenue = total_revenue / periods if periods > 0 else 0
        avg_cogs = total_cogs / periods if periods > 0 else 0
        margin_pct = (total_revenue - total_cogs) / total_revenue * 100 if total_revenue > 0 else 0
        
        # Create aggregate assumption
        assumption = ManufacturingAssumption(
            part_id="aggregate",
            part_name="All Products (Average)",
            category="aggregate",
            historical_sell_price=avg_revenue,
            historical_cogs=avg_cogs,
            historical_margin_pct=margin_pct,
            historical_volume=periods,
            proposed_manufacture_pct=0,  # Start conservative
            proposed_mfg_cost_pct=75,  # Typical assumption
            rationale="Aggregate analysis - part-level data not available"
        )
        
        # AI recommendation based on margin
        if margin_pct < 30:
            assumption.proposed_manufacture_pct = 50  # Consider manufacturing for low margins
            assumption.rationale = f"Low margin ({margin_pct:.1f}%) suggests manufacturing could improve profitability"
        elif margin_pct < 40:
            assumption.proposed_manufacture_pct = 25
            assumption.rationale = f"Moderate margin ({margin_pct:.1f}%) - selective manufacturing recommended"
        else:
            assumption.proposed_manufacture_pct = 0
            assumption.rationale = f"Good margin ({margin_pct:.1f}%) - current sourcing strategy effective"
        
        self.assumptions["aggregate"] = assumption
        return self.assumptions
    
    def analyze_part_level(self) -> Dict[str, ManufacturingAssumption]:
        """Analyze at part/SKU level when detailed data available."""
        
        if self.parts_data is None or self.parts_data.empty:
            return self.analyze_average_level()
        
        # Identify part identifier column
        part_col = None
        for col in ['part_number', 'sku', 'liner_type', 'product_code', 'item_id']:
            if col in self.parts_data.columns:
                part_col = col
                break
        
        if part_col is None:
            return self.analyze_average_level()
        
        # Analyze each part
        for part_id, group in self.parts_data.groupby(part_col):
            # Get metrics
            sell_price = 0
            cogs = 0
            volume = len(group)
            
            # Try different column names
            for col in ['selling_price', 'price', 'unit_price', 'annual_liner_value']:
                if col in group.columns:
                    sell_price = group[col].mean()
                    break
            
            for col in ['cost', 'cogs', 'unit_cost', 'buy_cost']:
                if col in group.columns:
                    cogs = group[col].mean()
                    break
            
            # Estimate COGS from margin if not directly available
            if cogs == 0 and sell_price > 0:
                cogs = sell_price * 0.6  # Assume 40% margin
            
            margin_pct = (sell_price - cogs) / sell_price * 100 if sell_price > 0 else 0
            
            # Get part name/description
            part_name = str(part_id)
            for col in ['description', 'name', 'part_name', 'product_name']:
                if col in group.columns:
                    part_name = str(group[col].iloc[0])
                    break
            
            # Get category
            category = "liner"
            for col in ['category', 'type', 'product_type']:
                if col in group.columns:
                    category = str(group[col].iloc[0])
                    break
            
            # Create assumption with AI recommendation
            assumption = ManufacturingAssumption(
                part_id=str(part_id),
                part_name=part_name,
                category=category,
                historical_sell_price=sell_price,
                historical_cogs=cogs,
                historical_margin_pct=margin_pct,
                historical_volume=volume,
                proposed_mfg_cost_pct=75
            )
            
            # AI recommendation based on volume and margin
            if volume >= 100 and margin_pct < 35:
                assumption.proposed_manufacture_pct = 75
                assumption.rationale = f"High volume ({volume}) + low margin ({margin_pct:.1f}%) = good manufacturing candidate"
            elif volume >= 50 and margin_pct < 40:
                assumption.proposed_manufacture_pct = 50
                assumption.rationale = f"Medium volume ({volume}) + moderate margin ({margin_pct:.1f}%) = consider partial manufacturing"
            elif volume < 20:
                assumption.proposed_manufacture_pct = 0
                assumption.rationale = f"Low volume ({volume}) - continue outsourcing for flexibility"
            else:
                assumption.proposed_manufacture_pct = 25
                assumption.rationale = f"Standard profile - limited manufacturing recommended"
            
            self.assumptions[str(part_id)] = assumption
        
        return self.assumptions


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def load_assumptions_from_db(db, scenario_id: str, user_id: str = None) -> Optional[AssumptionsSet]:
    """Load saved assumptions from database.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        user_id: User ID for RLS compliance (required for loading)
    """
    try:
        # Get user_id if not provided
        if not user_id:
            try:
                from supabase_utils import get_user_id
                user_id = get_user_id()
            except:
                pass
        
        # Try assumptions table first (primary storage location)
        try:
            if hasattr(db, 'get_scenario_assumptions'):
                assumptions_data = db.get_scenario_assumptions(scenario_id, user_id)
                if assumptions_data and assumptions_data.get('ai_assumptions'):
                    ai_assumptions_data = assumptions_data['ai_assumptions']
                    if isinstance(ai_assumptions_data, str):
                        ai_assumptions_data = json.loads(ai_assumptions_data)
                    return AssumptionsSet.from_dict(ai_assumptions_data)
        except Exception as e:
            pass
        
        # Fallback: Try direct table access
        if hasattr(db, 'client'):
            try:
                query = db.client.table('assumptions').select('data').eq(
                    'scenario_id', scenario_id
                )
                # Add user_id filter for RLS compliance
                if user_id:
                    query = query.eq('user_id', user_id)
                
                response = query.execute()
                
                if response.data and len(response.data) > 0:
                    data = response.data[0].get('data', {})
                    if isinstance(data, str):
                        data = json.loads(data)
                    
                    # Check for AI assumptions in the data
                    if data and data.get('ai_assumptions'):
                        ai_assumptions = data['ai_assumptions']
                        if isinstance(ai_assumptions, str):
                            ai_assumptions = json.loads(ai_assumptions)
                        return AssumptionsSet.from_dict(ai_assumptions)
            except Exception as e:
                pass
        
        # Last resort: Check session state
        session_key = f'ai_assumptions_{scenario_id}'
        if session_key in st.session_state:
            return st.session_state[session_key]
            
    except Exception as e:
        pass
    
    return None


def save_assumptions_to_db(db, assumptions_set: AssumptionsSet, user_id: str = None) -> bool:
    """Save assumptions to database.
    
    Args:
        db: Database handler
        assumptions_set: AssumptionsSet to save
        user_id: User ID for RLS compliance (required for persistence)
    """
    try:
        # Get user_id if not provided
        if not user_id:
            try:
                from supabase_utils import get_user_id
                user_id = get_user_id()
            except Exception as e:
                show_error("User ID Required", f"Unable to get user ID for saving assumptions: {str(e)}")
                return False
        
        if not user_id:
            show_error("User ID Required", "User ID is required but could not be determined. Please ensure you are logged in.")
            return False
        
        assumptions_set.updated_at = datetime.now().isoformat()
        if not assumptions_set.created_at:
            assumptions_set.created_at = assumptions_set.updated_at
        
        assumptions_set.assumptions_saved = True
        
        # Prepare assumptions data - store AI assumptions in the data JSONB
        # Convert to dict and ensure JSON serializable (handles numpy types)
        assumptions_dict = assumptions_set.to_dict()
        
        # Double-check serialization - convert any remaining numpy types
        assumptions_dict = convert_to_serializable(assumptions_dict)
        
        # Use the existing update_assumptions method which uses the 'assumptions' table
        # This stores AI assumptions as part of the assumptions.data JSONB blob
        if hasattr(db, 'update_assumptions'):
            try:
                # Get existing assumptions data
                existing_assumptions = db.get_scenario_assumptions(assumptions_set.scenario_id, user_id)
                
                # Merge AI assumptions into the existing assumptions data
                if not existing_assumptions:
                    existing_assumptions = {}
                
                # Store AI assumptions in the data blob (ensure it's serializable)
                existing_assumptions['ai_assumptions'] = convert_to_serializable(assumptions_dict)
                existing_assumptions['ai_assumptions_saved'] = True
                existing_assumptions['ai_assumptions_updated_at'] = assumptions_set.updated_at
                
                # Ensure entire dict is serializable before saving
                existing_assumptions = convert_to_serializable(existing_assumptions)
                
                # Save using the standard assumptions table
                success = db.update_assumptions(
                    assumptions_set.scenario_id,
                    user_id,
                    existing_assumptions
                )
                
                if success:
                    # Invalidate cache to ensure fresh load on next access
                    cache_key = f'assumptions_{assumptions_set.scenario_id}'
                    if cache_key in st.session_state:
                        del st.session_state[cache_key]
                    
                    # Also invalidate AI assumptions cache
                    ai_cache_key = f'ai_assumptions_{assumptions_set.scenario_id}'
                    if ai_cache_key in st.session_state:
                        del st.session_state[ai_cache_key]
                    
                    return True
                else:
                    show_error("Save Failed", "Failed to save assumptions using update_assumptions method. Please try again.")
                    return False
                    
            except Exception as e:
                show_error(f"Save Error: {str(e)}", details=traceback.format_exc())
                return False
        
        # Fallback: Try direct table access to 'assumptions' table
        if hasattr(db, 'client'):
            try:
                # Prepare data for assumptions table (ensure serializable)
                assumptions_data = {
                    'ai_assumptions': convert_to_serializable(assumptions_dict),
                    'ai_assumptions_saved': True,
                    'ai_assumptions_updated_at': assumptions_set.updated_at
                }
                
                # Ensure all data is serializable
                assumptions_data = convert_to_serializable(assumptions_data)
                
                payload = {
                    "scenario_id": assumptions_set.scenario_id,
                    "user_id": user_id,
                    "data": assumptions_data
                }
                
                # Ensure payload is serializable
                payload = convert_to_serializable(payload)
                
                # Check if record exists
                existing = db.client.table("assumptions").select("id").eq(
                    "scenario_id", assumptions_set.scenario_id
                ).eq("user_id", user_id).execute()
                
                if existing.data:
                    # Update existing record
                    assump_id = existing.data[0]['id']
                    db.client.table("assumptions").update({
                        "data": assumptions_data
                    }).eq("id", assump_id).execute()
                else:
                    # Insert new record
                    db.client.table("assumptions").insert(payload).execute()
                
                # Invalidate cache to ensure fresh load on next access
                cache_key = f'assumptions_{assumptions_set.scenario_id}'
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
                
                # Also invalidate AI assumptions cache
                ai_cache_key = f'ai_assumptions_{assumptions_set.scenario_id}'
                if ai_cache_key in st.session_state:
                    del st.session_state[ai_cache_key]
                
                    return True
                
            except Exception as e:
                show_error(f"Database Save Error: {str(e)}", details=traceback.format_exc())
                return False
        
        # Last resort: Store in session state
        show_warning("Database Save Unavailable", "Could not save to database. Storing in session state only (will be lost on restart).")
        st.session_state[f'ai_assumptions_{assumptions_set.scenario_id}'] = assumptions_set
        return True
        
    except Exception as e:
        show_error(f"Save Failed: {str(e)}", details=traceback.format_exc())
        return False


def aggregate_detailed_line_items_to_summary(db, scenario_id: str, user_id: str = None) -> pd.DataFrame:
    """
    Aggregate detailed line items into summary format (revenue, COGS, OPEX, etc.).
    
    This allows the AI Assumptions engine to work with detailed line items
    when summary-level data isn't available.
    """
    try:
        # Wrap the entire function to catch any duplicate key errors
        # Load detailed income statement line items (pass user_id)
        is_li_df = load_detailed_line_items(db, scenario_id, 'income_statement', user_id)
        
        if is_li_df.empty:
            return pd.DataFrame()
        
        # Aggregate by period - ensure period_date is datetime first
        if 'period_date' in is_li_df.columns:
            is_li_df['period_date'] = pd.to_datetime(is_li_df['period_date'], errors='coerce')
            # Remove rows with invalid dates
            is_li_df = is_li_df.dropna(subset=['period_date'])
            # Remove any duplicate rows (same period_date + line_item_name)
            if len(is_li_df) > 0:
                is_li_df = is_li_df.drop_duplicates(subset=['period_date', 'line_item_name'], keep='first')

        # Ensure amount is numeric (Supabase may return NUMERIC as strings)
        if 'amount' in is_li_df.columns:
            is_li_df['amount'] = pd.to_numeric(is_li_df['amount'], errors='coerce').fillna(0.0)
        
        if is_li_df.empty:
            return pd.DataFrame()
        
        # Aggregate by period using groupby (ensures each period is processed exactly once)
        summary_data = []
        
        # Group by period_date - this automatically handles uniqueness
        grouped = is_li_df.groupby('period_date', dropna=True)
        
        def _classify_is_row(cat: str, name: str, sub_category: str = None, amount: Any = None) -> str:
            """
            Map imported line items into canonical buckets.

            We accept a wide variety of category labels from uploads and map them
            into Revenue / COGS / OPEX / Depreciation / Interest / Tax / Other Income.
            """
            # Robust string normalization (avoid .strip() on NaN/float)
            c = str(cat).strip().lower() if cat is not None and pd.notna(cat) else ""
            n = str(name).strip().lower() if name is not None and pd.notna(name) else ""
            sc = str(sub_category).strip().lower() if sub_category is not None and pd.notna(sub_category) else ""
            s = f"{c} {sc} {n}".strip()

            # Other income (keep separate so we don't understate EBITDA)
            if ("other income" in s) or ("finance income" in s) or ("interest received" in s):
                return "other_income"

            # COGS / Cost of Sales
            if any(k in s for k in ["cost of sales", "cogs", "cost of goods", "cost of goods sold", "purchases", "direct cost", "direct costs"]):
                return "cogs"

            # Revenue (broad)
            if any(k in s for k in ["revenue", "sales", "turnover"]):
                return "revenue"

            # Domain-specific revenue labels (Installed Base model)
            # Many uploads use segment headers like "Existing Customers" / "Prospective Customers"
            # with revenue items like "Wear Parts" / "Refurbishment & Service".
            if any(k in s for k in ["existing customer", "existing customers", "prospective customer", "prospective customers", "installed base"]):
                return "revenue"

            # Revenue line-items that often appear without an explicit "Revenue" category
            # Guard against misclassifying costs/expenses.
            if any(k in s for k in ["wear part", "wear parts", "wearparts", "liner", "liners", "consumable", "refurb", "refurbishment", "service"]):
                if not any(k in s for k in ["expense", "opex", "operating expense", "operating expenses", "overhead", "admin", "administration", "depreciation", "amort", "tax", "interest", "finance cost", "finance costs"]):
                    return "revenue"

            # Depreciation & amortisation
            if ("depreciation" in s) or ("amort" in s):
                return "depreciation"

            # Interest / finance costs
            if ("finance cost" in s) or ("finance costs" in s) or ("interest" in s):
                return "interest"

            # Tax
            if ("tax" in s) or ("taxation" in s):
                return "tax"

            # Everything else that is an expense bucket should flow into OPEX.
            # Common labels: Other Expense, Distribution Cost, Operating Expenses, Admin, etc.
            if any(k in s for k in ["expense", "operating", "distribution", "overhead", "admin", "administration"]):
                return "opex"

            # Default: treat as OPEX (safer than dropping)
            return "opex"

        for period_date, period_df in grouped:
            # period_date is guaranteed unique here (groupby handles it)

            # Ensure we can classify even if category is missing
            cats = period_df.get('category')
            subs = period_df.get('sub_category')
            names = period_df.get('line_item_name')
            amts = period_df.get('amount')
            period_df = period_df.copy()
            period_df['__bucket'] = [
                _classify_is_row(
                    (cats.iloc[i] if cats is not None else None),
                    (names.iloc[i] if names is not None else None),
                    (subs.iloc[i] if subs is not None else None),
                    (amts.iloc[i] if amts is not None else None),
                )
                for i in range(len(period_df))
            ]

            # Sum by bucket. Source data often uses negative signs for expenses.
            sums = period_df.groupby('__bucket', dropna=False)['amount'].sum()

            revenue = float(sums.get('revenue', 0) or 0)
            cogs = abs(float(sums.get('cogs', 0) or 0))
            opex = abs(float(sums.get('opex', 0) or 0))
            depreciation = abs(float(sums.get('depreciation', 0) or 0))
            interest = abs(float(sums.get('interest', 0) or 0))
            tax = abs(float(sums.get('tax', 0) or 0))
            other_income = float(sums.get('other_income', 0) or 0)
            
            # Calculate derived metrics
            gross_profit = revenue - cogs
            gross_margin = (gross_profit / revenue * 100) if revenue != 0 else 0
            ebit = gross_profit - opex - depreciation + other_income
            
            # Convert period_date to string for consistency
            if isinstance(period_date, pd.Timestamp):
                period_str = period_date.strftime('%Y-%m-%d')
            elif hasattr(period_date, 'strftime'):
                period_str = period_date.strftime('%Y-%m-%d')
            else:
                period_str = str(period_date)
            
            # Convert all values to Python native types explicitly
            rev_val = float(revenue) if revenue is not None and pd.notna(revenue) else 0.0
            cogs_val = float(cogs) if cogs is not None and pd.notna(cogs) else 0.0
            opex_val = float(opex) if opex is not None and pd.notna(opex) else 0.0
            gp_val = float(gross_profit) if gross_profit is not None and pd.notna(gross_profit) else 0.0
            gm_val = float(gross_margin) if gross_margin is not None and pd.notna(gross_margin) else 0.0
            dep_val = float(depreciation) if depreciation is not None and pd.notna(depreciation) else 0.0
            int_val = float(interest) if interest is not None and pd.notna(interest) else 0.0
            tax_val = float(tax) if tax is not None and pd.notna(tax) else 0.0
            ebit_val = float(ebit) if ebit is not None and pd.notna(ebit) else 0.0
            
            # Create a fresh dict for each period (ensures no key conflicts)
            summary_row = {
                'period_date': period_str,
                'month': period_str,
                'revenue': rev_val,
                'total_revenue': rev_val,
                'cogs': cogs_val,
                'total_cogs': cogs_val,
                'gross_profit': gp_val,
                'total_gross_profit': gp_val,
                'gross_margin': gm_val,
                'opex': opex_val,
                'total_opex': opex_val,
                'depreciation': dep_val,
                'interest_expense': int_val,
                'tax': tax_val,
                'other_income': float(other_income) if other_income is not None and pd.notna(other_income) else 0.0,
                'ebit': ebit_val
            }
            summary_data.append(summary_row)
        
        if summary_data:
            # Build DataFrame directly - summary_data already has unique periods (from groupby)
            # Create a simple list of dicts with consistent structure
            rows = []
            for record in summary_data:
                if isinstance(record, dict):
                    # Ensure all values are Python native types (not numpy)
                    row = {
                        'period_date': str(record.get('period_date', '')),
                        'month': str(record.get('month', record.get('period_date', ''))),
                        'revenue': float(record.get('revenue', 0.0)),
                        'total_revenue': float(record.get('total_revenue', record.get('revenue', 0.0))),
                        'cogs': float(record.get('cogs', 0.0)),
                        'total_cogs': float(record.get('total_cogs', record.get('cogs', 0.0))),
                        'gross_profit': float(record.get('gross_profit', 0.0)),
                        'total_gross_profit': float(record.get('total_gross_profit', record.get('gross_profit', 0.0))),
                        'gross_margin': float(record.get('gross_margin', 0.0)),
                        'opex': float(record.get('opex', 0.0)),
                        'total_opex': float(record.get('total_opex', record.get('opex', 0.0))),
                        'depreciation': float(record.get('depreciation', 0.0)),
                        'interest_expense': float(record.get('interest_expense', 0.0)),
                        'tax': float(record.get('tax', 0.0)),
                        'ebit': float(record.get('ebit', 0.0))
                    }
                    rows.append(row)
            
            if not rows:
                return pd.DataFrame()
            
            # Create DataFrame with multiple fallback methods
            df = None
            try:
                # Method 1: Direct DataFrame creation
                df = pd.DataFrame(rows)
                df = df.reset_index(drop=True)
            except (ValueError, TypeError, KeyError) as e1:
                error_msg = str(e1).lower()
                if 'duplicate' in error_msg or 'cannot assemble' in error_msg:
                    # Method 2: Build row by row
                    try:
                        df = pd.DataFrame()
                        for row in rows:
                            row_df = pd.DataFrame([row])
                            if df.empty:
                                df = row_df
                            else:
                                df = pd.concat([df, row_df], ignore_index=True, sort=False, verify_integrity=False)
                        df = df.reset_index(drop=True)
                    except Exception as e2:
                        # Method 3: Use explicit column structure
                        try:
                            columns = ['period_date', 'month', 'revenue', 'total_revenue', 'cogs', 'total_cogs',
                                     'gross_profit', 'total_gross_profit', 'gross_margin', 'opex', 'total_opex',
                                     'depreciation', 'interest_expense', 'tax', 'ebit']
                            data_list = [[row.get(col, 0.0) for col in columns] for row in rows]
                            df = pd.DataFrame(data_list, columns=columns)
                        except Exception as e3:
                            # All methods failed
                            import traceback
                            try:
                                st.error(f"❌ All DataFrame creation methods failed: {str(e3)}")
                                with st.expander("🔍 Error Details"):
                                    st.write(f"Rows: {len(rows)}")
                                    if rows:
                                        st.write(f"Sample row: {rows[0]}")
                                    st.code(traceback.format_exc())
                            except:
                                pass
                            return pd.DataFrame()
                else:
                    raise
            
            # Post-process DataFrame
            if df is not None and not df.empty:
                # Convert period_date to datetime
                if 'period_date' in df.columns:
                    df['period_date'] = pd.to_datetime(df['period_date'], errors='coerce')
                    df = df.dropna(subset=['period_date'])
                
                # Sort by period_date
                if 'period_date' in df.columns and len(df) > 0:
                    df = df.sort_values('period_date').reset_index(drop=True)
                
                # Ensure month column
                if 'month' in df.columns:
                    df['month'] = pd.to_datetime(df['month'], errors='coerce')
                elif 'period_date' in df.columns:
                    df['month'] = df['period_date']
                
                return df
            else:
                return pd.DataFrame()
        
        return pd.DataFrame()
    except (ValueError, TypeError, KeyError) as e:
        # Handle duplicate key errors specifically
        error_msg = str(e).lower()
        if 'duplicate' in error_msg or 'cannot assemble' in error_msg:
            import traceback
            try:
                st.error(f"❌ Duplicate key error during aggregation: {str(e)}")
                st.info("""
                **This error occurs when creating a DataFrame from aggregated data.**
                
                **The aggregation function is working, but DataFrame creation is failing.**
                
                **Possible solutions:**
                1. The issue may resolve on refresh - try reloading the page
                2. Check if there are any data inconsistencies in your imported line items
                3. Contact support with the error details below
                """)
                with st.expander("🔍 Technical Error Details"):
                    st.code(traceback.format_exc())
            except:
                pass
        return pd.DataFrame()
    except Exception as e:
        # Other errors
        import traceback
        try:
            st.error(f"❌ Error in aggregation: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
        except:
            pass
        return pd.DataFrame()


def load_historical_data(db, scenario_id: str, user_id: str = None) -> pd.DataFrame:
    """Load historical financial data from database.
    
    Tries multiple sources in order:
    1. Database helper methods (get_historic_financials, get_historical_financials)
    2. Direct query to 'historic_financials' table (monthly P&L data)
    3. Direct query to 'historical_financials' table (annual/categorized data)
    4. Aggregate detailed line items into summary format (NEW)
    """
    try:
        data = []
        
        # Try database helper methods first
        if hasattr(db, 'get_historic_financials'):
            try:
                data = db.get_historic_financials(scenario_id)
            except Exception:
                data = []
        
        if not data and hasattr(db, 'get_historical_financials'):
            try:
                data = db.get_historical_financials(scenario_id)
            except Exception:
                data = []
        
        # If no data from helpers, query directly
        if not data and hasattr(db, 'client'):
            # Try 'historic_financials' table first (monthly P&L data)
            # This table has: month, revenue, cogs, gross_profit, gross_margin, opex, ebit
            try:
                response = db.client.table('historic_financials').select('*').eq(
                    'scenario_id', scenario_id
                ).order('month').execute()
                data = response.data or []
            except Exception as e:
                # Table might not exist or other error
                pass
            
            # Fallback to 'historical_financials' if no data
            # This table has: year, category, line_item, amount
            if not data:
                try:
                    response = db.client.table('historical_financials').select('*').eq(
                        'scenario_id', scenario_id
                    ).execute()
                    data = response.data or []
                except Exception:
                    pass
        
        # NEW: If still no data, try aggregating detailed line items
        if not data:
            try:
                aggregated_df = aggregate_detailed_line_items_to_summary(db, scenario_id, user_id)
                if not aggregated_df.empty and isinstance(aggregated_df, pd.DataFrame):
                    # Use the DataFrame directly - it's already fully processed
                    # Just add compatibility aliases if needed
                    df = aggregated_df.copy()
                    
                    # Add aliases (don't rename - just add if missing)
                    if 'total_revenue' not in df.columns and 'revenue' in df.columns:
                        df['total_revenue'] = df['revenue']
                    if 'total_cogs' not in df.columns and 'cogs' in df.columns:
                        df['total_cogs'] = df['cogs']
                    if 'total_gross_profit' not in df.columns and 'gross_profit' in df.columns:
                        df['total_gross_profit'] = df['gross_profit']
                    if 'total_opex' not in df.columns and 'opex' in df.columns:
                        df['total_opex'] = df['opex']
                    
                    # Ensure required columns exist
                    for col, default in [('interest_expense', 0.0), ('depreciation', 0.0), ('tax', 0.0)]:
                        if col not in df.columns:
                            df[col] = default
                    
                    # Return directly - no further processing needed
                    return df
            except Exception as e:
                # Log error for debugging
                import traceback
                error_msg = str(e)
                try:
                    st.error(f"❌ Error aggregating detailed line items: {error_msg}")
                    if 'duplicate' in error_msg.lower() or 'cannot assemble' in error_msg.lower():
                        st.info("""
                        **This error occurred during aggregation of detailed line items.**
                        
                        **Possible causes:**
                        - Duplicate period dates in source data
                        - Data type conversion issues
                        - DataFrame creation conflict
                        
                        **Debug steps:**
                        1. Check the error details below
                        2. Verify your imported line items have unique period dates
                        3. Try clearing and re-importing the data
                        """)
                    with st.expander("🔍 Full Error Traceback"):
                        st.code(traceback.format_exc())
                except:
                    pass
                # Don't set data = [] here - let it continue to try other sources
        
        if data:
            # Create DataFrame - handle potential duplicate keys more robustly
            df = None
            try:
                # First, check if data has duplicate period dates
                if isinstance(data, list) and len(data) > 0:
                    # Check for duplicate period keys
                    period_keys = []
                    for record in data:
                        if isinstance(record, dict):
                            period = record.get('period_date') or record.get('month')
                            if period:
                                period_keys.append(period)
                    
                    # If duplicates found, deduplicate
                    if len(period_keys) != len(set(period_keys)):
                        seen_periods = set()
                        deduplicated_data = []
                        for record in data:
                            if isinstance(record, dict):
                                period = record.get('period_date') or record.get('month')
                                if period not in seen_periods:
                                    seen_periods.add(period)
                                    deduplicated_data.append(record)
                        data = deduplicated_data
                    
                    # Normalize all records to have the same keys
                    if len(data) > 0:
                        all_keys = set()
                        for record in data:
                            if isinstance(record, dict):
                                all_keys.update(record.keys())
                        
                        normalized_data = []
                        for record in data:
                            if isinstance(record, dict):
                                normalized_record = {key: record.get(key, None) for key in all_keys}
                                normalized_data.append(normalized_record)
                        
                        # Use from_records which is more robust
                        try:
                            df = pd.DataFrame.from_records(normalized_data)
                        except (ValueError, TypeError, KeyError) as rec_error:
                            error_msg = str(rec_error).lower()
                            if 'duplicate' in error_msg or 'cannot assemble' in error_msg:
                                # Last resort: build DataFrame row by row
                                df = pd.DataFrame()
                                seen_periods = set()
                                for record in normalized_data:
                                    if isinstance(record, dict):
                                        period = record.get('period_date')
                                        period_key = str(period) if period else f"row_{len(df)}"
                                        if period_key in seen_periods:
                                            continue
                                        seen_periods.add(period_key)
                                        try:
                                            row_df = pd.DataFrame([record])
                                            if df.empty:
                                                df = row_df
                                            else:
                                                df = pd.concat([df, row_df], ignore_index=True, sort=False, verify_integrity=False)
                                        except:
                                            continue
                                if df.empty:
                                    return pd.DataFrame()
                            else:
                                raise
                    else:
                        return pd.DataFrame()
                else:
                    df = pd.DataFrame(data)
            except (ValueError, TypeError, KeyError) as e:
                error_msg = str(e).lower()
                if 'duplicate' in error_msg or 'cannot assemble' in error_msg or 'key' in error_msg:
                    # If there are duplicate keys, try alternative approach
                    # Create DataFrame row by row to avoid issues
                    df = pd.DataFrame()
                    seen_periods = set()
                    for i, record in enumerate(data):
                        try:
                            if isinstance(record, dict):
                                # Check for duplicate period
                                period = record.get('period_date') or record.get('month')
                                period_key = str(period) if period else None
                                if period_key and period_key in seen_periods:
                                    continue  # Skip duplicate period
                                if period_key:
                                    seen_periods.add(period_key)
                                
                                # Convert single record to DataFrame
                                record_df = pd.DataFrame([record])
                                if df.empty:
                                    df = record_df
                                else:
                                    df = pd.concat([df, record_df], ignore_index=True, sort=False, verify_integrity=False)
                        except Exception:
                            # Skip problematic records
                            continue
                    
                    # If still empty, return empty DataFrame
                    if df.empty:
                        return pd.DataFrame()
                else:
                    # Re-raise if it's a different error
                    raise
            
            # Remove duplicate rows if any (based on period_date if available)
            if df is not None and not df.empty:
                if 'period_date' in df.columns or 'month' in df.columns:
                    # Use period_date or month to identify duplicates
                    period_col = 'period_date' if 'period_date' in df.columns else 'month'
                    if period_col in df.columns:
                        # Check for duplicates before dropping
                        if df[period_col].duplicated().any():
                            df = df.drop_duplicates(subset=[period_col], keep='first')
                else:
                    # Otherwise just drop full duplicates
                    df = df.drop_duplicates()
                df = df.reset_index(drop=True)
            else:
                return pd.DataFrame()
            
            # Standardize column names (for historic_financials table format)
            # Map both original names and standardized names to ensure compatibility
            column_map = {
                'month': 'period_date',
                'revenue': 'total_revenue',
                'cogs': 'total_cogs',
                'gross_profit': 'total_gross_profit',
                'opex': 'total_opex',
                # Keep depreciation, interest_expense, and tax as-is (they're already in correct format)
                # But also map any variations
                'depreciation_amortization': 'depreciation',
                'interest': 'interest_expense',
                'interest_expense': 'interest_expense',  # Ensure it exists
                'tax_expense': 'tax',
                'income_tax': 'tax'
            }
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
            
            # If interest_expense doesn't exist, try to extract from other_expense
            # (Some trial balances may have interest in other_expense)
            # For now, we'll use other_expense as a proxy if interest_expense doesn't exist
            if 'interest_expense' not in df.columns:
                if 'other_expense' in df.columns:
                    # Use other_expense as interest_expense (common case where interest is the main other expense)
                    df['interest_expense'] = df['other_expense'].fillna(0)
                else:
                    df['interest_expense'] = 0
            
            # Ensure depreciation and tax columns exist (even if zero)
            # This ensures the forecast config UI can find them
            if 'depreciation' not in df.columns:
                df['depreciation'] = 0
            if 'tax' not in df.columns:
                df['tax'] = 0
            
            # Also ensure the original column names are preserved for backward compatibility
            # Add aliases so both naming conventions work
            if 'depreciation' in df.columns and 'total_depreciation' not in df.columns:
                df['total_depreciation'] = df['depreciation']
            if 'interest_expense' in df.columns and 'total_interest_expense' not in df.columns:
                df['total_interest_expense'] = df['interest_expense']
            if 'tax' in df.columns and 'total_tax' not in df.columns:
                df['total_tax'] = df['tax']
            
            # Convert period_date to datetime if present
            if 'period_date' in df.columns:
                df['period_date'] = pd.to_datetime(df['period_date'])
            elif 'month' in df.columns:
                df['period_date'] = pd.to_datetime(df['month'])
                # Don't drop month column yet - keep for compatibility
                # df = df.drop(columns=['month'])
            
            # Remove duplicate period dates (keep first)
            if 'period_date' in df.columns:
                df = df.drop_duplicates(subset=['period_date'], keep='first')
                df = df.sort_values('period_date').reset_index(drop=True)
            
            return df
        
        return pd.DataFrame()
        
    except Exception as e:
        error_msg = str(e)
        # Provide more helpful error message for duplicate keys
        if 'duplicate' in error_msg.lower() or 'cannot assemble' in error_msg.lower():
            st.error(f"❌ Error loading historical data: {error_msg}")
            st.info("""
            **Possible causes:**
            - Duplicate period dates in the data
            - Conflicting data formats between summary and detailed line items
            - Data structure inconsistencies
            
            **Try:**
            1. Check your imported data for duplicate periods
            2. Clear and re-import historical data
            3. Ensure all periods have unique dates
            """)
            with st.expander("🔍 Technical Details"):
                import traceback
                st.code(traceback.format_exc())
        else:
            st.error(f"Error loading historical data: {error_msg}")
        return pd.DataFrame()


def load_detailed_line_items(db, scenario_id: str, statement_type: str = 'income_statement', user_id: str = None) -> pd.DataFrame:
    """
    Load detailed line items from database.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        statement_type: 'income_statement', 'balance_sheet', or 'cash_flow'
        user_id: User ID (required for detailed line items tables)
    
    Returns:
        DataFrame with columns: period_date, line_item_name, category, sub_category, amount
    """
    table_map = {
        'income_statement': 'historical_income_statement_line_items',
        'balance_sheet': 'historical_balance_sheet_line_items',
        'cash_flow': 'historical_cashflow_line_items'
    }
    
    table_name = table_map.get(statement_type)
    if not table_name:
        return pd.DataFrame()
    
    try:
        if hasattr(db, 'client'):
            base_query = db.client.table(table_name).select('*').eq('scenario_id', scenario_id)
            rows: List[Dict[str, Any]] = []

            # Prefer user_id filter (RLS-friendly), then fall back to scenario-only.
            # IMPORTANT: Supabase/PostgREST commonly caps responses to 1000 rows; paginate.
            if fetch_all_rows:
                if user_id:
                    try:
                        rows = fetch_all_rows(base_query.eq('user_id', user_id), order_by="id")
                    except Exception:
                        rows = []
                if not rows:
                    rows = fetch_all_rows(base_query, order_by="id")
            else:
                response = None
                if user_id:
                    try:
                        response = base_query.eq('user_id', user_id).order('period_date').execute()
                    except Exception:
                        response = None
                if response is None or not getattr(response, 'data', None):
                    response = base_query.order('period_date').execute()
                rows = response.data or [] if response and getattr(response, 'data', None) else []

            if rows:
                df = pd.DataFrame(rows)
                if 'period_date' in df.columns:
                    df['period_date'] = pd.to_datetime(df['period_date'], errors='coerce')
                    # Remove any rows with invalid dates
                    df = df.dropna(subset=['period_date'])
                    # Remove duplicates based on scenario_id, period_date, and line_item_name
                    # (these should be unique per the table schema)
                    if len(df) > 0:
                        df = df.drop_duplicates(subset=['scenario_id', 'period_date', 'line_item_name'], keep='first')
                        df = df.sort_values('period_date').reset_index(drop=True)
                return df
    except Exception as e:
        # Table might not exist yet - that's okay
        pass
    
        return pd.DataFrame()


def load_parts_data(db, scenario_id: str, user_id: str = None) -> pd.DataFrame:
    """Load parts/installed base data from multiple sources.
    
    Combines data from:
    1. wear_profiles table (liner type definitions with pricing) - uses user_id
    2. installed_base table (fleet data with liner types) - uses scenario_id
    """
    try:
        # Source 1: Wear profiles (uses user_id, not scenario_id)
        wear_profiles_data = []
        if user_id and hasattr(db, 'client'):
            try:
                # Query wear_profiles with user_id
                response = db.client.table('wear_profiles').select('*').eq(
                    'user_id', user_id
                ).execute()
                wear_profiles_data = response.data or []
            except:
                pass
        
        # If we have wear profiles, use them as the primary source
        if wear_profiles_data:
            df_profiles = pd.DataFrame(wear_profiles_data)
            
            # Identify the liner type column
            liner_col = None
            for col in ['liner_type', 'machine_model', 'sku', 'part_number']:
                if col in df_profiles.columns:
                    liner_col = col
                    break
            
            if liner_col:
                df_profiles['part_number'] = df_profiles[liner_col]
                df_profiles['part_name'] = df_profiles[liner_col]
            
            # Map pricing columns
            if 'avg_consumable_revenue' in df_profiles.columns:
                df_profiles['selling_price'] = df_profiles['avg_consumable_revenue']
            elif 'annual_liner_value' in df_profiles.columns:
                df_profiles['selling_price'] = df_profiles['annual_liner_value']
            elif 'unit_price' in df_profiles.columns:
                df_profiles['selling_price'] = df_profiles['unit_price']
            
            return df_profiles
        
        # Source 2: Installed base (uses scenario_id)
        installed_base_data = []
        if hasattr(db, 'get_installed_base'):
            installed_base_data = db.get_installed_base(scenario_id) or []
        elif hasattr(db, 'client'):
            try:
                q = db.client.table('installed_base').select('*').eq('scenario_id', scenario_id)
                if fetch_all_rows:
                    installed_base_data = fetch_all_rows(q, order_by="id")
                else:
                    response = q.execute()
                    installed_base_data = response.data or []
            except:
                pass
        
        if installed_base_data:
            df = pd.DataFrame(installed_base_data)
            
            # Get unique liner types from installed base
            if 'liner_type' in df.columns:
                # Group by liner_type to get unique parts with aggregated metrics
                agg_dict = {'liner_type': 'first'}
                if 'annual_liner_value' in df.columns:
                    agg_dict['annual_liner_value'] = 'mean'
                if 'refurb_value' in df.columns:
                    agg_dict['refurb_value'] = 'mean'
                
                unique_liners = df.groupby('liner_type').agg(agg_dict).reset_index(drop=True)
                
                unique_liners['part_number'] = unique_liners['liner_type']
                unique_liners['part_name'] = unique_liners['liner_type']
                if 'annual_liner_value' in unique_liners.columns:
                    unique_liners['selling_price'] = unique_liners['annual_liner_value']
                
                return unique_liners
            
            return df
        
        return pd.DataFrame()
        
    except Exception as e:
        return pd.DataFrame()


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_distribution_preview(assumption: Assumption, num_samples: int = 1000) -> go.Figure:
    """Create a visualization comparing historical data distribution with fitted distribution."""
    
    params = assumption.user_distribution
    
    # Generate samples from the distribution
    samples = generate_distribution_samples(params, num_samples)
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram of theoretical distribution
    fig.add_trace(go.Histogram(
        x=samples,
        name=f'Fitted {params.distribution_type.title()}',
        opacity=0.7,
        marker_color=BLUE,
        nbinsx=30
    ))
    
    # Add vertical lines for key statistics
    fig.add_vline(x=assumption.historical_mean, line_dash="solid", line_color=GOLD,
                  annotation_text=f"Mean: {assumption.historical_mean:,.0f}")
    fig.add_vline(x=assumption.historical_min, line_dash="dot", line_color=RED,
                  annotation_text="Min")
    fig.add_vline(x=assumption.historical_max, line_dash="dot", line_color=GREEN,
                  annotation_text="Max")
    
    fig.update_layout(
        title=f"Distribution Preview: {assumption.display_name}",
        xaxis_title=f"{assumption.display_name} ({assumption.unit})",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_WHITE),
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )
    
    return fig


def generate_distribution_samples(params: DistributionParams, n: int = 1000) -> np.ndarray:
    """Generate samples from a distribution based on parameters."""
    
    dist_type = params.distribution_type
    
    if dist_type == "static":
        return np.full(n, params.static_value)
    
    elif dist_type == "normal":
        return np.random.normal(params.mean, params.std, n)
    
    elif dist_type == "lognormal":
        return np.random.lognormal(params.mean, params.std, n)
    
    elif dist_type == "triangular":
        if params.max_val > params.min_val:
            return np.random.triangular(params.min_val, params.mode_val, params.max_val, n)
        return np.full(n, params.mode_val)
    
    elif dist_type == "beta":
        samples = np.random.beta(params.alpha, params.beta, n)
        return samples * params.scale + params.loc
    
    elif dist_type == "uniform":
        return np.random.uniform(params.low, params.high, n)
    
    elif dist_type == "pert":
        # PERT is a modified beta distribution
        mean = (params.min_val + 4 * params.mode_val + params.max_val) / 6
        if params.max_val > params.min_val:
            alpha = 1 + 4 * (params.mode_val - params.min_val) / (params.max_val - params.min_val)
            beta_param = 1 + 4 * (params.max_val - params.mode_val) / (params.max_val - params.min_val)
            samples = np.random.beta(alpha, beta_param, n)
            return samples * (params.max_val - params.min_val) + params.min_val
        return np.full(n, params.mode_val)
    
    else:
        return np.full(n, params.static_value)


def create_historical_trend_chart(data: pd.DataFrame, column: str, assumption: Assumption) -> go.Figure:
    """Create a chart showing historical trend with fitted line."""
    
    fig = go.Figure()
    
    if 'period_date' in data.columns:
        x_vals = pd.to_datetime(data['period_date'])
    else:
        x_vals = list(range(len(data)))
    
    y_vals = data[column].values
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines+markers',
        name='Historical',
        line=dict(color=GOLD, width=2),
        marker=dict(size=6)
    ))
    
    # Trend line
    x_numeric = np.arange(len(y_vals))
    slope, intercept, _, _, _ = stats.linregress(x_numeric, y_vals)
    trend_line = slope * x_numeric + intercept
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=trend_line,
        mode='lines',
        name=f'Trend ({assumption.historical_trend})',
        line=dict(color=BLUE, width=2, dash='dash')
    ))
    
    # Mean line
    fig.add_hline(y=assumption.historical_mean, line_dash="dot", line_color=GREEN,
                  annotation_text=f"Mean: {assumption.historical_mean:,.0f}")
    
    fig.update_layout(
        title=f"{assumption.display_name} - Historical Trend",
        xaxis_title="Period",
        yaxis_title=f"{assumption.display_name} ({assumption.unit})",
        yaxis_tickformat=',.0f',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_WHITE),
        hovermode='x unified'
    )
    
    return fig


# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================

def section_header(title: str, subtitle: str = None):
    """Render a section header."""
    sub = f'<p style="color: {TEXT_MUTED}; margin: 0; font-size: 0.85rem;">{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div style="margin: 1.5rem 0 1rem 0;">
        <h3 style="color: {GOLD}; margin: 0; font-size: 1.2rem;">{title}</h3>
        {sub}
    </div>
    """, unsafe_allow_html=True)


def format_currency(value: float, decimals: int = 0) -> str:
    """Format value as currency."""
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


def status_badge(status: str, text: str):
    """Render a status badge."""
    colors = {
        'success': GREEN,
        'warning': ORANGE,
        'error': RED,
        'info': BLUE,
        'pending': TEXT_MUTED
    }
    color = colors.get(status, TEXT_MUTED)
    st.markdown(f"""
    <span style="
        background: {color}22;
        color: {color};
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    ">{text}</span>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_ai_assumptions_section(db, scenario_id: str, user_id: str):
    """
    Main entry point for the AI Assumptions Engine.
    This should be a required step after Setup, before Forecast.
    """
    st.header("🤖 AI Assumptions Engine")
    st.caption("Derive forecast assumptions from historical data analysis")
    
    # Store user_id in session state for child functions
    st.session_state['ai_assumptions_user_id'] = user_id
    
    # Initialize or load assumptions (Sprint 19 - Auto-persistence)
    if 'ai_assumptions_set' not in st.session_state or st.session_state.get('ai_assumptions_scenario') != scenario_id:
        # Try to load from database first (includes auto-saved analysis results)
        loaded = load_assumptions_from_db(db, scenario_id, user_id)
        if loaded:
            st.session_state.ai_assumptions_set = loaded
            # If loaded from database, it means analysis was run before (auto-saved or explicitly saved)
        else:
            # Check for auto-saved analysis results in assumptions table
            try:
                existing_assumptions = db.get_scenario_assumptions(scenario_id, user_id)
                if existing_assumptions and existing_assumptions.get('ai_assumptions'):
                    ai_assumptions_data = existing_assumptions['ai_assumptions']
                    if isinstance(ai_assumptions_data, str):
                        ai_assumptions_data = json.loads(ai_assumptions_data)
                    if isinstance(ai_assumptions_data, dict):
                        loaded_set = AssumptionsSet.from_dict(ai_assumptions_data)
                        if loaded_set.analysis_complete:
                            # Found auto-saved analysis results
                            st.session_state.ai_assumptions_set = loaded_set
                            loaded = loaded_set
            except Exception:
                pass
        
        if not loaded:
            st.session_state.ai_assumptions_set = AssumptionsSet(scenario_id=scenario_id)
        st.session_state.ai_assumptions_scenario = scenario_id
    
    assumptions_set = st.session_state.ai_assumptions_set
    
    # Status display
    col1, col2, col3 = st.columns(3)
    with col1:
        if assumptions_set.analysis_complete:
            status_badge('success', '✓ Analysis Complete')
        else:
            status_badge('pending', '○ Analysis Pending')
    with col2:
        # Check unified config saved status as well
        unified_saved = False
        try:
            assumptions = db.get_scenario_assumptions(scenario_id, user_id)
            if assumptions:
                unified_config = assumptions.get('unified_line_item_config', {})
                if unified_config and unified_config.get('last_updated'):
                    unified_saved = True
                # Also check top-level saved flag
                if assumptions.get('assumptions_saved'):
                    unified_saved = True
        except:
            pass
        
        if assumptions_set.assumptions_saved or unified_saved:
            status_badge('success', '✓ Assumptions Saved')
        else:
            status_badge('warning', '○ Not Saved')
    with col3:
        st.metric("Data Quality", f"{assumptions_set.data_quality_score:.0f}%")
    
    st.markdown("---")
    
    # Check if legacy tabs should be shown (can be toggled in session state)
    show_legacy = st.session_state.get('show_legacy_tabs', False)
    
    # Toggle for legacy tabs in a small expander
    with st.expander("⚙️ View Options", expanded=False):
        show_legacy = st.checkbox(
            "Show Legacy Tabs (Trend Forecast, Distributions)",
            value=show_legacy,
            help="Legacy tabs are kept for backward compatibility. The 'Configure Assumptions' tab is the new unified approach."
        )
        st.session_state['show_legacy_tabs'] = show_legacy
    
    st.markdown("---")
    
    # Build tab list based on legacy toggle
    if show_legacy:
        tab_names = [
        "📊 Run Analysis",
            "⚙️ Configure Assumptions",
            "📈 Trend Forecast (Legacy)",
            "💰 Distributions (Legacy)",
            "🏭 Manufacturing",
        "💾 Save & Apply"
        ]
        tabs = st.tabs(tab_names)
        tab1, tab2, tab3, tab4, tab5, tab6 = tabs
    else:
        tab_names = [
            "📊 Run Analysis",
            "⚙️ Configure Assumptions",
            "🏭 Manufacturing",
            "💾 Save & Apply"
        ]
        tabs = st.tabs(tab_names)
        tab1, tab2, tab3, tab4 = tabs
        tab5 = tab3  # Remap for manufacturing
        tab6 = tab4  # Remap for save/apply
        tab3 = None  # Legacy tabs not shown
        tab4 = None
    
    # ==========================================================================
    # TAB 1: Run Analysis
    # ==========================================================================
    with tab1:
        render_analysis_tab(db, scenario_id, user_id, assumptions_set)
    
    # ==========================================================================
    # TAB 2: Unified Configuration (NEW)
    # ==========================================================================
    with tab2:
        try:
            from components.unified_assumptions_config import render_unified_config_ui
            render_unified_config_ui(db, scenario_id, user_id)
        except ImportError as e:
            st.error(f"Could not load unified configuration UI: {e}")
            st.info("Please ensure unified_assumptions_config.py is in the components folder.")
    
    # ==========================================================================
    # TAB 3 & 4: Legacy Tabs (only shown if toggled)
    # ==========================================================================
    if show_legacy:
        with tab3:
            st.warning("⚠️ **Legacy Tab** - Use 'Configure Assumptions' tab for the new unified approach")
            render_trend_forecast_tab(db, scenario_id, user_id)
    
        with tab4:
            st.warning("⚠️ **Legacy Tab** - Use 'Configure Assumptions' tab for the new unified approach")
            render_financial_assumptions_tab(db, scenario_id, assumptions_set)
        
        # Manufacturing and Save/Apply are tabs 5 and 6 when legacy is shown
        with tab5:
            render_manufacturing_assumptions_tab(db, scenario_id, assumptions_set)
        
        with tab6:
            render_save_apply_tab(db, scenario_id, assumptions_set, user_id)
    else:
        # Manufacturing is tab3, Save/Apply is tab4 when legacy is hidden
        with tabs[2]:  # Manufacturing
            render_manufacturing_assumptions_tab(db, scenario_id, assumptions_set)
        
        with tabs[3]:  # Save & Apply
            render_save_apply_tab(db, scenario_id, assumptions_set, user_id)


def render_trend_forecast_tab(db, scenario_id: str, user_id: str):
    """Render the trend forecast analysis tab."""
    st.markdown("### 📈 Comprehensive Forecast Configuration")
    st.caption("Configure forecast methods with correlations, trends, and period overrides")
    
    # Load historical financial data using the same logic as AI Assumptions
    # This includes aggregation from detailed line items if summary data isn't available
    try:
        hist_df = load_historical_data(db, scenario_id, user_id)
        
        if hist_df.empty:
            st.warning("⚠️ No historical financial data found. Please import historical financial statements first.")
            st.info("💡 Go to **Setup → Historics** to import Income Statement, Balance Sheet, and Cash Flow data.")
            return
        
        # Forecast periods
        forecast_periods = st.slider(
            "Forecast Periods (months)",
            min_value=12,
            max_value=120,
            value=60,
            step=12,
            key="forecast_config_periods"
        )
        
        st.markdown("---")
        
        # Try to use comprehensive forecast config UI
        try:
            from components.forecast_config_ui import render_forecast_config_ui
            
            configs = render_forecast_config_ui(
                db,
                scenario_id,
                user_id,
                hist_df,
                forecast_periods
            )
            
            if configs:
                st.success("✅ Forecast configuration is ready!")
                st.info("💡 This configuration will be used in the Forecast section when 'Use Trend-Based Forecast' is enabled.")
        
        except ImportError:
            # Fallback to simple trend forecast UI
            if not TREND_ANALYZER_AVAILABLE:
                st.error("Trend Forecast Analyzer is not available.")
                return
            
            from components.trend_forecast_analyzer import render_trend_forecast_ui
            trend_config = render_trend_forecast_ui(db, scenario_id, user_id, hist_df)
            
            if trend_config:
                st.success("✅ Trend forecast configuration is ready!")
                st.info("💡 This configuration will be used in the Forecast section when 'Use Trend-Based Forecast' is enabled.")
    
    except Exception as e:
        st.error(f"❌ Error loading historical data: {str(e)}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())


def render_analysis_tab(db, scenario_id: str, user_id: str, assumptions_set: AssumptionsSet):
    """Render the analysis execution tab."""
    
    section_header("Historical Data Analysis", "Analyze historical data to derive assumptions")
    
    # Load historical data (pass user_id to allow detailed line items aggregation)
    try:
        historical_data = load_historical_data(db, scenario_id, user_id)
    except Exception as e:
        error_msg = str(e)
        if 'duplicate' in error_msg.lower() or 'cannot assemble' in error_msg.lower():
            st.error(f"❌ Error loading historical data: {error_msg}")
            st.info("""
            **The error suggests duplicate period dates in your data.**
            
            **To resolve:**
            1. Go to **Setup → Historics → Detailed Line Items**
            2. Check if you have duplicate periods imported
            3. Clear the line items and re-import, ensuring each period has a unique date
            """)
            with st.expander("🔍 Technical Details"):
                import traceback
                st.code(traceback.format_exc())
        else:
            st.error(f"Error loading historical data: {error_msg}")
        historical_data = pd.DataFrame()
    parts_data = load_parts_data(db, scenario_id, user_id)
    
    # Data summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Historical Periods", len(historical_data) if not historical_data.empty else 0)
    with col2:
        st.metric("Parts/SKUs", len(parts_data) if not parts_data.empty else 0)
    with col3:
        # Calculate data quality
        quality = 0
        if not historical_data.empty:
            quality += 40
            if len(historical_data) >= 12:
                quality += 20
            if len(historical_data) >= 24:
                quality += 20
        if not parts_data.empty:
            quality += 20
        st.metric("Data Quality", f"{quality}%")
    
    if historical_data.empty:
        st.warning("""
        ⚠️ **No Historical Data Found**
        
        Please import historical financial data in **Setup → Historics** before running analysis.
        
        Required data includes:
        - Monthly revenue
        - Cost of goods sold (COGS)
        - Operating expenses (OPEX)
        - At least 12 months of history recommended
        """)
        
        # Debug info to help troubleshoot
        with st.expander("🔧 Debug Info", expanded=False):
            st.write(f"Scenario ID: `{scenario_id}`")
            st.write("Attempted to load from tables: `historic_financials`, `historical_financials`, and detailed line items")
            if hasattr(db, 'client'):
                st.write("Database client: ✅ Available")
                # Try to show what's in the tables
                try:
                    # Check summary tables
                    test_response = db.client.table('historic_financials').select('id, month').eq(
                        'scenario_id', scenario_id
                    ).limit(5).execute()
                    if test_response.data:
                        st.write(f"Found {len(test_response.data)} records in historic_financials")
                        st.write("Sample:", test_response.data[:2])
                    else:
                        st.write("No records found in historic_financials for this scenario")
                    
                    # Check detailed line items
                    try:
                        li_response = db.client.table('historical_income_statement_line_items').select('period_date, line_item_name').eq(
                            'scenario_id', scenario_id
                        ).limit(5).execute()
                        if li_response.data:
                            periods = len(set(r.get('period_date') for r in li_response.data))
                            st.write(f"✅ Found detailed line items: {len(li_response.data)} items across {periods} periods")
                            st.info("💡 The system will aggregate detailed line items into summary format for analysis.")
                        else:
                            st.write("No detailed line items found in historical_income_statement_line_items")
                    except Exception as e:
                        st.write(f"Could not check detailed line items: {e}")
                except Exception as e:
                    st.write(f"Error querying tables: {e}")
            else:
                st.write("Database client: ❌ Not available")
        return
    
    # Show available columns
    with st.expander("📋 Available Data Columns", expanded=False):
        st.write("Financial Data Columns:", list(historical_data.columns))
        if not parts_data.empty:
            st.write("Parts Data Columns:", list(parts_data.columns))
    
    st.markdown("---")
    
    # Run Analysis Button
    if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True, key="run_analysis"):
        with st.spinner("Analyzing historical data..."):
            progress = st.progress(0)
            
            # Financial analysis
            progress.progress(20, "Analyzing financial metrics...")
            analyzer = HistoricalAnalyzer(historical_data)
            financial_assumptions = analyzer.analyze_all_financials()
            
            # NEW: Analyze detailed line items
            st.info("📊 Analyzing detailed line items...")
            line_item_assumptions = analyzer.analyze_detailed_line_items(db, scenario_id, user_id)
            
            if line_item_assumptions:
                assumptions_set.line_item_assumptions = line_item_assumptions
                total_line_items = sum(len(items) for items in line_item_assumptions.values())
                st.success(f"✅ Analyzed {total_line_items} detailed line items across {len(line_item_assumptions)} statement types")
            
            progress.progress(50, "Fitting probability distributions...")
            
            # Manufacturing analysis
            progress.progress(70, "Analyzing manufacturing data...")
            mfg_analyzer = ManufacturingAnalyzer(historical_data, parts_data)
            if not parts_data.empty:
                mfg_assumptions = mfg_analyzer.analyze_part_level()
                assumptions_set.manufacturing_mode = "part_level"
            else:
                mfg_assumptions = mfg_analyzer.analyze_average_level()
                assumptions_set.manufacturing_mode = "average"
            
            progress.progress(90, "Finalizing assumptions...")
            
            # Update assumptions set
            assumptions_set.assumptions = financial_assumptions
            assumptions_set.manufacturing_assumptions = mfg_assumptions
            assumptions_set.historical_periods = len(historical_data)
            assumptions_set.data_quality_score = quality
            assumptions_set.analysis_complete = True
            assumptions_set.assumptions_saved = False  # Mark as unsaved (not explicitly saved by user)
            assumptions_set.updated_at = datetime.now().isoformat()
            if not assumptions_set.created_at:
                assumptions_set.created_at = assumptions_set.updated_at
            
            progress.progress(100, "Complete!")
            
            # Auto-save analysis results to database (Sprint 19 - Auto-persistence)
            # This ensures results persist even if user doesn't explicitly save
            try:
                # Auto-save to database as draft (not marked as explicitly saved)
                assumptions_dict = assumptions_set.to_dict()
                assumptions_dict = convert_to_serializable(assumptions_dict)
                
                # Get existing assumptions data
                existing_assumptions = db.get_scenario_assumptions(assumptions_set.scenario_id, user_id)
                if not existing_assumptions:
                    existing_assumptions = {}
                
                # Store AI assumptions in the data blob (as draft/auto-saved)
                existing_assumptions['ai_assumptions'] = convert_to_serializable(assumptions_dict)
                existing_assumptions['ai_assumptions_auto_saved'] = True  # Mark as auto-saved
                existing_assumptions['ai_assumptions_updated_at'] = assumptions_set.updated_at
                # Note: ai_assumptions_saved remains False until user explicitly saves
                
                # Ensure entire dict is serializable before saving
                existing_assumptions = convert_to_serializable(existing_assumptions)
                
                # Auto-save using the standard assumptions table
                if hasattr(db, 'update_assumptions'):
                    db.update_assumptions(
                        assumptions_set.scenario_id,
                        user_id,
                        existing_assumptions
                    )
            except Exception as e:
                # Non-critical - analysis results are in session state
                # User can still save explicitly if auto-save fails
                pass
            
            # Store in session state
            st.session_state.ai_assumptions_set = assumptions_set
            
        st.success(f"""
        ✅ **Analysis Complete!**
        
        - **{len(financial_assumptions)}** financial assumptions derived
        - **{len(mfg_assumptions)}** manufacturing assumptions derived
        - Analysis mode: **{assumptions_set.manufacturing_mode.replace('_', ' ').title()}**
        
        **Note:** Analysis results have been automatically saved. Review and adjust assumptions in the following tabs, then Save to commit your final assumptions.
        """)
        st.rerun()
    
    # Show analysis results summary if complete
    if assumptions_set.analysis_complete:
        st.markdown("---")
        st.markdown("### 📊 Analysis Results Summary")
        
        if assumptions_set.assumptions:
            def _fmt_num(x):
                try:
                    if x is None:
                        return "N/A"
                    # Handle NaN/inf safely
                    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
                        return "N/A"
                    return f"{float(x):,.0f}"
                except Exception:
                    return "N/A"

            def _fmt_score(x):
                try:
                    if x is None:
                        return "N/A"
                    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
                        return "N/A"
                    return f"{float(x):.2f}"
                except Exception:
                    return "N/A"

            summary_data = []
            for key, assumption in assumptions_set.assumptions.items():
                summary_data.append({
                    'Metric': assumption.display_name,
                    'Category': assumption.category.title(),
                    'Mean': _fmt_num(getattr(assumption, 'historical_mean', None)),
                    'Std Dev': _fmt_num(getattr(assumption, 'historical_std', None)),
                    'Trend': (getattr(assumption, 'historical_trend', None) or 'N/A').title(),
                    'Best Fit': (getattr(getattr(assumption, 'proposed_distribution', None), 'distribution_type', None) or 'N/A').title(),
                    'Fit Score': _fmt_score(getattr(getattr(assumption, 'proposed_distribution', None), 'fit_score', None)),
                })
            
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)


def render_financial_assumptions_tab(db, scenario_id: str, assumptions_set: AssumptionsSet):
    """Render the financial assumptions configuration tab."""
    
    section_header("Financial Assumptions", "Configure Monte Carlo distributions for trend forecasts")
    
    if not assumptions_set.analysis_complete:
        st.info("📊 Run the AI Analysis first to generate assumptions.")
        return
    
    if not assumptions_set.assumptions:
        st.warning("No financial assumptions available. Please run analysis.")
        return
    
    # Clear explanation of what this section does
    st.info("""
**📊 Monte Carlo Distribution Configuration**

This section configures the **probability distributions** used in Monte Carlo simulation. 

**Key Points:**
- Configure distributions for **input elements only** (Revenue, COGS %, OPEX %)
- **Calculated elements** (Gross Profit, EBIT, Tax, Net Income) are **excluded** - they derive from simulated inputs
- Distributions are applied **around trend forecast values** at each period, not fixed historical values
- First configure your **trends** in the Trend Forecast tab, then use this section for **uncertainty bounds**
    """)
    
    # Category filter
    categories = list(set(a.category for a in assumptions_set.assumptions.values()))
    selected_category = st.selectbox(
        "Filter by Category",
        options=['All'] + categories,
        key="fin_category_filter"
    )
    
    st.markdown("---")
    
    # Create editable table with all assumptions
    st.markdown("### 📊 Configure Input Element Distributions")
    st.caption("Select Distribution or Static for each line item, then click 'Save All' at the bottom")
    
    # Import FORECAST_ELEMENTS to filter calculated elements
    from components.forecast_correlation_engine import FORECAST_ELEMENTS
    
    # ==========================================================================
    # CALCULATED ELEMENTS FILTER (CRITICAL FIX)
    # ==========================================================================
    # These elements are NEVER configurable - they are derived from other elements.
    # This list must match exactly with what's calculated in the forecast engine.
    # Both key patterns AND display name patterns must be checked.
    CALCULATED_KEY_PATTERNS = {
        'gross_profit', 'total_gross_profit', 'gp',
        'ebit', 'operating_income', 'operating_profit',
        'ebitda',
        'interest_expense', 'interest', 'finance_cost', 'total_interest_expense',
        'tax', 'income_tax', 'tax_expense', 'taxation', 'total_tax',
        'net_profit', 'net_income', 'profit_after_tax', 'pat',
        'ebt', 'profit_before_tax', 'pbt',
    }
    
    # Display name patterns that indicate calculated elements
    CALCULATED_DISPLAY_PATTERNS = [
        'gross profit', 'gross margin', 
        'ebit', 'operating income', 'operating profit',
        'ebitda',
        'interest expense', 'finance cost',
        'tax', 'taxation', 'income tax',
        'net profit', 'net income', 'profit after tax',
        'profit before tax', 'ebt',
    ]
    
    def is_calculated_element(key: str, display_name: str = None) -> bool:
        """Check if a key or display_name represents a calculated element that should not be configurable."""
        if not key:
            return False
        key_lower = str(key).lower().replace(' ', '_').replace('-', '_')
        
        # Direct key match
        if key_lower in CALCULATED_KEY_PATTERNS:
            return True
        
        # Normalized key (remove 'total_' prefix)
        normalized = key_lower.replace('total_', '')
        if normalized in CALCULATED_KEY_PATTERNS:
            return True
        
        # Check FORECAST_ELEMENTS (handles 'gross_profit', 'ebit', etc.)
        element_def = FORECAST_ELEMENTS.get(key, {})
        if element_def.get('is_calculated', False):
            return True
        
        element_def = FORECAST_ELEMENTS.get(normalized, {})
        if element_def.get('is_calculated', False):
            return True
        
        # Check display name patterns
        if display_name:
            display_lower = display_name.lower()
            for pattern in CALCULATED_DISPLAY_PATTERNS:
                if pattern in display_lower:
                    return True
        
        return False
    
    # Define canonical names to avoid duplicates
    # Priority: total_revenue > revenue, total_cogs > cogs, total_opex > opex, etc.
    canonical_names = {
        'revenue': 'total_revenue',
        'cogs': 'total_cogs',
        'opex': 'total_opex',
    }
    
    # Track which canonical names we've already included
    included_canonical = set()
    seen_display_names = set()
    
    # Prepare data for editable table
    assumptions_data = []
    assumption_keys = []
    
    for key, assumption in assumptions_set.assumptions.items():
        # Skip if category filter doesn't match
        if selected_category != 'All' and assumption.category != selected_category:
            continue
        
        # Skip calculated elements (using robust check with both key and display name)
        if is_calculated_element(key, assumption.display_name):
            continue  # Skip calculated elements (gross_profit, ebit, interest_expense, tax, net_profit)
        
        # Handle duplicates - prefer canonical names
        canonical_key = canonical_names.get(key, key)
        if canonical_key in included_canonical and key != canonical_key:
            continue  # Skip duplicate (e.g., 'revenue' if 'total_revenue' already included)
        
        # Check for duplicate display names
        if assumption.display_name in seen_display_names:
            continue  # Skip if we've already seen this display name
        
        # Mark as included
        if key == canonical_key or canonical_key not in assumptions_set.assumptions:
            included_canonical.add(key)
        else:
            included_canonical.add(canonical_key)
        
        seen_display_names.add(assumption.display_name)
        
        assumption_keys.append(key)
        
        # Format historical mean based on metric type (guard key for None)
        hist_mean = assumption.historical_mean
        key_safe = str(key or "").lower()
        if '%' in assumption.display_name or 'pct' in key_safe or 'margin' in key_safe:
            hist_mean_display = f"{hist_mean:.1f}%" if hist_mean else "N/A"
        else:
            hist_mean_display = f"R {hist_mean:,.0f}" if hist_mean else "N/A"
        
        assumptions_data.append({
            'Line Item': assumption.display_name,
            'Category': assumption.category.title(),
            'Historical Mean': hist_mean_display,
            'AI Distribution': assumption.proposed_distribution.distribution_type.title(),
            'AI Fit Score': assumption.proposed_distribution.fit_score,
            'Use Distribution': assumption.use_distribution,
            'Distribution Type': assumption.user_distribution.distribution_type if assumption.use_distribution else 'static',
            'Static Value': assumption.final_static_value if not assumption.use_distribution else None
        })
    
    if not assumptions_data:
        st.info("No assumptions match the selected category filter.")
        return
    
    # Create DataFrame
    summary_df = pd.DataFrame(assumptions_data)
    
    # Distribution type options
    dist_options = ['normal', 'triangular', 'lognormal', 'beta', 'uniform', 'pert', 'static']
    
    # Create editable table
    edited_df = st.data_editor(
        summary_df,
        column_config={
            'Line Item': st.column_config.TextColumn('Line Item', disabled=True, width="medium"),
            'Category': st.column_config.TextColumn('Category', disabled=True, width="small"),
            'Historical Mean': st.column_config.NumberColumn('Historical Mean', format="%.0f", disabled=True, width="small"),
            'AI Distribution': st.column_config.TextColumn('AI Distribution', disabled=True, width="small"),
            'AI Fit Score': st.column_config.NumberColumn('AI Fit Score', format="%.2f", disabled=True, width="small"),
            'Use Distribution': st.column_config.CheckboxColumn('Use Distribution', width="small"),
            'Distribution Type': st.column_config.SelectboxColumn(
                'Distribution Type',
                options=dist_options,
                width="medium"
            ),
            'Static Value': st.column_config.NumberColumn('Static Value', format="%.2f", width="small")
        },
        hide_index=True,
        use_container_width=True,
        key="financial_assumptions_editor"
    )
    
    # Update assumptions from edited table
    if edited_df is not None:
        for idx, key in enumerate(assumption_keys):
            if idx < len(edited_df):
                row = edited_df.iloc[idx]
                assumption = assumptions_set.assumptions[key]
                
                # Update use_distribution
                assumption.use_distribution = bool(row['Use Distribution'])
                
                # Update distribution type or static value
                if row['Use Distribution']:
                    assumption.user_distribution.distribution_type = str(row['Distribution Type'])
                    # If switching to static, keep current static value
                    if row['Distribution Type'] == 'static':
                        assumption.use_distribution = False
                        assumption.final_static_value = float(row['Static Value']) if pd.notna(row['Static Value']) else assumption.historical_mean
                else:
                    # Using static value
                    static_val = float(row['Static Value']) if pd.notna(row['Static Value']) else assumption.historical_mean
                    assumption.final_static_value = static_val
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("#### ⚡ Quick Actions")
    qcol1, qcol2, qcol3 = st.columns(3)
    
    with qcol1:
        if st.button("✅ Accept All AI Recommendations", use_container_width=True, key="fin_accept_all"):
            for key in assumption_keys:
                assumption = assumptions_set.assumptions[key]
                assumption.user_distribution = DistributionParams.from_dict(
                    assumption.proposed_distribution.to_dict()
                )
                assumption.use_distribution = True
                assumption.user_accepted = True
                assumption.user_modified = False
            st.success("✅ Accepted all AI recommendations")
            st.rerun()
    
    with qcol2:
        if st.button("🔒 Set All to Static", use_container_width=True, key="fin_all_static"):
            for key in assumption_keys:
                assumption = assumptions_set.assumptions[key]
                assumption.use_distribution = False
                assumption.final_static_value = assumption.historical_mean
                assumption.user_modified = True
            st.success("✅ Set all to static values")
            st.rerun()
    
    with qcol3:
        if st.button("↺ Reset All to Defaults", use_container_width=True, key="fin_reset_all"):
            for key in assumption_keys:
                assumption = assumptions_set.assumptions[key]
                assumption.use_distribution = False
                assumption.final_static_value = assumption.historical_mean
                assumption.user_distribution = DistributionParams.from_dict(
                    assumption.proposed_distribution.to_dict()
                )
                assumption.user_modified = False
            st.info("ℹ️ Reset all to defaults")
            st.rerun()
    
    st.markdown("---")
    
    # Individual assumption editors in expandable sections (for detailed parameter adjustment)
    with st.expander("⚙️ Advanced: Adjust Distribution Parameters", expanded=False):
        st.caption("""
        **📊 Distribution Parameters Applied to Trend Forecasts**
        
        Distributions are **not** fixed historical values. Instead:
        - **Mean** = Trend forecast value at each future period
        - **Standard Deviation** = Proportionally scaled from historical (scaled by forecast/historical ratio)
        - This ensures MC simulation confidence intervals follow the trend trajectory
        
        Example: If historical mean=100m, stdev=15m, and Year 3 forecast=200m:
        - MC uses: mean=200m, stdev=30m (15m × 200m/100m)
        """)
        for key in assumption_keys:
            assumption = assumptions_set.assumptions[key]
        render_assumption_editor(key, assumption, assumptions_set)
    
    st.markdown("---")
    
    # Single Save All button
    st.markdown("### 💾 Save All Financial Assumptions")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💾 Save All Financial Assumptions", type="primary", use_container_width=True, key="save_all_financial"):
            try:
                # Mark all as modified
                for key in assumption_keys:
                    assumptions_set.assumptions[key].user_modified = True
                
                # Save to database
                success = save_assumptions_to_db(db, assumptions_set, None)
                
                if success:
                    st.success("✅ All financial assumptions saved successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to save assumptions. Please try again.")
            except Exception as e:
                st.error(f"❌ Error saving assumptions: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
    
    with col2:
        # Show summary of what will be saved
        dist_count = sum(1 for key in assumption_keys if assumptions_set.assumptions[key].use_distribution)
        static_count = len(assumption_keys) - dist_count
        
        st.info(f"""
        **Summary:**
        - **Total Assumptions:** {len(assumption_keys)}
        - **Using Distributions:** {dist_count}
        - **Using Static Values:** {static_count}
        """)
    
    # NEW: Detailed Line Item Assumptions Section
    if assumptions_set.line_item_assumptions:
        st.markdown("---")
        st.markdown("### 📈 Detailed Line Item Assumptions")
        st.caption("AI-generated assumptions for individual line items (e.g., Accounting Fees, Advertising, Depreciation)")
        
        # Tabs for each statement type
        li_tabs = st.tabs([
            f"📊 Income Statement ({sum(len(items) for stmt, items in assumptions_set.line_item_assumptions.items() if stmt == 'income_statement')} items)",
            f"📋 Balance Sheet ({sum(len(items) for stmt, items in assumptions_set.line_item_assumptions.items() if stmt == 'balance_sheet')} items)",
            f"💵 Cash Flow ({sum(len(items) for stmt, items in assumptions_set.line_item_assumptions.items() if stmt == 'cash_flow')} items)"
        ])
        
        statement_types = ['income_statement', 'balance_sheet', 'cash_flow']
        
        for tab_idx, statement_type in enumerate(statement_types):
            with li_tabs[tab_idx]:
                stmt_assumptions = assumptions_set.line_item_assumptions.get(statement_type, {})
                
                if not stmt_assumptions:
                    st.info(f"No {statement_type.replace('_', ' ').title()} line item assumptions available.")
                    continue
                
                # Category filter for this statement type
                categories = list(set(a.category for a in stmt_assumptions.values()))
                selected_li_category = st.selectbox(
                    f"Filter by Category ({statement_type.replace('_', ' ').title()})",
                    options=['All'] + categories,
                    key=f"li_category_filter_{statement_type}"
                )
                
                # Prepare data for table
                li_data = []
                li_keys = []
                
                for line_item_name, assumption in stmt_assumptions.items():
                    if selected_li_category != 'All' and assumption.category != selected_li_category:
                        continue
                    
                    li_keys.append(line_item_name)
                    li_data.append({
                        'Line Item': assumption.display_name,
                        'Category': assumption.category.title(),
                        'Historical Mean': assumption.historical_mean,
                        'Std Dev': assumption.historical_std,
                        'Trend': assumption.historical_trend.title(),
                        'AI Distribution': assumption.proposed_distribution.distribution_type.title(),
                        'AI Fit Score': f"{assumption.proposed_distribution.fit_score:.2f}",
                        'Use Distribution': assumption.use_distribution,
                        'Distribution Type': assumption.user_distribution.distribution_type if assumption.use_distribution else 'static',
                        'Static Value': assumption.final_static_value if not assumption.use_distribution else None
                    })
                
                if not li_data:
                    st.info("No line items match the selected category filter.")
                    continue
                
                # Create DataFrame
                li_df = pd.DataFrame(li_data)
                
                # Distribution type options
                dist_options = ['normal', 'triangular', 'lognormal', 'beta', 'uniform', 'pert', 'static']
                
                # Create editable table
                edited_li_df = st.data_editor(
                    li_df,
                    column_config={
                        'Line Item': st.column_config.TextColumn('Line Item', disabled=True, width="medium"),
                        'Category': st.column_config.TextColumn('Category', disabled=True, width="small"),
                        'Historical Mean': st.column_config.NumberColumn('Historical Mean', format="%.0f", disabled=True, width="small"),
                        'Std Dev': st.column_config.NumberColumn('Std Dev', format="%.0f", disabled=True, width="small"),
                        'Trend': st.column_config.TextColumn('Trend', disabled=True, width="small"),
                        'AI Distribution': st.column_config.TextColumn('AI Distribution', disabled=True, width="small"),
                        'AI Fit Score': st.column_config.TextColumn('AI Fit Score', disabled=True, width="small"),
                        'Use Distribution': st.column_config.CheckboxColumn('Use Distribution', width="small"),
                        'Distribution Type': st.column_config.SelectboxColumn(
                            'Distribution Type',
                            options=dist_options,
                            width="medium"
                        ),
                        'Static Value': st.column_config.NumberColumn('Static Value', format="%.2f", width="small")
                    },
                    hide_index=True,
                    use_container_width=True,
                    key=f"line_item_assumptions_editor_{statement_type}"
                )
                
                # Update assumptions from edited table
                if edited_li_df is not None:
                    for idx, line_item_name in enumerate(li_keys):
                        if idx < len(edited_li_df):
                            row = edited_li_df.iloc[idx]
                            assumption = stmt_assumptions[line_item_name]
                            
                            # Update use_distribution
                            assumption.use_distribution = bool(row['Use Distribution'])
                            
                            # Update distribution type or static value
                            if row['Use Distribution']:
                                assumption.user_distribution.distribution_type = str(row['Distribution Type'])
                                if row['Distribution Type'] == 'static':
                                    assumption.use_distribution = False
                                    assumption.final_static_value = float(row['Static Value']) if pd.notna(row['Static Value']) else assumption.historical_mean
                            else:
                                static_val = float(row['Static Value']) if pd.notna(row['Static Value']) else assumption.historical_mean
                                assumption.final_static_value = static_val
                
                # Summary for this statement type
                dist_count = sum(1 for name in li_keys if stmt_assumptions[name].use_distribution)
                static_count = len(li_keys) - dist_count
                
                st.info(f"""
                **{statement_type.replace('_', ' ').title()} Summary:**
                - **Total Line Items:** {len(li_keys)}
                - **Using Distributions:** {dist_count}
                - **Using Static Values:** {static_count}
                """)


def render_assumption_editor(key: str, assumption: Assumption, assumptions_set: AssumptionsSet):
    """
    Render detailed editor for a single assumption.
    
    NOTE: Distributions are applied around trend forecasts, not fixed historical values.
    The mean at each period = trend forecast value for that period.
    Standard deviation is proportionally scaled from historical.
    """
    """Render an editor for a single assumption."""
    
    with st.expander(f"📊 {assumption.display_name}", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Historical stats
            st.markdown("#### Historical Statistics")
            st.caption("⚠️ **Note:** These are historical values. In MC simulation, distributions are applied around trend forecasts at each period, not these fixed values.")
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("Mean", f"{assumption.historical_mean:,.0f}")
            with stats_col2:
                st.metric("Std Dev", f"{assumption.historical_std:,.0f}")
            with stats_col3:
                st.metric("Trend", assumption.historical_trend.title())
            with stats_col4:
                st.metric("Data Points", assumption.data_points)
        
        with col2:
            # AI Recommendation
            st.markdown("#### AI Recommendation")
            st.info(f"""
            **Distribution:** {assumption.proposed_distribution.distribution_type.title()}
            **Fit Score:** {assumption.proposed_distribution.fit_score:.2f}
            """)
        
        st.markdown("---")
        
        # User selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Your Selection")
            
            use_dist = st.toggle(
                "Use Distribution (for MC Simulation)",
                value=assumption.use_distribution,
                key=f"use_dist_{key}",
                help="Enable to use probability distribution in Monte Carlo simulation. Distribution mean = trend forecast value at each period. Std dev = proportionally scaled from historical."
            )
            assumption.use_distribution = use_dist
            
            if not use_dist:
                # Static value selection
                static_val = st.number_input(
                    f"Static Value ({assumption.unit})",
                    value=float(assumption.final_static_value),
                    key=f"static_{key}",
                    format="%.2f"
                )
                assumption.final_static_value = static_val
            else:
                # Distribution selection
                dist_options = ['normal', 'triangular', 'lognormal', 'beta', 'uniform', 'pert']
                current_dist = assumption.user_distribution.distribution_type
                if current_dist not in dist_options:
                    current_dist = 'normal'
                
                selected_dist = st.selectbox(
                    "Distribution Type",
                    options=dist_options,
                    index=dist_options.index(current_dist),
                    key=f"dist_type_{key}"
                )
                assumption.user_distribution.distribution_type = selected_dist
                
                # Distribution parameters based on type
                if selected_dist == 'normal':
                    assumption.user_distribution.mean = st.number_input(
                        "Mean", value=float(assumption.user_distribution.mean or assumption.historical_mean),
                        key=f"mean_{key}"
                    )
                    assumption.user_distribution.std = st.number_input(
                        "Std Dev", value=float(assumption.user_distribution.std or assumption.historical_std),
                        min_value=0.01, key=f"std_{key}"
                    )
                
                elif selected_dist == 'triangular' or selected_dist == 'pert':
                    assumption.user_distribution.min_val = st.number_input(
                        "Minimum", value=float(assumption.user_distribution.min_val or assumption.historical_min),
                        key=f"min_{key}"
                    )
                    assumption.user_distribution.mode_val = st.number_input(
                        "Most Likely", value=float(assumption.user_distribution.mode_val or assumption.historical_median),
                        key=f"mode_{key}"
                    )
                    assumption.user_distribution.max_val = st.number_input(
                        "Maximum", value=float(assumption.user_distribution.max_val or assumption.historical_max),
                        key=f"max_{key}"
                    )
                
                elif selected_dist == 'uniform':
                    assumption.user_distribution.low = st.number_input(
                        "Low", value=float(assumption.user_distribution.low or assumption.historical_min),
                        key=f"low_{key}"
                    )
                    assumption.user_distribution.high = st.number_input(
                        "High", value=float(assumption.user_distribution.high or assumption.historical_max),
                        key=f"high_{key}"
                    )
        
        with col2:
            # Distribution preview
            if use_dist:
                st.markdown("#### Distribution Preview")
                fig = create_distribution_preview(assumption)
                st.plotly_chart(fig, use_container_width=True, key=f"preview_{key}")
        
        # Accept/Reset buttons
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("✓ Accept AI Recommendation", key=f"accept_{key}"):
                assumption.user_distribution = DistributionParams.from_dict(
                    assumption.proposed_distribution.to_dict()
                )
                assumption.final_static_value = assumption.historical_mean
                assumption.user_accepted = True
                assumption.user_modified = False
                st.success("Accepted AI recommendation")
                st.rerun()
        
        with bcol2:
            if st.button("↺ Reset to Defaults", key=f"reset_{key}"):
                assumption.use_distribution = False
                assumption.final_static_value = assumption.historical_mean
                assumption.user_modified = False
                st.info("Reset to defaults")
                st.rerun()


def render_manufacturing_assumptions_tab(db, scenario_id: str, assumptions_set: AssumptionsSet):
    """Render the manufacturing assumptions configuration tab."""
    
    section_header("Manufacturing Assumptions", "Configure make vs buy decisions based on AI analysis")
    
    if not assumptions_set.analysis_complete:
        st.info("📊 Run the AI Analysis first to generate manufacturing assumptions.")
        return
    
    # Mode indicator
    st.info(f"**Analysis Mode:** {assumptions_set.manufacturing_mode.replace('_', ' ').title()}")
    
    if not assumptions_set.manufacturing_assumptions:
        st.warning("No manufacturing assumptions available.")
        return
    
    st.markdown("---")
    
    # Summary metrics
    total_parts = len(assumptions_set.manufacturing_assumptions)
    avg_mfg_pct = np.mean([a.user_manufacture_pct or a.proposed_manufacture_pct 
                          for a in assumptions_set.manufacturing_assumptions.values()])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parts/Categories", total_parts)
    with col2:
        st.metric("Avg Manufacturing %", f"{avg_mfg_pct:.1f}%")
    with col3:
        parts_to_mfg = sum(1 for a in assumptions_set.manufacturing_assumptions.values() 
                          if (a.user_manufacture_pct or a.proposed_manufacture_pct) > 0)
        st.metric("Parts to Manufacture", parts_to_mfg)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("#### ⚡ Quick Actions")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    
    with qcol1:
        if st.button("Accept All AI", key="mfg_accept_all"):
            for a in assumptions_set.manufacturing_assumptions.values():
                a.user_manufacture_pct = a.proposed_manufacture_pct
                a.user_mfg_cost_pct = a.proposed_mfg_cost_pct
                a.user_accepted = True
            st.success("Accepted all AI recommendations")
            st.rerun()
    
    with qcol2:
        if st.button("Set All 0%", key="mfg_all_0"):
            for a in assumptions_set.manufacturing_assumptions.values():
                a.user_manufacture_pct = 0
            st.rerun()
    
    with qcol3:
        if st.button("Set All 50%", key="mfg_all_50"):
            for a in assumptions_set.manufacturing_assumptions.values():
                a.user_manufacture_pct = 50
            st.rerun()
    
    with qcol4:
        default_cost_pct = st.number_input("Default Mfg Cost %", value=75, min_value=50, max_value=100, key="default_mfg_cost")
    
    st.markdown("---")
    
    # Manufacturing assumptions table
    st.markdown("#### 📋 Manufacturing Decisions")
    
    # Create editable data
    mfg_data = []
    for key, mfg in assumptions_set.manufacturing_assumptions.items():
        mfg_data.append({
            'Part': mfg.part_name[:30],
            'Category': mfg.category,
            'Sell Price': mfg.historical_sell_price,
            'COGS': mfg.historical_cogs,
            'Margin %': mfg.historical_margin_pct,
            'Volume': mfg.historical_volume,
            'AI Mfg %': mfg.proposed_manufacture_pct,
            'Your Mfg %': mfg.user_manufacture_pct or mfg.proposed_manufacture_pct,
            'Mfg Cost %': mfg.user_mfg_cost_pct or mfg.proposed_mfg_cost_pct,
            'Rationale': mfg.rationale[:50] + '...' if len(mfg.rationale) > 50 else mfg.rationale
        })
    
    mfg_df = pd.DataFrame(mfg_data)
    
    edited_df = st.data_editor(
        mfg_df,
        column_config={
            "Part": st.column_config.TextColumn("Part", disabled=True),
            "Category": st.column_config.TextColumn("Category", disabled=True),
            "Sell Price": st.column_config.NumberColumn("Sell Price", format="R %.0f", disabled=True),
            "COGS": st.column_config.NumberColumn("COGS", format="R %.0f", disabled=True),
            "Margin %": st.column_config.NumberColumn("Margin", format="%.1f%%", disabled=True),
            "Volume": st.column_config.NumberColumn("Volume", disabled=True),
            "AI Mfg %": st.column_config.NumberColumn("AI Rec", format="%.0f%%", disabled=True),
            "Your Mfg %": st.column_config.NumberColumn("Your Mfg %", min_value=0, max_value=100, step=5),
            "Mfg Cost %": st.column_config.NumberColumn("Mfg Cost %", min_value=50, max_value=120, step=5),
            "Rationale": st.column_config.TextColumn("AI Rationale", disabled=True, width="large")
        },
        hide_index=True,
        use_container_width=True,
        key="mfg_editor"
    )
    
    # Update assumptions from edited data
    if edited_df is not None:
        for idx, (key, mfg) in enumerate(assumptions_set.manufacturing_assumptions.items()):
            if idx < len(edited_df):
                mfg.user_manufacture_pct = edited_df.iloc[idx]['Your Mfg %']
                mfg.user_mfg_cost_pct = edited_df.iloc[idx]['Mfg Cost %']


def render_save_apply_tab(db, scenario_id: str, assumptions_set: AssumptionsSet, user_id: str = None):
    """Render the save and apply tab.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        assumptions_set: AssumptionsSet to save
        user_id: User ID for RLS compliance
    """
    # Get user_id if not provided
    if not user_id:
        try:
            from supabase_utils import get_user_id
            user_id = get_user_id()
        except:
            st.error("User ID required for saving assumptions")
            return
    
    section_header("Save & Apply Assumptions", "Commit your assumptions and apply to forecast")
    
    if not assumptions_set.analysis_complete:
        st.warning("⚠️ Please run AI Analysis first before saving.")
        return
    
    # Summary
    st.markdown("### 📋 Assumptions Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Financial Assumptions")
        fin_count = len(assumptions_set.assumptions)
        dist_count = sum(1 for a in assumptions_set.assumptions.values() if a.use_distribution)
        static_count = fin_count - dist_count
        
        st.write(f"- **Total:** {fin_count} assumptions")
        st.write(f"- **Static Values:** {static_count}")
        st.write(f"- **Distributions (for MC):** {dist_count}")
    
    with col2:
        st.markdown("#### Manufacturing Assumptions")
        mfg_count = len(assumptions_set.manufacturing_assumptions)
        mfg_parts = sum(1 for a in assumptions_set.manufacturing_assumptions.values() 
                       if (a.user_manufacture_pct or a.proposed_manufacture_pct) > 0)
        
        st.write(f"- **Total Parts:** {mfg_count}")
        st.write(f"- **To Manufacture:** {mfg_parts}")
        st.write(f"- **Mode:** {assumptions_set.manufacturing_mode.replace('_', ' ').title()}")
    
    st.markdown("---")
    
    # Save status
    if assumptions_set.assumptions_saved:
        st.success("✅ Assumptions are saved and ready to apply")
        st.write(f"Last saved: {assumptions_set.updated_at}")
    else:
        st.warning("⚠️ Assumptions have not been saved yet")
    
    st.markdown("---")
    
    # Save button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Assumptions", type="primary", use_container_width=True, key="save_assumptions"):
            with st.spinner("Saving assumptions..."):
                success = save_assumptions_to_db(db, assumptions_set, user_id)
                
                if success:
                    show_success("Assumptions Saved", "Your assumptions have been saved and are ready to use in forecasts.")
                    st.session_state.ai_assumptions_set = assumptions_set
                    st.rerun()
                else:
                    show_error("Save Failed", "Failed to save assumptions. Please check the error messages above and try again.")
    
    with col2:
        # Export button
        if st.button("📤 Export Assumptions", use_container_width=True, key="export_assumptions"):
            export_data = assumptions_set.to_dict()
            json_str = json.dumps(export_data, indent=2, default=str)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"assumptions_{scenario_id}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    st.markdown("---")
    
    # Apply options
    st.markdown("### 🚀 Apply to Modules")
    
    st.info("""
    Once saved, assumptions are automatically available in:
    - **Forecast Section** - Uses static values or distribution means
    - **Manufacturing Strategy** - Uses manufacturing assumptions
    - **Monte Carlo Simulation** - Uses full distributions for stochastic analysis
    """)
    
    # Show what will be applied
    with st.expander("📊 Preview: Financial Assumptions to Apply", expanded=False):
        if assumptions_set.assumptions:
            preview_data = []
            for key, a in assumptions_set.assumptions.items():
                preview_data.append({
                    'Metric': a.display_name,
                    'Value Type': 'Distribution' if a.use_distribution else 'Static',
                    'Value/Distribution': f"{a.user_distribution.distribution_type.title()}" if a.use_distribution else f"{a.final_static_value:,.0f}",
                    'Applied To': 'Forecast, MC Sim' if a.use_distribution else 'Forecast'
                })
            st.dataframe(pd.DataFrame(preview_data), hide_index=True, use_container_width=True)
    
    with st.expander("🏭 Preview: Manufacturing Assumptions to Apply", expanded=False):
        if assumptions_set.manufacturing_assumptions:
            preview_data = []
            for key, m in assumptions_set.manufacturing_assumptions.items():
                mfg_pct = m.user_manufacture_pct or m.proposed_manufacture_pct
                preview_data.append({
                    'Part': m.part_name,
                    'Manufacture %': f"{mfg_pct:.0f}%",
                    'Mfg Cost %': f"{m.user_mfg_cost_pct:.0f}%",
                    'Status': '✓ Will Manufacture' if mfg_pct > 0 else '○ Outsource'
                })
            st.dataframe(pd.DataFrame(preview_data), hide_index=True, use_container_width=True)
    
    # ==========================================================================
    # STORAGE MIGRATION UTILITY (Tech Debt Resolution)
    # ==========================================================================
    st.markdown("---")
    
    with st.expander("🔧 Advanced: Storage Migration", expanded=False):
        st.caption("Consolidate legacy assumption storage into unified configuration")
        try:
            from components.assumption_migration import render_migration_ui
            render_migration_ui(db, scenario_id, user_id)
        except ImportError as e:
            st.info("Migration utility not available.")
        except Exception as e:
            st.error(f"Error loading migration utility: {e}")


# =============================================================================
# HELPER FUNCTIONS FOR OTHER MODULES
# =============================================================================

def get_saved_assumptions(db, scenario_id: str, user_id: str = None) -> Optional[AssumptionsSet]:
    """
    Get saved assumptions for use by other modules (Forecast, Manufacturing, MC Simulation).
    Call this from other modules to retrieve the assumptions.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        user_id: User ID for RLS compliance
    """
    # Get user_id if not provided
    if not user_id:
        try:
            from supabase_utils import get_user_id
            user_id = get_user_id()
        except:
            pass
    
    # First check session state
    if 'ai_assumptions_set' in st.session_state:
        if st.session_state.ai_assumptions_set.scenario_id == scenario_id:
            return st.session_state.ai_assumptions_set
    
    # Then try database
    return load_assumptions_from_db(db, scenario_id, user_id)


def get_assumption_value(assumptions_set: AssumptionsSet, assumption_id: str, 
                         default: float = 0.0) -> float:
    """Get the static value for an assumption."""
    if assumption_id in assumptions_set.assumptions:
        return assumptions_set.assumptions[assumption_id].final_static_value
    return default


def get_assumption_distribution(assumptions_set: AssumptionsSet, assumption_id: str) -> Optional[DistributionParams]:
    """Get the distribution parameters for an assumption (for MC Simulation)."""
    if assumption_id in assumptions_set.assumptions:
        assumption = assumptions_set.assumptions[assumption_id]
        if assumption.use_distribution:
            return assumption.user_distribution
    return None


def get_manufacturing_assumptions(assumptions_set: AssumptionsSet) -> Dict[str, ManufacturingAssumption]:
    """Get all manufacturing assumptions."""
    return assumptions_set.manufacturing_assumptions


def sample_from_assumptions(assumptions_set: AssumptionsSet, n_samples: int = 1000) -> Dict[str, np.ndarray]:
    """
    Generate samples from all distribution-enabled assumptions.
    Used by Monte Carlo Simulation module.
    """
    samples = {}
    
    for key, assumption in assumptions_set.assumptions.items():
        if assumption.use_distribution:
            samples[key] = generate_distribution_samples(assumption.user_distribution, n_samples)
        else:
            samples[key] = np.full(n_samples, assumption.final_static_value)
    
    return samples