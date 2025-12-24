"""
AI Trend Analysis Component
===========================
Sprint 14 Implementation
Date: December 14, 2025

Features:
1. Automated trend detection (linear regression, growth rates, CAGR)
2. Anomaly identification (Z-score, IQR methods)
3. Seasonality detection (monthly/quarterly patterns)
4. AI-generated insights and recommendations
5. Interactive visualizations

Integration:
- Can be used as standalone section via render_trend_analysis_section()
- Can be integrated into forecast section as additional tab
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# COLOR CONSTANTS
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
PURPLE = "#8b5cf6"
ORANGE = "#f59e0b"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrendResult:
    """Results from trend analysis."""
    metric_name: str
    direction: str  # 'increasing', 'decreasing', 'stable'
    strength: str  # 'strong', 'moderate', 'weak'
    slope: float
    r_squared: float
    cagr: float  # Compound Annual Growth Rate
    mom_growth: float  # Month-over-month average growth
    yoy_growth: float  # Year-over-year growth
    forecast_next_3m: List[float] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class AnomalyResult:
    """Results from anomaly detection."""
    metric_name: str
    anomaly_count: int
    anomaly_dates: List[str] = field(default_factory=list)
    anomaly_values: List[float] = field(default_factory=list)
    anomaly_types: List[str] = field(default_factory=list)  # 'spike', 'drop', 'outlier'
    severity: str = 'low'  # 'low', 'medium', 'high'


@dataclass
class SeasonalityResult:
    """Results from seasonality analysis."""
    metric_name: str
    has_seasonality: bool
    seasonal_strength: float  # 0-1
    peak_months: List[int] = field(default_factory=list)
    trough_months: List[int] = field(default_factory=list)
    monthly_indices: Dict[int, float] = field(default_factory=dict)  # month -> index
    quarterly_pattern: Dict[int, float] = field(default_factory=dict)  # quarter -> index


@dataclass
class AIInsight:
    """AI-generated insight."""
    category: str  # 'trend', 'anomaly', 'seasonality', 'performance', 'recommendation'
    priority: str  # 'high', 'medium', 'low'
    title: str
    description: str
    metric: str
    action: str = ""
    confidence: float = 0.0


# =============================================================================
# TREND ANALYSIS ENGINE
# =============================================================================

class TrendAnalyzer:
    """Engine for analyzing trends in financial data."""
    
    def __init__(self, data: pd.DataFrame, date_col: str = 'period_date'):
        """
        Initialize with data.
        
        Args:
            data: DataFrame with financial data
            date_col: Name of date column
        """
        self.data = data.copy()
        self.date_col = date_col
        
        # Ensure date column is datetime
        if date_col in self.data.columns:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            self.data = self.data.sort_values(date_col).reset_index(drop=True)
    
    def analyze_trend(self, metric: str) -> TrendResult:
        """
        Analyze trend for a specific metric.
        
        Args:
            metric: Column name to analyze
            
        Returns:
            TrendResult with trend statistics
        """
        if metric not in self.data.columns:
            return TrendResult(
                metric_name=metric,
                direction='unknown',
                strength='unknown',
                slope=0,
                r_squared=0,
                cagr=0,
                mom_growth=0,
                yoy_growth=0
            )
        
        values = self.data[metric].dropna().values
        if len(values) < 3:
            return TrendResult(
                metric_name=metric,
                direction='insufficient_data',
                strength='unknown',
                slope=0,
                r_squared=0,
                cagr=0,
                mom_growth=0,
                yoy_growth=0
            )
        
        # Linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        r_squared = r_value ** 2
        
        # Determine direction and strength
        if abs(slope) < 0.001 * np.mean(values):
            direction = 'stable'
            strength = 'weak'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Strength based on R-squared
        if r_squared > 0.7:
            strength = 'strong'
        elif r_squared > 0.4:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        # Calculate growth rates
        mom_growth = self._calculate_mom_growth(values)
        yoy_growth = self._calculate_yoy_growth(values)
        cagr = self._calculate_cagr(values)
        
        # Simple forecast for next 3 months
        forecast = [slope * (len(values) + i) + intercept for i in range(1, 4)]
        
        return TrendResult(
            metric_name=metric,
            direction=direction,
            strength=strength,
            slope=slope,
            r_squared=r_squared,
            cagr=cagr,
            mom_growth=mom_growth,
            yoy_growth=yoy_growth,
            forecast_next_3m=forecast,
            confidence=r_squared
        )
    
    def _calculate_mom_growth(self, values: np.ndarray) -> float:
        """Calculate average month-over-month growth rate."""
        if len(values) < 2:
            return 0.0
        
        growths = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                growth = (values[i] - values[i-1]) / abs(values[i-1])
                growths.append(growth)
        
        return np.mean(growths) * 100 if growths else 0.0
    
    def _calculate_yoy_growth(self, values: np.ndarray) -> float:
        """Calculate year-over-year growth rate."""
        if len(values) < 13:
            return 0.0
        
        current = values[-1]
        prior = values[-13]
        
        if prior != 0:
            return ((current - prior) / abs(prior)) * 100
        return 0.0
    
    def _calculate_cagr(self, values: np.ndarray) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(values) < 2 or values[0] <= 0 or values[-1] <= 0:
            return 0.0
        
        years = len(values) / 12
        if years < 0.25:  # Less than 3 months
            return 0.0
        
        cagr = (values[-1] / values[0]) ** (1 / years) - 1
        return cagr * 100


# =============================================================================
# ANOMALY DETECTION ENGINE
# =============================================================================

class AnomalyDetector:
    """Engine for detecting anomalies in financial data."""
    
    def __init__(self, data: pd.DataFrame, date_col: str = 'period_date'):
        self.data = data.copy()
        self.date_col = date_col
        
        if date_col in self.data.columns:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
    
    def detect_anomalies(self, metric: str, method: str = 'zscore', 
                         threshold: float = 2.5) -> AnomalyResult:
        """
        Detect anomalies in a metric.
        
        Args:
            metric: Column name to analyze
            method: 'zscore' or 'iqr'
            threshold: Z-score threshold or IQR multiplier
            
        Returns:
            AnomalyResult with detected anomalies
        """
        if metric not in self.data.columns:
            return AnomalyResult(metric_name=metric, anomaly_count=0)
        
        values = self.data[metric].dropna()
        if len(values) < 5:
            return AnomalyResult(metric_name=metric, anomaly_count=0)
        
        if method == 'zscore':
            anomalies = self._zscore_detection(values, threshold)
        else:
            anomalies = self._iqr_detection(values, threshold)
        
        # Get details of anomalies
        anomaly_dates = []
        anomaly_values = []
        anomaly_types = []
        
        for idx in anomalies:
            if self.date_col in self.data.columns:
                date_val = self.data.iloc[idx][self.date_col]
                if pd.notna(date_val):
                    anomaly_dates.append(str(date_val.strftime('%Y-%m-%d')))
            
            value = values.iloc[idx]
            anomaly_values.append(float(value))
            
            # Determine type
            mean_val = values.mean()
            if value > mean_val * 1.5:
                anomaly_types.append('spike')
            elif value < mean_val * 0.5:
                anomaly_types.append('drop')
            else:
                anomaly_types.append('outlier')
        
        # Determine severity
        anomaly_pct = len(anomalies) / len(values) * 100
        if anomaly_pct > 10:
            severity = 'high'
        elif anomaly_pct > 5:
            severity = 'medium'
        else:
            severity = 'low'
        
        return AnomalyResult(
            metric_name=metric,
            anomaly_count=len(anomalies),
            anomaly_dates=anomaly_dates,
            anomaly_values=anomaly_values,
            anomaly_types=anomaly_types,
            severity=severity
        )
    
    def _zscore_detection(self, values: pd.Series, threshold: float) -> List[int]:
        """Detect anomalies using Z-score method."""
        z_scores = np.abs(stats.zscore(values))
        return list(np.where(z_scores > threshold)[0])
    
    def _iqr_detection(self, values: pd.Series, multiplier: float) -> List[int]:
        """Detect anomalies using IQR method."""
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        
        return list(np.where((values < lower) | (values > upper))[0])


# =============================================================================
# SEASONALITY ANALYSIS ENGINE
# =============================================================================

class SeasonalityAnalyzer:
    """Engine for analyzing seasonality patterns."""
    
    def __init__(self, data: pd.DataFrame, date_col: str = 'period_date'):
        self.data = data.copy()
        self.date_col = date_col
        
        if date_col in self.data.columns:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            self.data['month'] = self.data[date_col].dt.month
            self.data['quarter'] = self.data[date_col].dt.quarter
    
    def analyze_seasonality(self, metric: str) -> SeasonalityResult:
        """
        Analyze seasonality for a metric.
        
        Args:
            metric: Column name to analyze
            
        Returns:
            SeasonalityResult with seasonality patterns
        """
        if metric not in self.data.columns or 'month' not in self.data.columns:
            return SeasonalityResult(metric_name=metric, has_seasonality=False, seasonal_strength=0)
        
        values = self.data[[self.date_col, 'month', 'quarter', metric]].dropna()
        
        if len(values) < 12:
            return SeasonalityResult(metric_name=metric, has_seasonality=False, seasonal_strength=0)
        
        # Calculate monthly indices
        overall_mean = values[metric].mean()
        monthly_means = values.groupby('month')[metric].mean()
        monthly_indices = (monthly_means / overall_mean).to_dict() if overall_mean > 0 else {}
        
        # Calculate quarterly pattern
        quarterly_means = values.groupby('quarter')[metric].mean()
        quarterly_pattern = (quarterly_means / overall_mean).to_dict() if overall_mean > 0 else {}
        
        # Determine if seasonality exists (coefficient of variation of monthly indices)
        if monthly_indices:
            indices_cv = np.std(list(monthly_indices.values())) / np.mean(list(monthly_indices.values()))
            has_seasonality = indices_cv > 0.1  # More than 10% variation
            seasonal_strength = min(indices_cv * 2, 1.0)  # Scale to 0-1
        else:
            has_seasonality = False
            seasonal_strength = 0
        
        # Find peak and trough months
        if monthly_indices:
            sorted_months = sorted(monthly_indices.items(), key=lambda x: x[1], reverse=True)
            peak_months = [m[0] for m in sorted_months[:3]]
            trough_months = [m[0] for m in sorted_months[-3:]]
        else:
            peak_months = []
            trough_months = []
        
        return SeasonalityResult(
            metric_name=metric,
            has_seasonality=has_seasonality,
            seasonal_strength=seasonal_strength,
            peak_months=peak_months,
            trough_months=trough_months,
            monthly_indices=monthly_indices,
            quarterly_pattern=quarterly_pattern
        )


# =============================================================================
# AI INSIGHT GENERATOR
# =============================================================================

class InsightGenerator:
    """Generate AI-powered insights from analysis results."""
    
    MONTH_NAMES = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    
    def generate_insights(self, 
                         trends: Dict[str, TrendResult],
                         anomalies: Dict[str, AnomalyResult],
                         seasonality: Dict[str, SeasonalityResult]) -> List[AIInsight]:
        """
        Generate insights from all analysis results.
        
        Returns:
            List of AIInsight objects sorted by priority
        """
        insights = []
        
        # Trend insights
        insights.extend(self._trend_insights(trends))
        
        # Anomaly insights
        insights.extend(self._anomaly_insights(anomalies))
        
        # Seasonality insights
        insights.extend(self._seasonality_insights(seasonality))
        
        # Cross-metric insights
        insights.extend(self._cross_metric_insights(trends, anomalies, seasonality))
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        insights.sort(key=lambda x: priority_order.get(x.priority, 3))
        
        return insights
    
    def _trend_insights(self, trends: Dict[str, TrendResult]) -> List[AIInsight]:
        """Generate insights from trend analysis."""
        insights = []
        
        for metric, trend in trends.items():
            display_name = metric.replace('_', ' ').title()
            
            # Strong declining revenue/profit - HIGH priority
            if metric in ['total_revenue', 'total_gross_profit', 'net_income', 'ebitda']:
                if trend.direction == 'decreasing' and trend.strength == 'strong':
                    insights.append(AIInsight(
                        category='trend',
                        priority='high',
                        title=f'⚠️ {display_name} Declining Sharply',
                        description=f'{display_name} shows a strong downward trend with {abs(trend.cagr):.1f}% CAGR. '
                                   f'Month-over-month decline averaging {abs(trend.mom_growth):.1f}%.',
                        metric=metric,
                        action='Investigate root causes immediately. Review pricing, costs, and market conditions.',
                        confidence=trend.confidence
                    ))
                elif trend.direction == 'increasing' and trend.strength == 'strong':
                    insights.append(AIInsight(
                        category='trend',
                        priority='medium',
                        title=f'📈 {display_name} Growing Strongly',
                        description=f'{display_name} shows strong growth with {trend.cagr:.1f}% CAGR. '
                                   f'Average monthly growth of {trend.mom_growth:.1f}%.',
                        metric=metric,
                        action='Capitalize on momentum. Consider capacity planning for continued growth.',
                        confidence=trend.confidence
                    ))
            
            # Cost metrics - reversed interpretation
            if metric in ['total_cogs', 'total_opex']:
                if trend.direction == 'increasing' and trend.strength in ['strong', 'moderate']:
                    insights.append(AIInsight(
                        category='trend',
                        priority='high' if trend.strength == 'strong' else 'medium',
                        title=f'💰 {display_name} Rising',
                        description=f'{display_name} increasing at {trend.cagr:.1f}% CAGR. '
                                   f'This may impact profitability if revenue growth doesn\'t keep pace.',
                        metric=metric,
                        action='Review cost structure. Identify opportunities for efficiency improvements.',
                        confidence=trend.confidence
                    ))
            
            # Margin compression warning
            if metric == 'total_gross_profit' and trend.direction == 'decreasing':
                insights.append(AIInsight(
                    category='performance',
                    priority='high',
                    title='📉 Margin Compression Detected',
                    description='Gross profit trend indicates margin compression. '
                               'This suggests pricing pressure or rising input costs.',
                    metric=metric,
                    action='Analyze product mix, pricing strategy, and supplier costs.',
                    confidence=trend.confidence
                ))
        
        return insights
    
    def _anomaly_insights(self, anomalies: Dict[str, AnomalyResult]) -> List[AIInsight]:
        """Generate insights from anomaly detection."""
        insights = []
        
        for metric, anomaly in anomalies.items():
            if anomaly.anomaly_count == 0:
                continue
            
            display_name = metric.replace('_', ' ').title()
            
            # Count types
            spikes = anomaly.anomaly_types.count('spike')
            drops = anomaly.anomaly_types.count('drop')
            
            if anomaly.severity == 'high':
                priority = 'high'
            elif anomaly.severity == 'medium':
                priority = 'medium'
            else:
                priority = 'low'
            
            if spikes > drops:
                insights.append(AIInsight(
                    category='anomaly',
                    priority=priority,
                    title=f'🔺 Unusual Spikes in {display_name}',
                    description=f'Detected {spikes} unusual spike(s) in {display_name}. '
                               f'Latest anomaly: {anomaly.anomaly_dates[-1] if anomaly.anomaly_dates else "N/A"}',
                    metric=metric,
                    action='Investigate causes - could indicate one-time events, data errors, or emerging patterns.',
                    confidence=0.8
                ))
            elif drops > spikes:
                insights.append(AIInsight(
                    category='anomaly',
                    priority=priority,
                    title=f'🔻 Unusual Drops in {display_name}',
                    description=f'Detected {drops} unusual drop(s) in {display_name}. '
                               f'These deviations may indicate operational issues or market changes.',
                    metric=metric,
                    action='Review affected periods for operational issues or external factors.',
                    confidence=0.8
                ))
            else:
                insights.append(AIInsight(
                    category='anomaly',
                    priority=priority,
                    title=f'⚡ Volatility in {display_name}',
                    description=f'Detected {anomaly.anomaly_count} anomalies (both spikes and drops) in {display_name}. '
                               f'This indicates high volatility.',
                    metric=metric,
                    action='Analyze volatility drivers. Consider risk mitigation strategies.',
                    confidence=0.7
                ))
        
        return insights
    
    def _seasonality_insights(self, seasonality: Dict[str, SeasonalityResult]) -> List[AIInsight]:
        """Generate insights from seasonality analysis."""
        insights = []
        
        for metric, seasonal in seasonality.items():
            if not seasonal.has_seasonality:
                continue
            
            display_name = metric.replace('_', ' ').title()
            peak_names = [self.MONTH_NAMES.get(m, str(m)) for m in seasonal.peak_months[:2]]
            trough_names = [self.MONTH_NAMES.get(m, str(m)) for m in seasonal.trough_months[:2]]
            
            insights.append(AIInsight(
                category='seasonality',
                priority='medium',
                title=f'📅 Seasonal Pattern in {display_name}',
                description=f'{display_name} shows {seasonal.seasonal_strength*100:.0f}% seasonal variation. '
                           f'Peak months: {", ".join(peak_names)}. Trough months: {", ".join(trough_names)}.',
                metric=metric,
                action='Align inventory, staffing, and cash flow planning with seasonal patterns.',
                confidence=seasonal.seasonal_strength
            ))
        
        return insights
    
    def _cross_metric_insights(self, 
                               trends: Dict[str, TrendResult],
                               anomalies: Dict[str, AnomalyResult],
                               seasonality: Dict[str, SeasonalityResult]) -> List[AIInsight]:
        """Generate insights from cross-metric analysis."""
        insights = []
        
        # Revenue vs COGS divergence
        rev_trend = trends.get('total_revenue')
        cogs_trend = trends.get('total_cogs')
        
        if rev_trend and cogs_trend:
            if rev_trend.cagr < cogs_trend.cagr:
                margin_erosion = cogs_trend.cagr - rev_trend.cagr
                insights.append(AIInsight(
                    category='performance',
                    priority='high',
                    title='⚠️ Margin Erosion Risk',
                    description=f'COGS growing faster than revenue ({cogs_trend.cagr:.1f}% vs {rev_trend.cagr:.1f}% CAGR). '
                               f'This {margin_erosion:.1f}% gap will compress margins over time.',
                    metric='multiple',
                    action='Review supplier contracts, optimize operations, or adjust pricing strategy.',
                    confidence=min(rev_trend.confidence, cogs_trend.confidence)
                ))
            elif rev_trend.cagr > cogs_trend.cagr + 5:
                insights.append(AIInsight(
                    category='performance',
                    priority='low',
                    title='✅ Margin Expansion Opportunity',
                    description=f'Revenue growing faster than COGS ({rev_trend.cagr:.1f}% vs {cogs_trend.cagr:.1f}% CAGR). '
                               f'Operating leverage is improving.',
                    metric='multiple',
                    action='Sustain growth momentum while maintaining cost discipline.',
                    confidence=min(rev_trend.confidence, cogs_trend.confidence)
                ))
        
        # OPEX efficiency
        opex_trend = trends.get('total_opex')
        if rev_trend and opex_trend:
            if opex_trend.cagr > rev_trend.cagr + 3:
                insights.append(AIInsight(
                    category='performance',
                    priority='medium',
                    title='💼 OPEX Outpacing Revenue',
                    description=f'Operating expenses growing at {opex_trend.cagr:.1f}% vs revenue at {rev_trend.cagr:.1f}%. '
                               f'Operational efficiency declining.',
                    metric='total_opex',
                    action='Conduct operational audit. Identify non-essential expenses to optimize.',
                    confidence=min(rev_trend.confidence, opex_trend.confidence)
                ))
        
        return insights


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_trend_chart(data: pd.DataFrame, metric: str, trend: TrendResult, 
                       date_col: str = 'period_date') -> go.Figure:
    """Create an interactive trend chart with regression line."""
    
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=data[date_col],
        y=data[metric],
        mode='lines+markers',
        name='Actual',
        line=dict(color=BLUE, width=2),
        marker=dict(size=6)
    ))
    
    # Trend line
    x_numeric = np.arange(len(data))
    trend_values = trend.slope * x_numeric + (data[metric].iloc[0] - trend.slope * 0)
    
    fig.add_trace(go.Scatter(
        x=data[date_col],
        y=trend_values,
        mode='lines',
        name=f'Trend (R²={trend.r_squared:.2f})',
        line=dict(color=GOLD, width=2, dash='dash')
    ))
    
    # Moving average
    ma_period = min(6, len(data) // 2)
    if ma_period >= 2:
        ma = data[metric].rolling(window=ma_period).mean()
        fig.add_trace(go.Scatter(
            x=data[date_col],
            y=ma,
            mode='lines',
            name=f'{ma_period}M Moving Avg',
            line=dict(color=PURPLE, width=1.5, dash='dot')
        ))
    
    # Direction indicator
    direction_color = GREEN if trend.direction == 'increasing' else (RED if trend.direction == 'decreasing' else TEXT_MUTED)
    direction_symbol = '↑' if trend.direction == 'increasing' else ('↓' if trend.direction == 'decreasing' else '→')
    
    fig.update_layout(
        title=dict(
            text=f"{metric.replace('_', ' ').title()} {direction_symbol} ({trend.strength.title()} {trend.direction.title()})",
            font=dict(color=TEXT_WHITE, size=16)
        ),
        xaxis=dict(
            title='Period',
            gridcolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_MUTED)
        ),
        yaxis=dict(
            title='Value (R)',
            gridcolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_MUTED),
            tickformat=',.0f'
        ),
        plot_bgcolor=DARKER_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_WHITE),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=BORDER_COLOR,
            font=dict(color=TEXT_WHITE)
        ),
        hovermode='x unified'
    )
    
    return fig


def create_anomaly_chart(data: pd.DataFrame, metric: str, anomaly: AnomalyResult,
                         date_col: str = 'period_date') -> go.Figure:
    """Create chart highlighting anomalies."""
    
    fig = go.Figure()
    
    # Base line
    fig.add_trace(go.Scatter(
        x=data[date_col],
        y=data[metric],
        mode='lines',
        name='Values',
        line=dict(color=BLUE, width=2)
    ))
    
    # Mean line
    mean_val = data[metric].mean()
    fig.add_hline(y=mean_val, line_dash="dash", line_color=TEXT_MUTED,
                  annotation_text=f"Mean: R{mean_val:,.0f}")
    
    # Highlight anomalies
    if anomaly.anomaly_dates:
        anomaly_df = pd.DataFrame({
            'date': pd.to_datetime(anomaly.anomaly_dates),
            'value': anomaly.anomaly_values,
            'type': anomaly.anomaly_types
        })
        
        colors = {'spike': RED, 'drop': ORANGE, 'outlier': PURPLE}
        
        for atype in ['spike', 'drop', 'outlier']:
            type_data = anomaly_df[anomaly_df['type'] == atype]
            if not type_data.empty:
                fig.add_trace(go.Scatter(
                    x=type_data['date'],
                    y=type_data['value'],
                    mode='markers',
                    name=f'{atype.title()}s',
                    marker=dict(
                        color=colors.get(atype, RED),
                        size=12,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    )
                ))
    
    fig.update_layout(
        title=dict(
            text=f"Anomalies in {metric.replace('_', ' ').title()} ({anomaly.anomaly_count} detected)",
            font=dict(color=TEXT_WHITE, size=16)
        ),
        xaxis=dict(gridcolor=BORDER_COLOR, tickfont=dict(color=TEXT_MUTED)),
        yaxis=dict(gridcolor=BORDER_COLOR, tickfont=dict(color=TEXT_MUTED), tickformat=',.0f'),
        plot_bgcolor=DARKER_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_WHITE),
        legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor=BORDER_COLOR),
        hovermode='x unified'
    )
    
    return fig


def create_seasonality_heatmap(seasonal: SeasonalityResult) -> go.Figure:
    """Create a heatmap showing monthly seasonality patterns."""
    
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    values = [seasonal.monthly_indices.get(m, 1.0) for m in months]
    
    # Create color scale (below 1 = red/orange, above 1 = green/blue)
    colors = []
    for v in values:
        if v < 0.9:
            colors.append(RED)
        elif v < 1.0:
            colors.append(ORANGE)
        elif v > 1.1:
            colors.append(GREEN)
        else:
            colors.append(BLUE)
    
    fig = go.Figure(data=[
        go.Bar(
            x=month_names,
            y=values,
            marker_color=colors,
            text=[f'{v:.2f}' for v in values],
            textposition='outside',
            textfont=dict(color=TEXT_WHITE)
        )
    ])
    
    # Add reference line at 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color=TEXT_MUTED,
                  annotation_text="Average (1.0)")
    
    fig.update_layout(
        title=dict(
            text=f"Seasonal Index by Month - {seasonal.metric_name.replace('_', ' ').title()}",
            font=dict(color=TEXT_WHITE, size=16)
        ),
        xaxis=dict(
            title='Month',
            tickfont=dict(color=TEXT_MUTED)
        ),
        yaxis=dict(
            title='Seasonal Index',
            tickfont=dict(color=TEXT_MUTED),
            range=[0, max(values) * 1.2]
        ),
        plot_bgcolor=DARKER_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_WHITE),
        showlegend=False
    )
    
    return fig


def create_summary_dashboard(trends: Dict[str, TrendResult],
                            anomalies: Dict[str, AnomalyResult]) -> go.Figure:
    """Create a summary dashboard with multiple metrics."""
    
    metrics = list(trends.keys())[:6]  # Limit to 6 metrics
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[m.replace('_', ' ').title() for m in metrics],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    for idx, metric in enumerate(metrics):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        trend = trends.get(metric)
        if not trend:
            continue
        
        # Determine color based on direction
        if metric in ['total_cogs', 'total_opex']:
            # For costs, increasing is bad
            color = RED if trend.direction == 'increasing' else GREEN
        else:
            # For revenue/profit, increasing is good
            color = GREEN if trend.direction == 'increasing' else RED
        
        # Add indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=trend.cagr,
                number=dict(suffix="%", font=dict(color=color, size=24)),
                delta=dict(
                    reference=0,
                    increasing=dict(color=GREEN if metric not in ['total_cogs', 'total_opex'] else RED),
                    decreasing=dict(color=RED if metric not in ['total_cogs', 'total_opex'] else GREEN)
                ),
                title=dict(text="CAGR", font=dict(color=TEXT_MUTED, size=12)),
                domain=dict(row=row-1, column=col-1)
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=dict(text="Key Metrics Growth Summary", font=dict(color=TEXT_WHITE, size=18)),
        plot_bgcolor=DARKER_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_WHITE),
        height=400
    )
    
    return fig


# =============================================================================
# MAIN UI RENDERER
# =============================================================================

def render_trend_analysis_section(db, scenario_id: str, user_id: str):
    """
    Main entry point for rendering the AI Trend Analysis section.
    
    Args:
        db: Database handler
        scenario_id: Current scenario ID
        user_id: Current user ID
    """
    st.header("🤖 AI Trend Analysis")
    st.caption("Automated insights from your historical and forecast data")
    
    # Load data
    historical_data = load_analysis_data(db, scenario_id)
    
    if historical_data.empty:
        st.warning("No historical data available for analysis. Please import historical financials first.")
        st.markdown("""
        **To enable AI Trend Analysis:**
        1. Go to **Setup** → **Historics** tab
        2. Import your historical financial data (Income Statement)
        3. Return here to see automated insights
        """)
        return
    
    # Show data summary
    with st.expander("📊 Data Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Periods", len(historical_data))
        with col2:
            if 'period_date' in historical_data.columns:
                date_range = f"{historical_data['period_date'].min().strftime('%b %Y')} - {historical_data['period_date'].max().strftime('%b %Y')}"
                st.metric("Date Range", date_range)
        with col3:
            numeric_cols = historical_data.select_dtypes(include=[np.number]).columns
            st.metric("Metrics Available", len(numeric_cols))
    
    # Analysis settings
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metrics_to_analyze = st.multiselect(
            "📈 Metrics to Analyze",
            options=['total_revenue', 'total_cogs', 'total_gross_profit', 'total_opex', 
                    'ebitda', 'ebit', 'net_income'],
            default=['total_revenue', 'total_gross_profit', 'net_income'],
            key="trend_metrics"
        )
    
    with col2:
        anomaly_method = st.selectbox(
            "🔍 Anomaly Detection Method",
            options=['zscore', 'iqr'],
            format_func=lambda x: 'Z-Score' if x == 'zscore' else 'IQR (Interquartile Range)',
            key="anomaly_method"
        )
    
    with col3:
        anomaly_threshold = st.slider(
            "⚡ Anomaly Sensitivity",
            min_value=1.5,
            max_value=3.5,
            value=2.5,
            step=0.5,
            help="Lower = more sensitive (more anomalies detected)",
            key="anomaly_threshold"
        )
    
    # Run Analysis button
    if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True, key="run_analysis"):
        with st.spinner("Analyzing trends, detecting anomalies, and generating insights..."):
            run_full_analysis(historical_data, metrics_to_analyze, anomaly_method, anomaly_threshold)
    
    # Display results if available
    if 'trend_results' in st.session_state:
        display_analysis_results()


def load_analysis_data(db, scenario_id: str) -> pd.DataFrame:
    """Load historical data for analysis."""
    try:
        # Try multiple methods to get historical financials
        df = pd.DataFrame()
        
        # Method 1: Use db.get_historic_financials() if available
        if hasattr(db, 'get_historic_financials'):
            data = db.get_historic_financials(scenario_id)
            if data:
                df = pd.DataFrame(data)
        
        # Method 2: Use db.get_historical_financials() (alternative naming)
        if df.empty and hasattr(db, 'get_historical_financials'):
            data = db.get_historical_financials(scenario_id)
            if data:
                df = pd.DataFrame(data)
        
        # Method 3: Direct client access
        if df.empty and hasattr(db, 'client'):
            try:
                # Try historic_financials table first
                response = db.client.table('historic_financials').select('*').eq(
                    'scenario_id', scenario_id
                ).execute()
                if response.data:
                    df = pd.DataFrame(response.data)
            except:
                pass
            
            # Try historical_financials table if first failed
            if df.empty:
                try:
                    response = db.client.table('historical_financials').select('*').eq(
                        'scenario_id', scenario_id
                    ).execute()
                    if response.data:
                        df = pd.DataFrame(response.data)
                except:
                    pass
        
        if df.empty:
            return pd.DataFrame()
        
        # Standardize column names
        column_map = {
            'month': 'period_date',
            'revenue': 'total_revenue',
            'cogs': 'total_cogs',
            'gross_profit': 'total_gross_profit',
            'opex': 'total_opex'
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Calculate derived columns if missing
        if 'total_gross_profit' not in df.columns and 'total_revenue' in df.columns and 'total_cogs' in df.columns:
            df['total_gross_profit'] = df['total_revenue'] - df['total_cogs']
        
        # Ensure period_date is datetime
        if 'period_date' in df.columns:
            df['period_date'] = pd.to_datetime(df['period_date'])
            df = df.sort_values('period_date')
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


def run_full_analysis(data: pd.DataFrame, metrics: List[str], 
                      anomaly_method: str, anomaly_threshold: float):
    """Run complete analysis and store results in session state."""
    
    progress = st.progress(0, text="Initializing analysis...")
    
    # Initialize analyzers
    trend_analyzer = TrendAnalyzer(data)
    anomaly_detector = AnomalyDetector(data)
    seasonality_analyzer = SeasonalityAnalyzer(data)
    insight_generator = InsightGenerator()
    
    trends = {}
    anomalies = {}
    seasonality = {}
    
    total_steps = len(metrics) * 3  # 3 analyses per metric
    current_step = 0
    
    for metric in metrics:
        if metric not in data.columns:
            continue
        
        # Trend analysis
        progress.progress(current_step / total_steps, text=f"Analyzing {metric} trends...")
        trends[metric] = trend_analyzer.analyze_trend(metric)
        current_step += 1
        
        # Anomaly detection
        progress.progress(current_step / total_steps, text=f"Detecting anomalies in {metric}...")
        anomalies[metric] = anomaly_detector.detect_anomalies(metric, anomaly_method, anomaly_threshold)
        current_step += 1
        
        # Seasonality analysis
        progress.progress(current_step / total_steps, text=f"Analyzing {metric} seasonality...")
        seasonality[metric] = seasonality_analyzer.analyze_seasonality(metric)
        current_step += 1
    
    # Generate insights
    progress.progress(0.95, text="Generating AI insights...")
    insights = insight_generator.generate_insights(trends, anomalies, seasonality)
    
    # Store in session state
    st.session_state['trend_results'] = trends
    st.session_state['anomaly_results'] = anomalies
    st.session_state['seasonality_results'] = seasonality
    st.session_state['ai_insights'] = insights
    st.session_state['analysis_data'] = data
    
    progress.progress(1.0, text="Analysis complete!")
    st.success(f"✅ Analysis complete! Generated {len(insights)} insights from {len(metrics)} metrics.")


def display_analysis_results():
    """Display the analysis results from session state."""
    
    trends = st.session_state.get('trend_results', {})
    anomalies = st.session_state.get('anomaly_results', {})
    seasonality = st.session_state.get('seasonality_results', {})
    insights = st.session_state.get('ai_insights', [])
    data = st.session_state.get('analysis_data', pd.DataFrame())
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 AI Insights",
        "📈 Trends",
        "⚡ Anomalies",
        "📅 Seasonality",
        "📊 Summary"
    ])
    
    with tab1:
        render_insights_tab(insights)
    
    with tab2:
        render_trends_tab(data, trends)
    
    with tab3:
        render_anomalies_tab(data, anomalies)
    
    with tab4:
        render_seasonality_tab(seasonality)
    
    with tab5:
        render_summary_tab(trends, anomalies, seasonality, insights)


def render_insights_tab(insights: List[AIInsight]):
    """Render the AI Insights tab."""
    
    if not insights:
        st.info("No significant insights detected. Your data appears stable.")
        return
    
    st.subheader("🎯 AI-Generated Insights")
    st.caption(f"Found {len(insights)} actionable insights from your data")
    
    # Filter by priority
    col1, col2 = st.columns([1, 3])
    with col1:
        priority_filter = st.multiselect(
            "Filter by Priority",
            options=['high', 'medium', 'low'],
            default=['high', 'medium'],
            key="insight_priority_filter"
        )
    
    filtered_insights = [i for i in insights if i.priority in priority_filter]
    
    # Display insights
    for insight in filtered_insights:
        priority_colors = {'high': RED, 'medium': ORANGE, 'low': GREEN}
        priority_icons = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        
        with st.container():
            st.markdown(f"""
            <div style="
                background: {DARK_BG};
                border-left: 4px solid {priority_colors.get(insight.priority, BLUE)};
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 0 8px 8px 0;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="color: {TEXT_WHITE}; margin: 0;">{insight.title}</h4>
                    <span style="color: {TEXT_MUTED}; font-size: 0.8rem;">
                        {priority_icons.get(insight.priority, '')} {insight.priority.upper()} | 
                        {insight.category.title()} | 
                        Confidence: {insight.confidence*100:.0f}%
                    </span>
                </div>
                <p style="color: {TEXT_MUTED}; margin: 0.5rem 0;">{insight.description}</p>
                <div style="
                    background: rgba(212, 165, 55, 0.1);
                    padding: 0.5rem;
                    border-radius: 4px;
                    margin-top: 0.5rem;
                ">
                    <strong style="color: {GOLD};">💡 Recommended Action:</strong>
                    <span style="color: {TEXT_WHITE};"> {insight.action}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_trends_tab(data: pd.DataFrame, trends: Dict[str, TrendResult]):
    """Render the Trends analysis tab."""
    
    if not trends:
        st.info("No trend data available.")
        return
    
    st.subheader("📈 Trend Analysis")
    
    # Metric selector
    metric_options = list(trends.keys())
    selected_metric = st.selectbox(
        "Select Metric",
        options=metric_options,
        format_func=lambda x: x.replace('_', ' ').title(),
        key="trend_metric_select"
    )
    
    if selected_metric and selected_metric in trends:
        trend = trends[selected_metric]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            direction_icon = '📈' if trend.direction == 'increasing' else ('📉' if trend.direction == 'decreasing' else '➡️')
            st.metric("Direction", f"{direction_icon} {trend.direction.title()}")
        
        with col2:
            st.metric("CAGR", f"{trend.cagr:.1f}%")
        
        with col3:
            st.metric("MoM Growth", f"{trend.mom_growth:.1f}%")
        
        with col4:
            st.metric("R² (Fit)", f"{trend.r_squared:.2f}")
        
        # Trend chart
        if selected_metric in data.columns and 'period_date' in data.columns:
            fig = create_trend_chart(data, selected_metric, trend)
            st.plotly_chart(fig, use_container_width=True)
        
        # Forecast
        if trend.forecast_next_3m:
            st.markdown("#### 📊 3-Month Trend Projection")
            st.caption("Based on linear regression extrapolation")
            
            fcol1, fcol2, fcol3 = st.columns(3)
            for i, (col, val) in enumerate(zip([fcol1, fcol2, fcol3], trend.forecast_next_3m)):
                with col:
                    st.metric(f"Month +{i+1}", f"R{val:,.0f}")


def render_anomalies_tab(data: pd.DataFrame, anomalies: Dict[str, AnomalyResult]):
    """Render the Anomalies tab."""
    
    if not anomalies:
        st.info("No anomaly data available.")
        return
    
    st.subheader("⚡ Anomaly Detection")
    
    # Summary
    total_anomalies = sum(a.anomaly_count for a in anomalies.values())
    high_severity = sum(1 for a in anomalies.values() if a.severity == 'high')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Anomalies", total_anomalies)
    with col2:
        st.metric("High Severity Metrics", high_severity)
    with col3:
        st.metric("Metrics Analyzed", len(anomalies))
    
    st.markdown("---")
    
    # Metric selector
    metric_options = [m for m in anomalies.keys() if anomalies[m].anomaly_count > 0]
    
    if not metric_options:
        st.success("✅ No anomalies detected in your data. All metrics are within normal ranges.")
        return
    
    selected_metric = st.selectbox(
        "Select Metric",
        options=metric_options,
        format_func=lambda x: f"{x.replace('_', ' ').title()} ({anomalies[x].anomaly_count} anomalies)",
        key="anomaly_metric_select"
    )
    
    if selected_metric:
        anomaly = anomalies[selected_metric]
        
        # Chart
        if selected_metric in data.columns and 'period_date' in data.columns:
            fig = create_anomaly_chart(data, selected_metric, anomaly)
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly details
        if anomaly.anomaly_dates:
            st.markdown("#### 📋 Anomaly Details")
            
            anomaly_df = pd.DataFrame({
                'Date': anomaly.anomaly_dates,
                'Value': [f"R{v:,.0f}" for v in anomaly.anomaly_values],
                'Type': [t.title() for t in anomaly.anomaly_types]
            })
            
            st.dataframe(anomaly_df, use_container_width=True, hide_index=True)


def render_seasonality_tab(seasonality: Dict[str, SeasonalityResult]):
    """Render the Seasonality tab."""
    
    if not seasonality:
        st.info("No seasonality data available.")
        return
    
    st.subheader("📅 Seasonality Analysis")
    
    # Find metrics with seasonality
    seasonal_metrics = {k: v for k, v in seasonality.items() if v.has_seasonality}
    
    if not seasonal_metrics:
        st.info("No significant seasonal patterns detected in your data.")
        return
    
    st.caption(f"Found seasonal patterns in {len(seasonal_metrics)} metrics")
    
    # Metric selector
    selected_metric = st.selectbox(
        "Select Metric",
        options=list(seasonal_metrics.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        key="seasonality_metric_select"
    )
    
    if selected_metric:
        seasonal = seasonal_metrics[selected_metric]
        
        # Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Seasonal Strength", f"{seasonal.seasonal_strength*100:.0f}%")
        
        with col2:
            peak_names = [['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m-1] 
                         for m in seasonal.peak_months[:2]]
            st.metric("Peak Months", ", ".join(peak_names))
        
        with col3:
            trough_names = [['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m-1] 
                           for m in seasonal.trough_months[:2]]
            st.metric("Trough Months", ", ".join(trough_names))
        
        # Seasonality heatmap
        fig = create_seasonality_heatmap(seasonal)
        st.plotly_chart(fig, use_container_width=True)
        
        # Quarterly pattern
        if seasonal.quarterly_pattern:
            st.markdown("#### 📊 Quarterly Pattern")
            
            quarters = ['Q1', 'Q2', 'Q3', 'Q4']
            q_values = [seasonal.quarterly_pattern.get(i+1, 1.0) for i in range(4)]
            
            qcol1, qcol2, qcol3, qcol4 = st.columns(4)
            for col, q, v in zip([qcol1, qcol2, qcol3, qcol4], quarters, q_values):
                with col:
                    delta_pct = (v - 1) * 100
                    st.metric(q, f"{v:.2f}", f"{delta_pct:+.1f}%")


def render_summary_tab(trends: Dict[str, TrendResult], 
                       anomalies: Dict[str, AnomalyResult],
                       seasonality: Dict[str, SeasonalityResult],
                       insights: List[AIInsight]):
    """Render the Summary tab."""
    
    st.subheader("📊 Analysis Summary")
    
    # High-level stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        increasing = sum(1 for t in trends.values() if t.direction == 'increasing')
        st.metric("📈 Increasing Trends", increasing)
    
    with col2:
        decreasing = sum(1 for t in trends.values() if t.direction == 'decreasing')
        st.metric("📉 Decreasing Trends", decreasing)
    
    with col3:
        total_anomalies = sum(a.anomaly_count for a in anomalies.values())
        st.metric("⚡ Total Anomalies", total_anomalies)
    
    with col4:
        seasonal_count = sum(1 for s in seasonality.values() if s.has_seasonality)
        st.metric("📅 Seasonal Patterns", seasonal_count)
    
    st.markdown("---")
    
    # Trend summary table
    st.markdown("#### 📈 Trend Summary")
    
    trend_data = []
    for metric, trend in trends.items():
        direction_icon = '↑' if trend.direction == 'increasing' else ('↓' if trend.direction == 'decreasing' else '→')
        trend_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Direction': f"{direction_icon} {trend.direction.title()}",
            'Strength': trend.strength.title(),
            'CAGR': f"{trend.cagr:.1f}%",
            'MoM Growth': f"{trend.mom_growth:.1f}%",
            'R²': f"{trend.r_squared:.2f}"
        })
    
    if trend_data:
        st.dataframe(pd.DataFrame(trend_data), use_container_width=True, hide_index=True)
    
    # Insights by priority
    st.markdown("---")
    st.markdown("#### 🎯 Insights by Priority")
    
    priority_counts = {'high': 0, 'medium': 0, 'low': 0}
    for insight in insights:
        priority_counts[insight.priority] = priority_counts.get(insight.priority, 0) + 1
    
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        st.markdown(f"""
        <div style="background: {RED}22; border: 1px solid {RED}; border-radius: 8px; padding: 1rem; text-align: center;">
            <h2 style="color: {RED}; margin: 0;">{priority_counts['high']}</h2>
            <p style="color: {TEXT_MUTED}; margin: 0;">High Priority</p>
        </div>
        """, unsafe_allow_html=True)
    
    with pcol2:
        st.markdown(f"""
        <div style="background: {ORANGE}22; border: 1px solid {ORANGE}; border-radius: 8px; padding: 1rem; text-align: center;">
            <h2 style="color: {ORANGE}; margin: 0;">{priority_counts['medium']}</h2>
            <p style="color: {TEXT_MUTED}; margin: 0;">Medium Priority</p>
        </div>
        """, unsafe_allow_html=True)
    
    with pcol3:
        st.markdown(f"""
        <div style="background: {GREEN}22; border: 1px solid {GREEN}; border-radius: 8px; padding: 1rem; text-align: center;">
            <h2 style="color: {GREEN}; margin: 0;">{priority_counts['low']}</h2>
            <p style="color: {TEXT_MUTED}; margin: 0;">Low Priority</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Export button
    st.markdown("---")
    
    if st.button("📥 Export Analysis Report", type="secondary", use_container_width=True):
        export_analysis_report(trends, anomalies, seasonality, insights)


def export_analysis_report(trends, anomalies, seasonality, insights):
    """Export analysis results to CSV."""
    
    # Build insights export
    insights_data = []
    for i in insights:
        insights_data.append({
            'Priority': i.priority.upper(),
            'Category': i.category.title(),
            'Title': i.title,
            'Description': i.description,
            'Metric': i.metric,
            'Action': i.action,
            'Confidence': f"{i.confidence*100:.0f}%"
        })
    
    if insights_data:
        df = pd.DataFrame(insights_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Insights CSV",
            data=csv,
            file_name=f"ai_insights_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def get_trend_analysis_tab_content(db, scenario_id: str, user_id: str):
    """
    Helper function for integrating trend analysis as a tab in forecast section.
    Returns the tab content without the full section wrapper.
    """
    historical_data = load_analysis_data(db, scenario_id)
    
    if historical_data.empty:
        st.info("📊 Import historical data to enable AI Trend Analysis")
        return
    
    # Quick analysis with defaults
    metrics = ['total_revenue', 'total_gross_profit', 'total_opex', 'net_income']
    available_metrics = [m for m in metrics if m in historical_data.columns]
    
    if not available_metrics:
        st.warning("Required metrics not found in historical data.")
        return
    
    if st.button("🚀 Quick Analysis", type="primary", key="quick_trend_analysis"):
        run_full_analysis(historical_data, available_metrics, 'zscore', 2.5)
    
    if 'ai_insights' in st.session_state:
        insights = st.session_state['ai_insights']
        
        # Show top 5 insights
        st.markdown("#### 🎯 Top Insights")
        for insight in insights[:5]:
            priority_colors = {'high': RED, 'medium': ORANGE, 'low': GREEN}
            st.markdown(f"""
            <div style="
                background: {DARK_BG}; 
                border-left: 3px solid {priority_colors.get(insight.priority, BLUE)};
                padding: 0.5rem 1rem;
                margin: 0.25rem 0;
                border-radius: 0 4px 4px 0;
            ">
                <strong style="color: {TEXT_WHITE};">{insight.title}</strong>
                <p style="color: {TEXT_MUTED}; margin: 0; font-size: 0.85rem;">{insight.description[:150]}...</p>
            </div>
            """, unsafe_allow_html=True)
