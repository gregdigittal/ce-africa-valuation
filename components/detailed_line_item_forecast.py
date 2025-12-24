"""
Detailed Line Item Forecast Engine
==================================
Generates detailed line item forecasts matching historical granularity.

This module:
1. Loads historical detailed line items from database
2. Applies trend analysis/correlation to each line item
3. Generates forecasts for all individual line items
4. Saves to forecast_line_items tables
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from components.trend_forecast_analyzer import TrendForecastAnalyzer, TrendFunction
from components.forecast_correlation_engine import ForecastCorrelationEngine, ForecastMethod


def load_historical_line_items(
    db,
    scenario_id: str,
    statement_type: str = 'income_statement'
) -> pd.DataFrame:
    """
    Load historical detailed line items from database.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        statement_type: 'income_statement', 'balance_sheet', or 'cash_flow'
    
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
            response = db.client.table(table_name).select('*').eq(
                'scenario_id', scenario_id
            ).order('period_date').execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                if 'period_date' in df.columns:
                    df['period_date'] = pd.to_datetime(df['period_date'])
                return df
    except Exception as e:
        st.warning(f"Could not load historical {statement_type} line items: {e}")
    
    return pd.DataFrame()


def get_unique_line_items(historical_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Get unique line items from historical data with their categories.
    
    Returns:
        List of dicts with keys: line_item_name, category, sub_category
    """
    if historical_df.empty:
        return []
    
    unique_items = historical_df.groupby(['line_item_name', 'category', 'sub_category']).size().reset_index()
    unique_items = unique_items[['line_item_name', 'category', 'sub_category']].to_dict('records')
    
    return unique_items


def forecast_line_item(
    line_item_name: str,
    historical_series: pd.Series,
    forecast_periods: int,
    method: str = 'trend_fit',
    correlation_source: Optional[str] = None,
    correlation_pct: Optional[float] = None,
    fixed_value: Optional[float] = None,
    trend_function: TrendFunction = TrendFunction.LINEAR
) -> pd.Series:
    """
    Forecast a single line item using specified method.
    
    Args:
        line_item_name: Name of the line item
        historical_series: Historical values (indexed by date)
        forecast_periods: Number of periods to forecast
        method: 'trend_fit', 'correlation', 'fixed', 'percentage'
        correlation_source: Source element name if using correlation
        correlation_pct: Percentage of source if using correlation
        fixed_value: Fixed value if using fixed method
        trend_function: Trend function type for trend fitting
    
    Returns:
        Series of forecasted values
    """
    if method == 'fixed' and fixed_value is not None:
        return pd.Series([fixed_value] * forecast_periods)
    
    if method == 'correlation' and correlation_source and correlation_pct is not None:
        # This will be handled by the caller with the source forecast
        return pd.Series([0] * forecast_periods)  # Placeholder
    
    # Default to trend fitting
    if len(historical_series) < 3:
        # Not enough data, use last value
        last_value = historical_series.iloc[-1] if len(historical_series) > 0 else 0
        return pd.Series([last_value] * forecast_periods)
    
    try:
        analyzer = TrendForecastAnalyzer()
        trend_params = analyzer.fit_trend(historical_series, trend_function)
        
        # Generate forecast periods
        last_date = historical_series.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + relativedelta(months=1),
            periods=forecast_periods,
            freq='MS'
        )
        
        forecast_values = []
        for i, forecast_date in enumerate(forecast_dates):
            periods_ahead = i + 1
            value = analyzer.forecast_value(trend_params, periods_ahead)
            forecast_values.append(value)
        
        return pd.Series(forecast_values)
    except Exception as e:
        st.warning(f"Error forecasting {line_item_name}: {e}")
        # Fallback to last value
        last_value = historical_series.iloc[-1] if len(historical_series) > 0 else 0
        return pd.Series([last_value] * forecast_periods)


def generate_detailed_forecast(
    db,
    scenario_id: str,
    user_id: str,
    statement_type: str,
    forecast_periods: int,
    start_date: date,
    source_forecast: Optional[pd.DataFrame] = None,
    assumptions: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Generate detailed line item forecasts for a financial statement.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        user_id: User ID
        statement_type: 'income_statement', 'balance_sheet', or 'cash_flow'
        forecast_periods: Number of periods to forecast
        start_date: Start date for forecast
        source_forecast: Summary forecast DataFrame (for correlations)
        assumptions: Forecast assumptions/configurations
    
    Returns:
        DataFrame with columns: period_date, line_item_name, category, sub_category, amount, forecast_method
    """
    # Load historical line items
    historical_df = load_historical_line_items(db, scenario_id, statement_type)
    
    if historical_df.empty:
        st.warning(f"No historical {statement_type} line items found. Import historical data first.")
        return pd.DataFrame()
    
    # Get unique line items
    unique_items = get_unique_line_items(historical_df)
    
    if not unique_items:
        return pd.DataFrame()
    
    # Generate forecast dates
    forecast_dates = pd.date_range(
        start=start_date,
        periods=forecast_periods,
        freq='MS'
    )
    
    forecast_results = []
    
    for item in unique_items:
        line_item_name = item['line_item_name']
        category = item['category']
        sub_category = item.get('sub_category')
        
        # Get historical series for this line item
        item_historical = historical_df[
            historical_df['line_item_name'] == line_item_name
        ].set_index('period_date')['amount'].sort_index()
        
        if item_historical.empty:
            continue
        
        # Determine forecast method from assumptions or default to trend
        method = 'trend_fit'
        correlation_source = None
        correlation_pct = None
        fixed_value = None
        trend_function = TrendFunction.LINEAR
        
        if assumptions:
            # Try to get from line item assumptions (AI-generated)
            line_item_assumptions = assumptions.get('line_item_assumptions', {})
            stmt_assumptions = line_item_assumptions.get(statement_type, {})
            line_item_assumption = stmt_assumptions.get(line_item_name)
            
            if line_item_assumption:
                # Use AI assumption
                if line_item_assumption.get('use_distribution', False):
                    # Use distribution mean as static for now (can be enhanced to use full distribution)
                    fixed_value = line_item_assumption.get('final_static_value', 
                                                          line_item_assumption.get('historical_mean', 0))
                    method = 'fixed'
                else:
                    fixed_value = line_item_assumption.get('final_static_value',
                                                          line_item_assumption.get('historical_mean', 0))
                    method = 'fixed'
            
            # Also check manual configs (user overrides)
            item_config = assumptions.get('line_item_configs', {}).get(line_item_name, {})
            if item_config:
                method = item_config.get('method', method)
                correlation_source = item_config.get('correlation_source')
                correlation_pct = item_config.get('correlation_pct')
                fixed_value = item_config.get('fixed_value', fixed_value)
                trend_function_str = item_config.get('trend_function', 'linear')
                try:
                    trend_function = TrendFunction[trend_function_str.upper()]
                except:
                    trend_function = TrendFunction.LINEAR
        
        # Generate forecast
        if method == 'correlation' and correlation_source and source_forecast is not None:
            # Forecast as percentage of source element
            if correlation_source in source_forecast.columns:
                source_values = source_forecast[correlation_source].values
                forecast_values = source_values * (correlation_pct / 100) if correlation_pct else source_values * 0.01
            else:
                # Fallback to trend
                forecast_values = forecast_line_item(
                    line_item_name, item_historical, forecast_periods,
                    'trend_fit', trend_function=trend_function
                ).values
        else:
            forecast_values = forecast_line_item(
                line_item_name, item_historical, forecast_periods,
                method, correlation_source, correlation_pct, fixed_value, trend_function
            ).values
        
        # Create forecast records
        for i, forecast_date in enumerate(forecast_dates):
            forecast_results.append({
                'period_date': forecast_date.strftime('%Y-%m-%d'),
                'line_item_name': line_item_name,
                'category': category,
                'sub_category': sub_category,
                'amount': float(forecast_values[i]),
                'forecast_method': method,
                'forecast_source': correlation_source if method == 'correlation' else None
            })
    
    return pd.DataFrame(forecast_results)


def save_forecast_line_items(
    db,
    scenario_id: str,
    user_id: str,
    snapshot_id: Optional[str],
    statement_type: str,
    forecast_df: pd.DataFrame
) -> bool:
    """
    Save detailed line item forecasts to database.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        user_id: User ID
        snapshot_id: Forecast snapshot ID (optional)
        statement_type: 'income_statement', 'balance_sheet', or 'cash_flow'
        forecast_df: DataFrame with forecast line items
    
    Returns:
        True if successful
    """
    if forecast_df.empty:
        return False
    
    table_map = {
        'income_statement': 'forecast_income_statement_line_items',
        'balance_sheet': 'forecast_balance_sheet_line_items',
        'cash_flow': 'forecast_cashflow_line_items'
    }
    
    table_name = table_map.get(statement_type)
    if not table_name:
        return False
    
    try:
        # Prepare records
        records = []
        for _, row in forecast_df.iterrows():
            record = {
                'scenario_id': scenario_id,
                'user_id': user_id,
                'period_date': row['period_date'],
                'line_item_name': row['line_item_name'],
                'category': row['category'],
                'sub_category': row.get('sub_category'),
                'amount': float(row['amount']),
                'forecast_method': row.get('forecast_method', 'trend_fit'),
                'forecast_source': row.get('forecast_source')
            }
            if snapshot_id:
                record['snapshot_id'] = snapshot_id
            records.append(record)
        
        # Upsert records
        if hasattr(db, 'client'):
            # Delete existing forecasts for this snapshot/scenario
            delete_query = db.client.table(table_name).delete()
            if snapshot_id:
                delete_query = delete_query.eq('snapshot_id', snapshot_id)
            else:
                delete_query = delete_query.eq('scenario_id', scenario_id).is_('snapshot_id', 'null')
            delete_query.execute()
            
            # Insert new forecasts
            if records:
                db.client.table(table_name).insert(records).execute()
            
            return True
    except Exception as e:
        st.error(f"Error saving {statement_type} line item forecasts: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False
    
    return False


def load_forecast_line_items(
    db,
    scenario_id: str,
    snapshot_id: Optional[str] = None,
    statement_type: str = 'income_statement'
) -> pd.DataFrame:
    """
    Load detailed line item forecasts from database.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        snapshot_id: Forecast snapshot ID (optional)
        statement_type: 'income_statement', 'balance_sheet', or 'cash_flow'
    
    Returns:
        DataFrame with forecast line items
    """
    table_map = {
        'income_statement': 'forecast_income_statement_line_items',
        'balance_sheet': 'forecast_balance_sheet_line_items',
        'cash_flow': 'forecast_cashflow_line_items'
    }
    
    table_name = table_map.get(statement_type)
    if not table_name:
        return pd.DataFrame()
    
    try:
        if hasattr(db, 'client'):
            query = db.client.table(table_name).select('*').eq('scenario_id', scenario_id)
            
            if snapshot_id:
                query = query.eq('snapshot_id', snapshot_id)
            else:
                query = query.is_('snapshot_id', 'null')
            
            response = query.order('period_date').order('line_item_name').execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                if 'period_date' in df.columns:
                    df['period_date'] = pd.to_datetime(df['period_date'])
                return df
    except Exception as e:
        st.warning(f"Could not load {statement_type} forecast line items: {e}")
    
    return pd.DataFrame()


def aggregate_line_items_to_summary(
    forecast_df: pd.DataFrame,
    statement_type: str
) -> pd.DataFrame:
    """
    Aggregate detailed line items to summary totals.
    
    Args:
        forecast_df: DataFrame with detailed line items
        statement_type: 'income_statement', 'balance_sheet', or 'cash_flow'
    
    Returns:
        DataFrame with summary totals by period
    """
    if forecast_df.empty:
        return pd.DataFrame()
    
    # Group by period and category, then aggregate
    summary = forecast_df.groupby(['period_date', 'category'])['amount'].sum().reset_index()
    
    # Pivot to get categories as columns
    summary_pivot = summary.pivot(index='period_date', columns='category', values='amount').reset_index()
    
    # Calculate totals based on statement type
    if statement_type == 'income_statement':
        if 'Revenue' in summary_pivot.columns:
            summary_pivot['total_revenue'] = summary_pivot['Revenue']
        if 'Cost of Sales' in summary_pivot.columns:
            summary_pivot['total_cogs'] = -summary_pivot['Cost of Sales']  # Expenses are negative
        if 'Operating Expenses' in summary_pivot.columns:
            summary_pivot['total_opex'] = -summary_pivot['Operating Expenses']
        if 'Gross Profit' in summary_pivot.columns:
            summary_pivot['total_gross_profit'] = summary_pivot['Gross Profit']
    
    return summary_pivot
