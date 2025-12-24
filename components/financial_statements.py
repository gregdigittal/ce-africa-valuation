"""
Financial Statements Component (Sprint 8)
==========================================
Professional 3-statement financial view with:
- Historic actuals + forecast periods
- Proper labeling of Actual vs Forecast periods
- Monthly/Annual toggle with annual subtotals
- Current year mixed (YTD Actual + Forecast) handling
- Full Balance Sheet projections
- Full Cash Flow Statement
- Revenue segmentation by customer type
- AI Assumptions Integration (NEW)

Crusher Equipment Africa - Empowering Mining Excellence
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# AI Assumptions Integration
try:
    from components.ai_assumptions_integration import (
        render_ai_assumptions_summary,
        render_ai_assumption_badge,
        get_assumption_with_source
    )
    AI_INTEGRATION_AVAILABLE = True
except ImportError:
    AI_INTEGRATION_AVAILABLE = False


# =============================================================================
# STYLING CONSTANTS
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
ACTUAL_COLOR = "#10b981"  # Green for actuals
FORECAST_COLOR = "#3b82f6"  # Blue for forecast
MIXED_COLOR = "#f59e0b"  # Amber for mixed periods


# =============================================================================
# DATABASE LOADING FUNCTIONS
# =============================================================================

def load_historical_income_statement(db, scenario_id: str) -> pd.DataFrame:
    """Load historical income statement data from database.
    
    Tries multiple tables in order of preference:
    1. historical_income_statement (detailed format)
    2. historic_financials (summary format - legacy)
    """
    df = pd.DataFrame()
    
    # Try the detailed table first
    try:
        if hasattr(db, 'client'):
            response = db.client.table('historical_income_statement').select('*').eq(
                'scenario_id', scenario_id
            ).order('period_year').order('period_month').execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
    except Exception as e:
        pass  # Table may not exist
    
    # If no data, try the legacy historic_financials table
    if df.empty:
        try:
            if hasattr(db, 'client'):
                response = db.client.table('historic_financials').select('*').eq(
                    'scenario_id', scenario_id
                ).order('month').execute()
                
                if response.data:
                    df = pd.DataFrame(response.data)
                    # Map legacy columns to expected format
                    if 'month' in df.columns:
                        df['month'] = pd.to_datetime(df['month'])
                        df['period_year'] = df['month'].dt.year
                        df['period_month'] = df['month'].dt.month
                        df['period_date'] = df['month']
                        df['period_label'] = df['month'].dt.strftime('%b %Y')
                    
                    # Map summary columns to detailed format (use totals)
                    df['total_revenue'] = df.get('revenue', 0)
                    df['total_cogs'] = df.get('cogs', 0)
                    df['total_gross_profit'] = df.get('gross_profit', 0)
                    df['total_opex'] = df.get('opex', 0)
                    df['ebit'] = df.get('ebit', 0)
                    df['ebitda'] = df.get('ebit', 0)  # Approximate
                    
                    # Set revenue breakdown (allocate to existing wear as placeholder)
                    df['rev_wear_existing'] = df['total_revenue']
                    df['rev_service_existing'] = 0
                    df['rev_wear_prospect'] = 0
                    df['rev_service_prospect'] = 0
                    df['revenue_existing'] = df['total_revenue']
                    df['revenue_prospect'] = 0
                    
                    # Set COGS breakdown
                    df['cogs_wear_existing'] = df['total_cogs']
                    df['cogs_service_existing'] = 0
                    df['cogs_wear_prospect'] = 0
                    df['cogs_service_prospect'] = 0
                    
                    # Set OPEX breakdown (estimate from total)
                    total_opex = df['total_opex']
                    df['opex_personnel'] = total_opex * 0.45  # 45% personnel
                    df['opex_facilities'] = total_opex * 0.20  # 20% facilities
                    df['opex_admin'] = total_opex * 0.15  # 15% admin
                    df['opex_sales'] = total_opex * 0.12  # 12% sales
                    df['opex_other'] = total_opex * 0.08  # 8% other
                    
                    # Other fields
                    df['depreciation'] = 0
                    df['interest_expense'] = 0
                    df['ebt'] = df['ebit']
                    df['tax_expense'] = 0
                    df['net_income'] = df['ebit']
        except Exception as e:
            pass  # Table may not exist
    
    # Ensure required columns exist
    if not df.empty:
        if 'period_label' not in df.columns or df['period_label'].isna().any():
            df['period_label'] = df.apply(
                lambda r: datetime(int(r['period_year']), int(r['period_month']), 1).strftime('%b %Y'),
                axis=1
            )
        if 'period_date' not in df.columns:
            df['period_date'] = df.apply(
                lambda r: datetime(int(r['period_year']), int(r['period_month']), 1),
                axis=1
            )
    
    return df


def load_historical_balance_sheet(db, scenario_id: str) -> pd.DataFrame:
    """Load historical balance sheet data from database."""
    try:
        if hasattr(db, 'client'):
            response = db.client.table('historical_balance_sheet').select('*').eq(
                'scenario_id', scenario_id
            ).order('period_year').order('period_month').execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                # Ensure period_label exists
                if 'period_label' not in df.columns or df['period_label'].isna().any():
                    df['period_label'] = df.apply(
                        lambda r: datetime(int(r['period_year']), int(r['period_month']), 1).strftime('%b %Y'),
                        axis=1
                    )
                # Ensure period_date exists
                if 'period_date' not in df.columns:
                    df['period_date'] = df.apply(
                        lambda r: datetime(int(r['period_year']), int(r['period_month']), 1),
                        axis=1
                    )
                return df
    except Exception as e:
        pass
    
    return pd.DataFrame()


def load_historical_cashflow(db, scenario_id: str) -> pd.DataFrame:
    """Load historical cash flow data from database."""
    try:
        if hasattr(db, 'client'):
            response = db.client.table('historical_cashflow').select('*').eq(
                'scenario_id', scenario_id
            ).order('month').execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                # Map month to period columns
                if 'month' in df.columns:
                    df['month'] = pd.to_datetime(df['month'])
                    df['period_year'] = df['month'].dt.year
                    df['period_month'] = df['month'].dt.month
                    df['period_date'] = df['month']
                    df['period_label'] = df['month'].dt.strftime('%b %Y')
                
                # Ensure all expected columns exist
                expected_cols = [
                    'net_income', 'depreciation_amortization', 'change_in_receivables',
                    'change_in_inventory', 'change_in_payables', 'change_in_accruals',
                    'cash_from_operations', 'capital_expenditure', 'asset_disposals',
                    'cash_from_investing', 'debt_proceeds', 'debt_repayment',
                    'dividends_paid', 'cash_from_financing', 'net_change_in_cash'
                ]
                for col in expected_cols:
                    if col not in df.columns:
                        df[col] = 0
                
                return df
    except Exception as e:
        pass
    
    return pd.DataFrame()


def load_financial_periods(db, scenario_id: str) -> Dict:
    """Load financial period configuration."""
    try:
        if hasattr(db, 'client'):
            response = db.client.table('financial_periods').select('*').eq(
                'scenario_id', scenario_id
            ).single().execute()
            
            if response.data:
                return response.data
    except Exception as e:
        pass
    
    # Return defaults
    return {
        'fiscal_year_end_month': 12,
        'actuals_through_date': None,
        'current_fiscal_year': datetime.now().year
    }


# =============================================================================
# DATA PREPARATION
# =============================================================================

def build_financial_data(
    forecast_result: Dict[str, Any],
    assumptions: Dict[str, Any],
    db = None,
    scenario_id: str = None,
    start_date: datetime = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build complete financial data combining historicals and forecast.
    
    Returns:
        Tuple of (income_statement_df, balance_sheet_df, cash_flow_df)
    """
    # Load historical data if available
    hist_is = pd.DataFrame()
    hist_bs = pd.DataFrame()
    hist_cf = pd.DataFrame()
    period_config = {'actuals_through_date': None}
    
    if db and scenario_id:
        hist_is = load_historical_income_statement(db, scenario_id)
        hist_bs = load_historical_balance_sheet(db, scenario_id)
        hist_cf = load_historical_cashflow(db, scenario_id)
        period_config = load_financial_periods(db, scenario_id)
    
    # Determine actuals cutoff
    actuals_through = period_config.get('actuals_through_date')
    if actuals_through and isinstance(actuals_through, str):
        actuals_through = datetime.strptime(actuals_through, '%Y-%m-%d').date()
    
    # Build forecast data
    forecast_is = build_forecast_income_statement(forecast_result, assumptions, start_date)
    
    # Merge historical + forecast
    if not hist_is.empty:
        # Mark historical as actual
        hist_is['is_actual'] = True
        hist_is['data_type'] = 'Actual'
        
        # Mark forecast as forecast
        forecast_is['is_actual'] = False
        forecast_is['data_type'] = 'Forecast'
        
        # Combine, removing any forecast months that have actuals
        hist_periods = set(zip(hist_is['period_year'], hist_is['period_month']))
        forecast_is = forecast_is[~forecast_is.apply(
            lambda x: (x['period_year'], x['period_month']) in hist_periods, axis=1
        )]
        
        income_statement = pd.concat([hist_is, forecast_is], ignore_index=True)
        income_statement = income_statement.sort_values(['period_year', 'period_month']).reset_index(drop=True)
    else:
        # No historical data - use forecast only but check against actuals_through date
        forecast_is['is_actual'] = False
        forecast_is['data_type'] = 'Forecast'
        
        if actuals_through:
            # Mark periods before cutoff as needing actuals
            forecast_is['data_type'] = forecast_is.apply(
                lambda x: 'Actual (TBC)' if date(int(x['period_year']), int(x['period_month']), 1) <= actuals_through else 'Forecast',
                axis=1
            )
        
        income_statement = forecast_is
    
    # Extract manufacturing strategy data from forecast_result (if available)
    # This includes manufacturing PPE and working capital for balance sheet and cash flow
    manufacturing_strategy = forecast_result.get('manufacturing_strategy')
    if manufacturing_strategy:
        # Extract manufacturing PPE components
        mfg_equipment = manufacturing_strategy.get('equipment_cost', 0) or 0
        mfg_facility = manufacturing_strategy.get('facility_cost', 0) or 0
        mfg_tooling = manufacturing_strategy.get('tooling_cost', 0) or 0
        assumptions['manufacturing_ppe'] = mfg_equipment + mfg_facility + mfg_tooling
        assumptions['manufacturing_accum_depr'] = 0  # Start with zero accumulated depreciation
        
        # Extract manufacturing working capital
        assumptions['manufacturing_working_capital'] = manufacturing_strategy.get('working_capital', 0) or 0
    
    # Build Balance Sheet projections
    balance_sheet = build_balance_sheet_projections(income_statement, assumptions, hist_bs)
    
    # Build Cash Flow projections  
    forecast_cf = build_cash_flow_projections(income_statement, balance_sheet, assumptions, forecast_result)
    
    # Merge historical cash flow if available
    if not hist_cf.empty:
        # Mark historical as actual
        hist_cf['is_actual'] = True
        hist_cf['data_type'] = 'Actual'
        
        # Mark forecast as forecast
        forecast_cf['is_actual'] = False
        forecast_cf['data_type'] = 'Forecast'
        
        # Combine, removing any forecast months that have actuals
        hist_periods = set(zip(hist_cf['period_year'], hist_cf['period_month']))
        forecast_cf = forecast_cf[~forecast_cf.apply(
            lambda x: (x['period_year'], x['period_month']) in hist_periods, axis=1
        )]
        
        cash_flow = pd.concat([hist_cf, forecast_cf], ignore_index=True)
        cash_flow = cash_flow.sort_values(['period_year', 'period_month']).reset_index(drop=True)
    else:
        cash_flow = forecast_cf
    
    return income_statement, balance_sheet, cash_flow


def build_forecast_income_statement(
    forecast_result: Dict[str, Any],
    assumptions: Dict[str, Any],
    start_date: datetime = None
) -> pd.DataFrame:
    """Build forecast income statement from forecast engine results.
    
    Handles two formats:
    1. Original format with revenue_grid DataFrame
    2. forecast_section.py format with revenue/costs/profit dicts of lists
    """
    
    summary = forecast_result.get('summary', {})
    revenue_grid = forecast_result.get('revenue_grid')
    
    # Check for forecast_section.py format (has 'revenue' dict with 'total' list)
    revenue_dict = forecast_result.get('revenue', {})
    timeline = forecast_result.get('timeline', [])
    
    if start_date is None:
        start_date = datetime.now().replace(day=1)
    
    # Get margins
    gross_margin_wear = assumptions.get('gross_margin_liner', assumptions.get('gross_margin_wear', 0.38))
    gross_margin_service = assumptions.get('gross_margin_refurb', assumptions.get('gross_margin_service', 0.32))
    avg_gross_margin = (gross_margin_wear + gross_margin_service) / 2
    opex_pct = assumptions.get('opex_pct_revenue', 0.15)
    tax_rate = assumptions.get('tax_rate', 0.28)
    depreciation_pct = assumptions.get('depreciation_pct_revenue', 0.03)
    interest_annual = assumptions.get('interest_expense', 0)
    
    months_data = []
    
    # Handle forecast_section.py format
    if timeline and isinstance(revenue_dict.get('total'), list) and len(revenue_dict['total']) > 0:
        n_months = len(timeline)
        consumables = revenue_dict.get('consumables', [0] * n_months)
        refurb = revenue_dict.get('refurb', [0] * n_months)
        pipeline = revenue_dict.get('pipeline', [0] * n_months)
        total_rev = revenue_dict.get('total', [0] * n_months)
        
        costs_dict = forecast_result.get('costs', {})
        cogs_list = costs_dict.get('cogs', [0] * n_months)
        opex_list = costs_dict.get('opex', [0] * n_months)
        
        profit_dict = forecast_result.get('profit', {})
        gross_profit_list = profit_dict.get('gross', [0] * n_months)
        ebit_list = profit_dict.get('ebit', [0] * n_months)
        
        for i in range(n_months):
            # Parse timeline 'YYYY-MM' format
            try:
                year_month = timeline[i]
                year = int(year_month[:4])
                month = int(year_month[5:7])
                month_date = datetime(year, month, 1)
            except:
                month_date = start_date + relativedelta(months=i)
            
            # Revenue breakdown - existing is consumables + refurb, prospect is pipeline
            rev_wear_existing = consumables[i] if i < len(consumables) else 0
            rev_service_existing = refurb[i] if i < len(refurb) else 0
            rev_wear_prospect = (pipeline[i] * 0.7) if i < len(pipeline) else 0
            rev_service_prospect = (pipeline[i] * 0.3) if i < len(pipeline) else 0
            
            revenue_existing = rev_wear_existing + rev_service_existing
            revenue_prospect = rev_wear_prospect + rev_service_prospect
            total_revenue = total_rev[i] if i < len(total_rev) else 0
            
            # Use provided COGS and OPEX if available
            total_cogs = cogs_list[i] if i < len(cogs_list) else total_revenue * (1 - avg_gross_margin)
            total_gross_profit = gross_profit_list[i] if i < len(gross_profit_list) else total_revenue - total_cogs
            total_opex = opex_list[i] if i < len(opex_list) else total_revenue * opex_pct
            ebit = ebit_list[i] if i < len(ebit_list) else total_gross_profit - total_opex
            
            # Manufacturing COGS breakdown (NEW: Bought vs Manufactured split)
            costs_dict = forecast_result.get('costs', {})
            cogs_buy = costs_dict.get('cogs_buy', [0] * n_months)
            cogs_make = costs_dict.get('cogs_make', [0] * n_months)
            mfg_variable_overhead = costs_dict.get('mfg_variable_overhead', [0] * n_months)
            mfg_fixed_overhead = costs_dict.get('mfg_fixed_overhead', [0] * n_months)
            mfg_overhead = costs_dict.get('mfg_overhead', [0] * n_months)
            mfg_depreciation = costs_dict.get('mfg_depreciation', [0] * n_months)
            
            cogs_buy_val = cogs_buy[i] if i < len(cogs_buy) else total_cogs
            cogs_make_val = cogs_make[i] if i < len(cogs_make) else 0
            mfg_var_overhead_val = mfg_variable_overhead[i] if i < len(mfg_variable_overhead) else 0
            mfg_fixed_overhead_val = mfg_fixed_overhead[i] if i < len(mfg_fixed_overhead) else 0
            mfg_overhead_val = mfg_overhead[i] if i < len(mfg_overhead) else 0
            mfg_depreciation_val = mfg_depreciation[i] if i < len(mfg_depreciation) else 0
            
            # Calculate revenue split (bought vs manufactured) based on COGS ratio
            # This represents which portion of revenue corresponds to bought vs manufactured products
            if total_cogs > 0:
                buy_pct = cogs_buy_val / total_cogs
                make_pct = cogs_make_val / total_cogs if cogs_make_val > 0 else 0
            else:
                buy_pct = 1.0
                make_pct = 0.0
            
            revenue_bought = total_revenue * buy_pct
            revenue_manufactured = total_revenue * make_pct
            
            # Derive other values
            depreciation = total_revenue * depreciation_pct
            ebitda = ebit + depreciation
            interest = interest_annual / 12
            ebt = ebit - interest
            tax = max(ebt * tax_rate, 0)
            net_income = ebt - tax
            
            # Split COGS proportionally
            cogs_wear_existing = total_cogs * (rev_wear_existing / total_revenue) if total_revenue > 0 else 0
            cogs_service_existing = total_cogs * (rev_service_existing / total_revenue) if total_revenue > 0 else 0
            cogs_wear_prospect = total_cogs * (rev_wear_prospect / total_revenue) if total_revenue > 0 else 0
            cogs_service_prospect = total_cogs * (rev_service_prospect / total_revenue) if total_revenue > 0 else 0
            
            # Split OPEX (simplified)
            opex_personnel = total_opex * 0.45
            opex_facilities = total_opex * 0.20
            opex_admin = total_opex * 0.15
            opex_sales = total_opex * 0.12
            opex_other = total_opex * 0.08
            
            months_data.append({
                'period_year': month_date.year,
                'period_month': month_date.month,
                'period_date': month_date,
                'period_label': month_date.strftime('%b %Y'),
                'is_actual': False,
                'data_type': 'Forecast',
                
                # Revenue - with bought/manufactured split (NEW)
                'rev_wear_existing': rev_wear_existing,
                'rev_service_existing': rev_service_existing,
                'rev_wear_prospect': rev_wear_prospect,
                'rev_service_prospect': rev_service_prospect,
                'revenue_existing': revenue_existing,
                'revenue_prospect': revenue_prospect,
                'revenue_bought': revenue_bought,  # NEW: Revenue from bought products
                'revenue_manufactured': revenue_manufactured,  # NEW: Revenue from manufactured products
                'total_revenue': total_revenue,
                
                # COGS - with manufacturing breakdown (NEW)
                'cogs_wear_existing': cogs_wear_existing,
                'cogs_service_existing': cogs_service_existing,
                'cogs_wear_prospect': cogs_wear_prospect,
                'cogs_service_prospect': cogs_service_prospect,
                'cogs_buy': cogs_buy_val,  # NEW: COGS for purchased products
                'cogs_make': cogs_make_val,  # NEW: Direct COGS for manufactured products
                'mfg_variable_overhead': mfg_var_overhead_val,  # NEW: Manufacturing variable overhead
                'mfg_fixed_overhead': mfg_fixed_overhead_val,  # NEW: Manufacturing fixed overhead
                'mfg_overhead': mfg_overhead_val,  # Combined overhead (for backward compatibility)
                'mfg_depreciation': mfg_depreciation_val,  # NEW: Manufacturing depreciation
                'total_cogs': total_cogs,
                
                'total_gross_profit': total_gross_profit,
                
                # OpEx
                'opex_personnel': opex_personnel,
                'opex_facilities': opex_facilities,
                'opex_admin': opex_admin,
                'opex_sales': opex_sales,
                'opex_other': opex_other,
                'total_opex': total_opex,
                
                # P&L
                'ebitda': ebitda,
                'depreciation': depreciation,
                'ebit': ebit,
                'interest_expense': interest,
                'ebt': ebt,
                'tax_expense': tax,
                'net_income': net_income,
            })
        
        return pd.DataFrame(months_data)
    
    # Handle original revenue_grid format
    if revenue_grid is None or (isinstance(revenue_grid, pd.DataFrame) and revenue_grid.empty):
        # No valid data - return empty
        return pd.DataFrame()
    
    if isinstance(revenue_grid, dict):
        revenue_grid = pd.DataFrame(revenue_grid)
    
    n_months = len(revenue_grid) if hasattr(revenue_grid, '__len__') else summary.get('forecast_months', 60)
    
    for i in range(n_months):
        month_date = start_date + relativedelta(months=i)
        
        # Get revenue data
        if isinstance(revenue_grid, pd.DataFrame) and i < len(revenue_grid):
            row = revenue_grid.iloc[i]
            
            rev_wear_existing = (row.get('fleet_liner_revenue') or row.get('liner_revenue') or 
                                row.get('wear_parts_existing') or row.get('existing_liner', 0)) or 0
            rev_service_existing = (row.get('fleet_refurb_revenue') or row.get('refurb_revenue') or 
                                   row.get('service_existing') or row.get('existing_refurb', 0)) or 0
            rev_wear_prospect = (row.get('prospect_liner_revenue') or row.get('wear_parts_prospect') or 
                                row.get('prospect_liner', 0)) or 0
            rev_service_prospect = (row.get('prospect_refurb_revenue') or row.get('service_prospect') or 
                                   row.get('prospect_refurb', 0)) or 0
            
            if rev_wear_existing == 0 and rev_service_existing == 0:
                total_existing = row.get('fleet_total', row.get('existing_total', 0)) or 0
                total_prospect = row.get('prospect_total', 0) or 0
                rev_wear_existing = total_existing * 0.7
                rev_service_existing = total_existing * 0.3
                rev_wear_prospect = total_prospect * 0.7
                rev_service_prospect = total_prospect * 0.3
        else:
            base = 500000 * (1 + i * 0.02)
            rev_wear_existing = base * 0.5
            rev_service_existing = base * 0.2
            rev_wear_prospect = base * 0.2
            rev_service_prospect = base * 0.1
        
        # Calculate financials
        revenue_existing = rev_wear_existing + rev_service_existing
        revenue_prospect = rev_wear_prospect + rev_service_prospect
        total_revenue = revenue_existing + revenue_prospect
        
        # COGS
        cogs_wear_existing = rev_wear_existing * (1 - gross_margin_wear)
        cogs_service_existing = rev_service_existing * (1 - gross_margin_service)
        cogs_wear_prospect = rev_wear_prospect * (1 - gross_margin_wear)
        cogs_service_prospect = rev_service_prospect * (1 - gross_margin_service)
        total_cogs = cogs_wear_existing + cogs_service_existing + cogs_wear_prospect + cogs_service_prospect
        
        total_gross_profit = total_revenue - total_cogs
        
        # OpEx
        total_opex = total_revenue * opex_pct
        opex_personnel = total_opex * 0.50
        opex_facilities = total_opex * 0.20
        opex_admin = total_opex * 0.12
        opex_sales = total_opex * 0.10
        opex_other = total_opex * 0.08
        
        # P&L
        ebitda = total_gross_profit - total_opex
        depreciation = total_revenue * depreciation_pct
        ebit = ebitda - depreciation
        interest = interest_annual / 12
        ebt = ebit - interest
        tax = max(ebt * tax_rate, 0)
        net_income = ebt - tax
        
        months_data.append({
            'period_year': month_date.year,
            'period_month': month_date.month,
            'period_date': month_date,
            'period_label': month_date.strftime('%b %Y'),
            'is_actual': False,
            'data_type': 'Forecast',
            
            # Revenue
            'rev_wear_existing': rev_wear_existing,
            'rev_service_existing': rev_service_existing,
            'rev_wear_prospect': rev_wear_prospect,
            'rev_service_prospect': rev_service_prospect,
            'revenue_existing': revenue_existing,
            'revenue_prospect': revenue_prospect,
            'total_revenue': total_revenue,
            
            # COGS
            'cogs_wear_existing': cogs_wear_existing,
            'cogs_service_existing': cogs_service_existing,
            'cogs_wear_prospect': cogs_wear_prospect,
            'cogs_service_prospect': cogs_service_prospect,
            'total_cogs': total_cogs,
            
            'total_gross_profit': total_gross_profit,
            
            # OpEx
            'opex_personnel': opex_personnel,
            'opex_facilities': opex_facilities,
            'opex_admin': opex_admin,
            'opex_sales': opex_sales,
            'opex_other': opex_other,
            'total_opex': total_opex,
            
            # P&L
            'ebitda': ebitda,
            'depreciation': depreciation,
            'ebit': ebit,
            'interest_expense': interest,
            'ebt': ebt,
            'tax_expense': tax,
            'net_income': net_income,
        })
    
    return pd.DataFrame(months_data)


def build_balance_sheet_projections(
    income_statement: pd.DataFrame,
    assumptions: Dict[str, Any],
    hist_bs: pd.DataFrame = None,
    debt_schedule: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Build balance sheet projections from income statement.
    
    For historical periods (is_actual=True), uses actual historical balance sheet data if available.
    For forecast periods, generates projections based on the income statement.
    """
    
    if income_statement.empty:
        return pd.DataFrame()
    
    # Get working capital assumptions
    ar_days = assumptions.get('ar_days', 45)
    inventory_days = assumptions.get('inventory_days', 60)
    ap_days = assumptions.get('ap_days', 30)
    capex_pct = assumptions.get('capex_pct_revenue', 0.05)
    
    # NEW: Get debt schedule if available
    debt_principal = 0
    debt_interest_payable = 0
    if debt_schedule:
        # Extract current period debt balances
        debt_principal = debt_schedule.get('outstanding_principal', 0) or 0
        debt_interest_payable = debt_schedule.get('accrued_interest', 0) or 0
    else:
        # Fallback: try to get from assumptions
        debt_principal = assumptions.get('initial_debt', 0) or assumptions.get('net_debt', 0) or 0
    
    # Create a lookup for historical balance sheet by period
    hist_bs_lookup = {}
    if hist_bs is not None and not hist_bs.empty:
        for _, row in hist_bs.iterrows():
            key = (int(row.get('period_year', 0)), int(row.get('period_month', 0)))
            hist_bs_lookup[key] = row
    
    # Get starting balance sheet values from last historical or assumptions
    if hist_bs is not None and not hist_bs.empty:
        last_bs = hist_bs.iloc[-1]
        starting_cash = last_bs.get('cash_and_equivalents', 0) or 0
        starting_ppe = last_bs.get('property_plant_equipment', 0) or 0
        starting_accum_depr = last_bs.get('accumulated_depreciation', 0) or 0
        starting_debt = (last_bs.get('long_term_debt', 0) or 0) + (last_bs.get('short_term_debt', 0) or 0)
        starting_equity = last_bs.get('share_capital', 0) or 0
        starting_retained = last_bs.get('retained_earnings', 0) or 0
    else:
        # Use assumptions for starting values
        starting_cash = assumptions.get('starting_cash', 5000000)
        starting_ppe = assumptions.get('starting_ppe', 10000000)
        starting_accum_depr = assumptions.get('starting_accum_depr', 2000000)
        starting_debt = assumptions.get('net_debt', 0)
        starting_equity = assumptions.get('share_capital', 1000000)
        starting_retained = assumptions.get('retained_earnings', 5000000)
    
    bs_data = []
    cumulative_net_income = starting_retained
    cumulative_depreciation = starting_accum_depr
    cumulative_ppe = starting_ppe
    prev_cash = starting_cash
    
    # NEW: Track manufacturing PPE separately
    mfg_ppe_base = assumptions.get('manufacturing_ppe', 0) or 0
    mfg_ppe_accum_depr = assumptions.get('manufacturing_accum_depr', 0) or 0
    other_ppe = starting_ppe - mfg_ppe_base
    other_accum_depr = starting_accum_depr - mfg_ppe_accum_depr
    
    for i, row in income_statement.iterrows():
        period_year = int(row['period_year'])
        period_month = int(row['period_month'])
        is_actual = row.get('is_actual', False)
        
        # Check if we have historical balance sheet data for this period
        hist_row = hist_bs_lookup.get((period_year, period_month))
        
        if is_actual and hist_row is not None:
            # Use actual historical balance sheet data
            bs_data.append({
                'period_year': period_year,
                'period_month': period_month,
                'period_date': row.get('period_date'),
                'period_label': row.get('period_label'),
                'is_actual': True,
                'data_type': 'Actual',
                
                # Current Assets
                'cash_and_equivalents': hist_row.get('cash_and_equivalents', 0) or 0,
                'accounts_receivable': hist_row.get('accounts_receivable', 0) or 0,
                'inventory': hist_row.get('inventory', 0) or 0,
                'total_current_assets': hist_row.get('total_current_assets', 0) or 0,
                
                # Non-Current Assets
                'property_plant_equipment': hist_row.get('property_plant_equipment', 0) or 0,
                'accumulated_depreciation': hist_row.get('accumulated_depreciation', 0) or 0,
                'net_ppe': hist_row.get('net_ppe', 0) or 0,
                'total_non_current_assets': hist_row.get('total_non_current_assets', 0) or 0,
                
                'total_assets': hist_row.get('total_assets', 0) or 0,
                
                # Current Liabilities
                'accounts_payable': hist_row.get('accounts_payable', 0) or 0,
                'short_term_debt': hist_row.get('short_term_debt', 0) or 0,
                'total_current_liabilities': hist_row.get('total_current_liabilities', 0) or 0,
                
                # Non-Current Liabilities
                'long_term_debt': hist_row.get('long_term_debt', 0) or 0,
                'total_non_current_liabilities': hist_row.get('total_non_current_liabilities', 0) or 0,
                
                'total_liabilities': hist_row.get('total_liabilities', 0) or 0,
                
                # Equity
                'share_capital': hist_row.get('share_capital', 0) or 0,
                'retained_earnings': hist_row.get('retained_earnings', 0) or 0,
                'total_equity': hist_row.get('total_equity', 0) or 0,
                
                'total_liabilities_and_equity': hist_row.get('total_liabilities_and_equity', 0) or hist_row.get('total_assets', 0) or 0,
                
                # Working Capital
                'net_working_capital': hist_row.get('net_working_capital', 0) or 0,
            })
            
            # Update running totals from historical data for forecasting
            prev_cash = hist_row.get('cash_and_equivalents', prev_cash) or prev_cash
            cumulative_ppe = hist_row.get('property_plant_equipment', cumulative_ppe) or cumulative_ppe
            cumulative_depreciation = hist_row.get('accumulated_depreciation', cumulative_depreciation) or cumulative_depreciation
            cumulative_net_income = hist_row.get('retained_earnings', cumulative_net_income) or cumulative_net_income
            starting_debt = (hist_row.get('long_term_debt', 0) or 0) + (hist_row.get('short_term_debt', 0) or 0)
            starting_equity = hist_row.get('share_capital', starting_equity) or starting_equity
        else:
            # Generate forecast balance sheet based on income statement
            monthly_revenue = row.get('total_revenue', 0) or 0
            monthly_cogs = row.get('total_cogs', 0) or 0
            
            # NEW: Get manufacturing COGS splits for inventory calculation
            cogs_buy = row.get('cogs_buy', monthly_cogs) or monthly_cogs
            cogs_make = row.get('cogs_make', 0) or 0
            
            # Ensure consistency: use split COGS total (accounts for any rounding differences)
            # This ensures inventory and payables are based on the same COGS basis
            total_cogs_split = cogs_buy + cogs_make
            
            accounts_receivable = monthly_revenue * (ar_days / 30)
            
            # NEW: Split inventory - Raw Materials (for manufacturing) vs Finished Goods (trading)
            # Raw materials inventory = manufacturing COGS * (raw material days / 30)
            # Finished goods inventory = buy COGS * (inventory days / 30)
            raw_material_days = assumptions.get('raw_material_days', 30)  # Days of raw material inventory
            inventory_raw_materials = cogs_make * (raw_material_days / 30) if cogs_make > 0 else 0
            inventory_finished_goods = cogs_buy * (inventory_days / 30)
            inventory_total = inventory_raw_materials + inventory_finished_goods
            
            # FIX: Use consistent COGS split for accounts payable to match inventory calculation
            # Accounts payable includes payables for both bought products and raw materials
            accounts_payable = total_cogs_split * (ap_days / 30)
            
            # PPE and Depreciation - NEW: Split manufacturing vs other PPE
            monthly_capex = monthly_revenue * capex_pct
            monthly_depr = row.get('depreciation', 0) or 0
            mfg_depreciation = row.get('mfg_depreciation', 0) or 0
            
            # Track manufacturing PPE separately
            # Add manufacturing depreciation to accumulated
            mfg_ppe_accum_depr += mfg_depreciation
            
            # Other PPE (non-manufacturing)
            other_ppe += monthly_capex  # Add regular capex to other PPE
            other_accum_depr += (monthly_depr - mfg_depreciation)
            
            cumulative_ppe = other_ppe + mfg_ppe_base
            cumulative_depreciation = other_accum_depr + mfg_ppe_accum_depr
            net_ppe = cumulative_ppe - cumulative_depreciation
            net_ppe_other = other_ppe - other_accum_depr
            net_ppe_manufacturing = mfg_ppe_base - mfg_ppe_accum_depr
            
            # Retained earnings
            cumulative_net_income += row.get('net_income', 0) or 0
            
            # Calculate total assets/liabilities
            total_current_assets = prev_cash + accounts_receivable + inventory_total
            total_non_current_assets = net_ppe
            total_assets = total_current_assets + total_non_current_assets
            
            total_current_liabilities = accounts_payable
            total_non_current_liabilities = starting_debt
            total_liabilities = total_current_liabilities + total_non_current_liabilities
            
            total_equity = starting_equity + cumulative_net_income
            
            # Plug cash to balance
            implied_cash = total_liabilities + total_equity - (accounts_receivable + inventory_total + net_ppe)
            prev_cash = max(implied_cash, 0)
            
            bs_data.append({
                'period_year': period_year,
                'period_month': period_month,
                'period_date': row.get('period_date'),
                'period_label': row.get('period_label'),
                'is_actual': False,
                'data_type': 'Forecast',
                
                # Current Assets - NEW: Split inventory
                'cash_and_equivalents': prev_cash,
                'accounts_receivable': accounts_receivable,
                'inventory_raw_materials': inventory_raw_materials,  # NEW: Manufacturing raw materials
                'inventory_finished_goods': inventory_finished_goods,  # NEW: Trading finished goods
                'inventory': inventory_total,  # Total inventory (for backward compatibility)
                'total_current_assets': prev_cash + accounts_receivable + inventory_total,
                
                # Non-Current Assets - NEW: Split PPE
                'ppe_other': other_ppe,  # NEW: Non-manufacturing PPE
                'ppe_manufacturing': mfg_ppe_base,  # NEW: Manufacturing PPE
                'property_plant_equipment': cumulative_ppe,  # Total PPE
                'accumulated_depreciation_other': other_accum_depr,  # NEW: Non-manufacturing depreciation
                'accumulated_depreciation_manufacturing': mfg_ppe_accum_depr,  # NEW: Manufacturing depreciation
                'accumulated_depreciation': cumulative_depreciation,  # Total depreciation
                'net_ppe_other': net_ppe_other,  # NEW: Net non-manufacturing PPE
                'net_ppe_manufacturing': net_ppe_manufacturing,  # NEW: Net manufacturing PPE
                'net_ppe': net_ppe,  # Total net PPE
                'total_non_current_assets': net_ppe,
                
                'total_assets': prev_cash + accounts_receivable + inventory_total + net_ppe,
                
                # Current Liabilities
                'accounts_payable': accounts_payable,
                'short_term_debt': 0,
                'total_current_liabilities': accounts_payable,
                
                # Non-Current Liabilities - NEW: Split debt principal and interest
                'debt_principal': debt_principal,  # NEW: Principal amount
                'debt_interest_payable': debt_interest_payable,  # NEW: Accrued interest
                'long_term_debt': debt_principal,  # Principal only
                'total_non_current_liabilities': debt_principal,
                
                'total_liabilities': accounts_payable + starting_debt,
                
                # Equity
                'share_capital': starting_equity,
                'retained_earnings': cumulative_net_income,
                'total_equity': starting_equity + cumulative_net_income,
                
                'total_liabilities_and_equity': accounts_payable + starting_debt + starting_equity + cumulative_net_income,
                
                # Working Capital
                'net_working_capital': (prev_cash + accounts_receivable + inventory_total) - accounts_payable,
            })
    
    return pd.DataFrame(bs_data)


def build_cash_flow_projections(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    assumptions: Dict[str, Any],
    forecast_result: Dict[str, Any] = None
) -> pd.DataFrame:
    """Build cash flow statement from income statement and balance sheet changes.
    
    NEW: Extracts manufacturing commissioning costs from forecast_result if available.
    """
    
    if income_statement.empty or balance_sheet.empty:
        return pd.DataFrame()
    
    # Get manufacturing commissioning costs from forecast_result if available
    commissioning_costs = []
    if forecast_result:
        costs_dict = forecast_result.get('costs', {})
        commissioning_costs = costs_dict.get('commissioning', [])
    
    cf_data = []
    prev_bs = None
    
    for i in range(len(income_statement)):
        is_row = income_statement.iloc[i]
        bs_row = balance_sheet.iloc[i] if i < len(balance_sheet) else None
        
        # Operating Activities
        net_income = is_row.get('net_income', 0)
        depreciation = is_row.get('depreciation', 0)
        
        # Working capital changes - NEW: Detailed breakdown
        if prev_bs is not None and bs_row is not None:
            change_ar = -(bs_row.get('accounts_receivable', 0) - prev_bs.get('accounts_receivable', 0))
            
            # NEW: Split inventory changes
            change_inv_raw_materials = -(bs_row.get('inventory_raw_materials', 0) - prev_bs.get('inventory_raw_materials', 0))
            change_inv_finished_goods = -(bs_row.get('inventory_finished_goods', 0) - prev_bs.get('inventory_finished_goods', 0))
            change_inv = change_inv_raw_materials + change_inv_finished_goods
            
            change_ap = bs_row.get('accounts_payable', 0) - prev_bs.get('accounts_payable', 0)
            
            # NEW: Manufacturing working capital change (if applicable)
            change_mfg_wc = 0
            if 'inventory_raw_materials' in bs_row and 'inventory_raw_materials' in prev_bs:
                change_mfg_wc = change_inv_raw_materials
        else:
            change_ar = 0
            change_inv_raw_materials = 0
            change_inv_finished_goods = 0
            change_inv = 0
            change_ap = 0
            change_mfg_wc = 0
        
        cash_from_operations = net_income + depreciation + change_ar + change_inv + change_ap
        
        # Investing Activities - NEW: Split manufacturing vs other investments
        capex_pct = assumptions.get('capex_pct_revenue', 0.05)
        capex_other = -is_row.get('total_revenue', 0) * capex_pct
        
        # Get manufacturing commissioning costs and PPE investments
        commissioning_cost = commissioning_costs[i] if i < len(commissioning_costs) else 0
        mfg_ppe_investment = 0
        if i == 0:  # Manufacturing PPE is added once at start
            mfg_ppe_investment = -(assumptions.get('manufacturing_ppe', 0) or 0)
        
        # Manufacturing working capital (one-time at start)
        mfg_wc_investment = 0
        if i == 0:
            mfg_wc_investment = -(assumptions.get('manufacturing_working_capital', 0) or 0)
        
        capex_manufacturing = commissioning_cost + mfg_ppe_investment
        total_capex = capex_other + capex_manufacturing
        cash_from_investing = total_capex
        
        # Financing Activities
        cash_from_financing = 0  # Simplified - no debt changes
        
        # Net Change
        net_change = cash_from_operations + cash_from_investing + cash_from_financing
        
        beginning_cash = prev_bs.get('cash_and_equivalents', 0) if prev_bs is not None else assumptions.get('starting_cash', 5000000)
        ending_cash = beginning_cash + net_change
        
        cf_data.append({
            'period_year': is_row['period_year'],
            'period_month': is_row['period_month'],
            'period_date': is_row.get('period_date'),
            'period_label': is_row.get('period_label'),
            'is_actual': is_row.get('is_actual', False),
            'data_type': is_row.get('data_type', 'Forecast'),
            
            # Operating - NEW: Detailed working capital breakdown
            'net_income': net_income,
            'depreciation_amortization': depreciation,
            'change_in_receivables': change_ar,
            'change_in_inventory_raw_materials': change_inv_raw_materials,  # NEW: Manufacturing raw materials
            'change_in_inventory_finished_goods': change_inv_finished_goods,  # NEW: Trading finished goods
            'change_in_inventory': change_inv,  # Total (for backward compatibility)
            'change_in_payables': change_ap,
            'change_in_manufacturing_wc': change_mfg_wc,  # NEW: Manufacturing working capital change
            'cash_from_operations': cash_from_operations,
            
            # Investing - NEW: Split manufacturing vs other
            'capital_expenditures_other': capex_other,  # NEW: Non-manufacturing CAPEX
            'mfg_ppe_investment': mfg_ppe_investment,  # NEW: Manufacturing PPE investment
            'mfg_commissioning_costs': -commissioning_cost,  # NEW: Manufacturing commissioning (negative = outflow)
            'mfg_working_capital': mfg_wc_investment,  # NEW: Manufacturing working capital investment
            'capital_expenditures': total_capex,  # Total CAPEX (for backward compatibility)
            'cash_from_investing': cash_from_investing,
            
            # Financing
            'cash_from_financing': cash_from_financing,
            
            # Net
            'net_change_in_cash': net_change,
            'beginning_cash': beginning_cash,
            'ending_cash': ending_cash,
        })
        
        prev_bs = bs_row
    
    return pd.DataFrame(cf_data)


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_to_annual(df: pd.DataFrame, value_columns: List[str]) -> pd.DataFrame:
    """Aggregate monthly data to annual with proper handling of mixed years."""
    if df.empty:
        return df
    
    annual_data = []
    
    for year in sorted(df['period_year'].unique()):
        year_data = df[df['period_year'] == year]
        
        # Check if mixed actual/forecast
        has_actuals = year_data['is_actual'].any() if 'is_actual' in year_data.columns else False
        has_forecast = (~year_data['is_actual']).any() if 'is_actual' in year_data.columns else True
        
        if has_actuals and has_forecast:
            data_type = 'Mixed (Actual + Forecast)'
            n_actual = year_data['is_actual'].sum()
            n_forecast = len(year_data) - n_actual
            period_label = f'{year}*'
            period_note = f'{n_actual}mo Actual + {n_forecast}mo Forecast'
        elif has_actuals:
            data_type = 'Actual'
            period_label = str(year)
            period_note = 'Full Year Actual'
        else:
            data_type = 'Forecast'
            period_label = str(year)
            period_note = 'Full Year Forecast'
        
        # Sum numeric columns
        row_data = {
            'period_year': year,
            'period_label': period_label,
            'period_note': period_note,
            'data_type': data_type,
            'is_actual': has_actuals and not has_forecast,
            'is_mixed': has_actuals and has_forecast,
            'is_annual_total': True,
        }
        
        for col in value_columns:
            if col in year_data.columns:
                row_data[col] = year_data[col].sum()
        
        annual_data.append(row_data)
    
    return pd.DataFrame(annual_data)


def add_annual_subtotals(df: pd.DataFrame, value_columns: List[str]) -> pd.DataFrame:
    """Add annual subtotal rows after each year's monthly data."""
    if df.empty:
        return df
    
    result = []
    
    for year in sorted(df['period_year'].unique()):
        year_data = df[df['period_year'] == year]
        
        # Add monthly rows
        for _, row in year_data.iterrows():
            row_dict = row.to_dict()
            row_dict['is_annual_total'] = False
            result.append(row_dict)
        
        # Add annual subtotal
        has_actuals = year_data['is_actual'].any() if 'is_actual' in year_data.columns else False
        has_forecast = (~year_data['is_actual']).any() if 'is_actual' in year_data.columns else True
        
        if has_actuals and has_forecast:
            n_actual = year_data['is_actual'].sum()
            n_forecast = len(year_data) - n_actual
            label = f'FY{year} Total ({n_actual}mo Act + {n_forecast}mo Fct)'
            data_type = 'Mixed'
        elif has_actuals:
            label = f'FY{year} Total (Actual)'
            data_type = 'Actual'
        else:
            label = f'FY{year} Total (Forecast)'
            data_type = 'Forecast'
        
        subtotal = {
            'period_year': year,
            'period_month': 13,
            'period_label': label,
            'data_type': data_type,
            'is_actual': False,
            'is_annual_total': True,
            'is_mixed': has_actuals and has_forecast,
        }
        
        for col in value_columns:
            if col in year_data.columns:
                subtotal[col] = year_data[col].sum()
        
        result.append(subtotal)
    
    return pd.DataFrame(result)


# =============================================================================
# TABLE RENDERING
# =============================================================================

def render_statement_table(
    data: pd.DataFrame,
    statement_type: str,  # 'income', 'balance', 'cashflow'
    show_detail: Dict[str, bool],
    view_mode: str
):
    """Render a financial statement as HTML table with horizontal scrolling."""
    
    if data.empty:
        st.warning(f"No data available for {statement_type} statement.")
        return
    
    periods = data['period_label'].tolist()
    
    # Show all periods - table will scroll horizontally
    display_data = data
    
    # Define line items based on statement type
    if statement_type == 'income':
        sections = get_income_statement_structure(show_detail, display_data)
    elif statement_type == 'balance':
        sections = get_balance_sheet_structure(show_detail, display_data)
    else:
        sections = get_cash_flow_structure(show_detail, display_data)
    
    # Build CSS and HTML
    html = build_table_css()
    html += build_table_header(periods, display_data)
    html += build_table_body(sections, periods, display_data)
    html += "</tbody></table></div>"
    
    st.markdown(html, unsafe_allow_html=True)


def get_income_statement_structure(show_detail: Dict[str, bool], display_data: pd.DataFrame = None) -> List[Dict]:
    """Define income statement line items structure.
    
    NEW: Shows bought vs manufactured split when manufacturing strategy is active.
    """
    sections = []
    
    # Check if manufacturing data is available
    has_mfg_data = False
    if display_data is not None and not display_data.empty:
        has_mfg_data = (
            'revenue_manufactured' in display_data.columns and 
            display_data['revenue_manufactured'].sum() > 0
        ) or (
            'cogs_make' in display_data.columns and 
            display_data['cogs_make'].sum() > 0
        )
    
    # Revenue
    sections.append({'type': 'section', 'label': '📈 REVENUE'})
    
    if show_detail.get('revenue', True):
        sections.append({'type': 'subtotal', 'label': 'Existing Customers', 'field': 'revenue_existing', 'indent': 1})
        sections.append({'type': 'detail', 'label': 'Wear Parts', 'field': 'rev_wear_existing', 'indent': 2})
        sections.append({'type': 'detail', 'label': 'Refurbishment & Service', 'field': 'rev_service_existing', 'indent': 2})
        sections.append({'type': 'divider'})
        sections.append({'type': 'subtotal', 'label': 'Prospective Customers', 'field': 'revenue_prospect', 'indent': 1})
        sections.append({'type': 'detail', 'label': 'Wear Parts', 'field': 'rev_wear_prospect', 'indent': 2})
        sections.append({'type': 'detail', 'label': 'Refurbishment & Service', 'field': 'rev_service_prospect', 'indent': 2})
        
        # NEW: Show bought vs manufactured revenue split if manufacturing is active
        if has_mfg_data:
            sections.append({'type': 'divider'})
            sections.append({'type': 'subtotal', 'label': 'Revenue - Purchased Products', 'field': 'revenue_bought', 'indent': 1})
            sections.append({'type': 'subtotal', 'label': 'Revenue - Manufactured Products', 'field': 'revenue_manufactured', 'indent': 1})
        
        sections.append({'type': 'divider'})
    
    sections.append({'type': 'total', 'label': 'TOTAL REVENUE', 'field': 'total_revenue'})
    
    # COGS - Enhanced with manufacturing breakdown
    sections.append({'type': 'section', 'label': '📦 COST OF GOODS SOLD'})
    
    # Check if we have manufacturing COGS breakdown
    has_mfg_cogs = False
    has_var_overhead = False
    if display_data is not None and not display_data.empty:
        has_mfg_cogs = 'cogs_make' in display_data.columns and display_data['cogs_make'].sum() != 0
        has_var_overhead = 'mfg_variable_overhead' in display_data.columns and display_data['mfg_variable_overhead'].sum() != 0
    
    if has_mfg_cogs:
        # Show detailed manufacturing breakdown
        sections.append({'type': 'detail', 'label': 'COGS - Purchased Products', 'field': 'cogs_buy', 'indent': 1, 'negative': True})
        sections.append({'type': 'detail', 'label': 'COGS - Manufactured (Direct)', 'field': 'cogs_make', 'indent': 1, 'negative': True})
        
        # Show variable and fixed overhead separately if available
        if has_var_overhead:
            sections.append({'type': 'detail', 'label': 'Mfg Variable Overhead', 'field': 'mfg_variable_overhead', 'indent': 1, 'negative': True})
            sections.append({'type': 'detail', 'label': 'Mfg Fixed Overhead', 'field': 'mfg_fixed_overhead', 'indent': 1, 'negative': True})
        else:
            # Fall back to combined overhead
            sections.append({'type': 'detail', 'label': 'Manufacturing Overhead', 'field': 'mfg_overhead', 'indent': 1, 'negative': True})
        
        sections.append({'type': 'detail', 'label': 'Manufacturing Depreciation', 'field': 'mfg_depreciation', 'indent': 1, 'negative': True})
        sections.append({'type': 'divider'})
    
    sections.append({'type': 'subtotal', 'label': 'Total Cost of Goods Sold', 'field': 'total_cogs', 'negative': True})
    
    # Gross Profit
    sections.append({'type': 'divider'})
    sections.append({'type': 'total', 'label': 'GROSS PROFIT', 'field': 'total_gross_profit', 'show_pct': 'total_revenue'})
    
    # OpEx
    sections.append({'type': 'section', 'label': '💼 OPERATING EXPENSES'})
    
    if show_detail.get('opex', True):
        sections.append({'type': 'detail', 'label': 'Personnel & Salaries', 'field': 'opex_personnel', 'indent': 1, 'negative': True})
        sections.append({'type': 'detail', 'label': 'Facilities & Utilities', 'field': 'opex_facilities', 'indent': 1, 'negative': True})
        sections.append({'type': 'detail', 'label': 'Administrative', 'field': 'opex_admin', 'indent': 1, 'negative': True})
        sections.append({'type': 'detail', 'label': 'Sales & Marketing', 'field': 'opex_sales', 'indent': 1, 'negative': True})
        sections.append({'type': 'detail', 'label': 'Other Operating', 'field': 'opex_other', 'indent': 1, 'negative': True})
        sections.append({'type': 'divider'})
    
    sections.append({'type': 'subtotal', 'label': 'Total Operating Expenses', 'field': 'total_opex', 'negative': True})
    
    # Operating Results
    sections.append({'type': 'section', 'label': '📊 OPERATING RESULTS'})
    sections.append({'type': 'subtotal', 'label': 'EBITDA', 'field': 'ebitda', 'show_pct': 'total_revenue'})
    sections.append({'type': 'detail', 'label': 'Depreciation & Amortization', 'field': 'depreciation', 'indent': 1, 'negative': True})
    sections.append({'type': 'divider'})
    sections.append({'type': 'subtotal', 'label': 'EBIT (Operating Income)', 'field': 'ebit', 'show_pct': 'total_revenue'})
    sections.append({'type': 'detail', 'label': 'Interest Expense', 'field': 'interest_expense', 'indent': 1, 'negative': True})
    sections.append({'type': 'divider'})
    sections.append({'type': 'subtotal', 'label': 'EBT (Pre-Tax Income)', 'field': 'ebt'})
    sections.append({'type': 'detail', 'label': 'Income Tax Expense', 'field': 'tax_expense', 'indent': 1, 'negative': True})
    
    # Net Income
    sections.append({'type': 'divider'})
    sections.append({'type': 'total', 'label': 'NET INCOME', 'field': 'net_income', 'show_pct': 'total_revenue'})
    
    return sections


def get_balance_sheet_structure(show_detail: Dict[str, bool], display_data: pd.DataFrame = None) -> List[Dict]:
    """Define balance sheet line items structure.
    
    NEW: Shows manufacturing splits for inventory and PPE when manufacturing strategy is active.
    """
    sections = []
    
    # Check if manufacturing data is available
    has_mfg_data = False
    if display_data is not None and not display_data.empty:
        has_mfg_data = (
            'inventory_raw_materials' in display_data.columns and 
            display_data['inventory_raw_materials'].sum() > 0
        ) or (
            'ppe_manufacturing' in display_data.columns and 
            display_data['ppe_manufacturing'].sum() > 0
        )
    
    # Assets
    sections.append({'type': 'section', 'label': '📊 ASSETS'})
    sections.append({'type': 'subtotal', 'label': 'Current Assets', 'field': 'total_current_assets', 'indent': 1})
    
    if show_detail.get('assets', True):
        sections.append({'type': 'detail', 'label': 'Cash & Equivalents', 'field': 'cash_and_equivalents', 'indent': 2})
        sections.append({'type': 'detail', 'label': 'Accounts Receivable', 'field': 'accounts_receivable', 'indent': 2})
        
        # NEW: Show inventory split if manufacturing is active
        if has_mfg_data:
            sections.append({'type': 'detail', 'label': 'Inventory - Raw Materials (Mfg)', 'field': 'inventory_raw_materials', 'indent': 2})
            sections.append({'type': 'detail', 'label': 'Inventory - Finished Goods', 'field': 'inventory_finished_goods', 'indent': 2})
            sections.append({'type': 'detail', 'label': 'Total Inventory', 'field': 'inventory', 'indent': 2})
        else:
            sections.append({'type': 'detail', 'label': 'Inventory', 'field': 'inventory', 'indent': 2})
    
    sections.append({'type': 'divider'})
    sections.append({'type': 'subtotal', 'label': 'Non-Current Assets', 'field': 'total_non_current_assets', 'indent': 1})
    
    if show_detail.get('assets', True):
        # NEW: Show PPE split if manufacturing is active
        if has_mfg_data:
            sections.append({'type': 'detail', 'label': 'PP&E - Other', 'field': 'ppe_other', 'indent': 2})
            sections.append({'type': 'detail', 'label': 'PP&E - Manufacturing', 'field': 'ppe_manufacturing', 'indent': 2})
            sections.append({'type': 'detail', 'label': 'Total Property, Plant & Equipment', 'field': 'property_plant_equipment', 'indent': 2})
            sections.append({'type': 'detail', 'label': 'Less: Accum Depr - Other', 'field': 'accumulated_depreciation_other', 'indent': 2, 'negative': True})
            sections.append({'type': 'detail', 'label': 'Less: Accum Depr - Manufacturing', 'field': 'accumulated_depreciation_manufacturing', 'indent': 2, 'negative': True})
            sections.append({'type': 'detail', 'label': 'Total Accumulated Depreciation', 'field': 'accumulated_depreciation', 'indent': 2, 'negative': True})
            sections.append({'type': 'detail', 'label': 'Net PP&E - Other', 'field': 'net_ppe_other', 'indent': 2})
            sections.append({'type': 'detail', 'label': 'Net PP&E - Manufacturing', 'field': 'net_ppe_manufacturing', 'indent': 2})
            sections.append({'type': 'detail', 'label': 'Total Net PP&E', 'field': 'net_ppe', 'indent': 2})
        else:
            sections.append({'type': 'detail', 'label': 'Property, Plant & Equipment', 'field': 'property_plant_equipment', 'indent': 2})
            sections.append({'type': 'detail', 'label': 'Less: Accumulated Depreciation', 'field': 'accumulated_depreciation', 'indent': 2, 'negative': True})
            sections.append({'type': 'detail', 'label': 'Net PP&E', 'field': 'net_ppe', 'indent': 2})
    
    sections.append({'type': 'divider'})
    sections.append({'type': 'total', 'label': 'TOTAL ASSETS', 'field': 'total_assets'})
    
    # Liabilities
    sections.append({'type': 'section', 'label': '📋 LIABILITIES'})
    sections.append({'type': 'subtotal', 'label': 'Current Liabilities', 'field': 'total_current_liabilities', 'indent': 1})
    
    if show_detail.get('liabilities', True):
        sections.append({'type': 'detail', 'label': 'Accounts Payable', 'field': 'accounts_payable', 'indent': 2})
        # NEW: Show accrued interest if available
        has_accrued_interest = display_data is not None and not display_data.empty and 'accrued_interest' in display_data.columns and display_data['accrued_interest'].sum() > 0
        if has_accrued_interest:
            sections.append({'type': 'detail', 'label': 'Accrued Interest', 'field': 'accrued_interest', 'indent': 2})
        sections.append({'type': 'detail', 'label': 'Short-term Debt', 'field': 'short_term_debt', 'indent': 2})
    
    sections.append({'type': 'divider'})
    sections.append({'type': 'subtotal', 'label': 'Non-Current Liabilities', 'field': 'total_non_current_liabilities', 'indent': 1})
    
    if show_detail.get('liabilities', True):
        # NEW: Show debt breakdown if available
        has_debt_breakdown = display_data is not None and not display_data.empty and 'debt_principal' in display_data.columns and display_data['debt_principal'].sum() > 0
        if has_debt_breakdown:
            sections.append({'type': 'detail', 'label': 'Debt Principal', 'field': 'debt_principal', 'indent': 2})
            sections.append({'type': 'detail', 'label': 'Long-term Debt (Total)', 'field': 'long_term_debt', 'indent': 2})
        else:
            sections.append({'type': 'detail', 'label': 'Long-term Debt', 'field': 'long_term_debt', 'indent': 2})
    
    sections.append({'type': 'divider'})
    sections.append({'type': 'subtotal', 'label': 'TOTAL LIABILITIES', 'field': 'total_liabilities'})
    
    # Equity
    sections.append({'type': 'section', 'label': '💰 EQUITY'})
    
    if show_detail.get('equity', True):
        sections.append({'type': 'detail', 'label': 'Share Capital', 'field': 'share_capital', 'indent': 1})
        sections.append({'type': 'detail', 'label': 'Retained Earnings', 'field': 'retained_earnings', 'indent': 1})
    
    sections.append({'type': 'subtotal', 'label': 'TOTAL EQUITY', 'field': 'total_equity'})
    
    sections.append({'type': 'divider'})
    sections.append({'type': 'total', 'label': 'TOTAL LIABILITIES & EQUITY', 'field': 'total_liabilities_and_equity'})
    
    return sections


def get_cash_flow_structure(show_detail: Dict[str, bool], display_data: pd.DataFrame = None) -> List[Dict]:
    """Define cash flow statement structure.
    
    NEW: Shows manufacturing investments separately when manufacturing strategy is active.
    """
    sections = []
    
    # Check if manufacturing data is available
    has_mfg_data = False
    if display_data is not None and not display_data.empty:
        has_mfg_data = (
            'mfg_ppe_investment' in display_data.columns and 
            display_data['mfg_ppe_investment'].sum() != 0
        ) or (
            'mfg_commissioning_costs' in display_data.columns and 
            display_data['mfg_commissioning_costs'].sum() != 0
        )
    
    # Operating
    sections.append({'type': 'section', 'label': '💵 OPERATING ACTIVITIES'})
    sections.append({'type': 'detail', 'label': 'Net Income', 'field': 'net_income', 'indent': 1})
    sections.append({'type': 'detail', 'label': 'Add: Depreciation & Amortization', 'field': 'depreciation_amortization', 'indent': 1})
    
    if show_detail.get('working_capital', True):
        sections.append({'type': 'detail', 'label': 'Change in Receivables', 'field': 'change_in_receivables', 'indent': 1})
        
        # NEW: Detailed inventory breakdown if available
        has_inv_detail = display_data is not None and not display_data.empty and 'change_in_inventory_raw_materials' in display_data.columns
        if has_inv_detail:
            sections.append({'type': 'detail', 'label': 'Change in Inventory - Raw Materials (Mfg)', 'field': 'change_in_inventory_raw_materials', 'indent': 1})
            sections.append({'type': 'detail', 'label': 'Change in Inventory - Finished Goods', 'field': 'change_in_inventory_finished_goods', 'indent': 1})
            sections.append({'type': 'detail', 'label': 'Total Change in Inventory', 'field': 'change_in_inventory', 'indent': 1})
        else:
            sections.append({'type': 'detail', 'label': 'Change in Inventory', 'field': 'change_in_inventory', 'indent': 1})
        
        sections.append({'type': 'detail', 'label': 'Change in Payables', 'field': 'change_in_payables', 'indent': 1})
        
        # NEW: Manufacturing working capital change if available
        has_mfg_wc = display_data is not None and not display_data.empty and 'change_in_manufacturing_wc' in display_data.columns and display_data['change_in_manufacturing_wc'].sum() != 0
        if has_mfg_wc:
            sections.append({'type': 'detail', 'label': 'Change in Manufacturing WC', 'field': 'change_in_manufacturing_wc', 'indent': 1})
    
    sections.append({'type': 'subtotal', 'label': 'Net Cash from Operations', 'field': 'cash_from_operations'})
    
    # Investing - NEW: Split manufacturing vs other
    sections.append({'type': 'section', 'label': '🏭 INVESTING ACTIVITIES'})
    
    if has_mfg_data:
        sections.append({'type': 'detail', 'label': 'Capital Expenditures - Other', 'field': 'capital_expenditures_other', 'indent': 1, 'negative': True})
        sections.append({'type': 'detail', 'label': 'Manufacturing PP&E Investment', 'field': 'mfg_ppe_investment', 'indent': 1, 'negative': True})
        sections.append({'type': 'detail', 'label': 'Manufacturing Commissioning Costs', 'field': 'mfg_commissioning_costs', 'indent': 1, 'negative': True})
        sections.append({'type': 'detail', 'label': 'Manufacturing Working Capital', 'field': 'mfg_working_capital', 'indent': 1, 'negative': True})
        sections.append({'type': 'divider'})
        sections.append({'type': 'subtotal', 'label': 'Total Capital Expenditures', 'field': 'capital_expenditures', 'indent': 1, 'negative': True})
    else:
        sections.append({'type': 'detail', 'label': 'Capital Expenditures', 'field': 'capital_expenditures', 'indent': 1, 'negative': True})
    
    sections.append({'type': 'subtotal', 'label': 'Net Cash from Investing', 'field': 'cash_from_investing'})
    
    # Financing
    sections.append({'type': 'section', 'label': '🏦 FINANCING ACTIVITIES'})
    sections.append({'type': 'subtotal', 'label': 'Net Cash from Financing', 'field': 'cash_from_financing'})
    
    # Net Change
    sections.append({'type': 'divider'})
    sections.append({'type': 'total', 'label': 'NET CHANGE IN CASH', 'field': 'net_change_in_cash'})
    sections.append({'type': 'detail', 'label': 'Beginning Cash', 'field': 'beginning_cash', 'indent': 1})
    sections.append({'type': 'total', 'label': 'ENDING CASH', 'field': 'ending_cash'})
    
    return sections


def build_table_css() -> str:
    """Build CSS for the table with horizontal scrolling and frozen first column."""
    return f"""
    <style>
    .fs-container {{ 
        overflow-x: auto; 
        margin: 1rem 0; 
        max-width: 100%;
        border: 1px solid {BORDER_COLOR};
        border-radius: 4px;
    }}
    .fs-table {{ 
        border-collapse: separate; 
        border-spacing: 0;
        font-family: 'Inter', 'Segoe UI', sans-serif; 
        font-size: 13px; 
        min-width: max-content;
    }}
    .fs-table th {{ 
        background: linear-gradient(180deg, {DARK_BG}, {DARKER_BG}); 
        color: {GOLD}; 
        padding: 12px 10px; 
        text-align: right; 
        font-weight: 600; 
        border-bottom: 2px solid {GOLD}; 
        white-space: nowrap;
        position: sticky;
        top: 0;
        z-index: 1;
    }}
    .fs-table th:first-child {{ 
        text-align: left; 
        min-width: 220px;
        position: sticky;
        left: 0;
        z-index: 3;
        background: {DARKER_BG};
        border-right: 2px solid {GOLD};
    }}
    .fs-table th.actual {{ color: {ACTUAL_COLOR}; }}
    .fs-table th.forecast {{ color: {FORECAST_COLOR}; }}
    .fs-table th.mixed {{ color: {MIXED_COLOR}; }}
    .fs-table td {{ 
        padding: 8px 10px; 
        text-align: right; 
        border-bottom: 1px solid {BORDER_COLOR}; 
        color: {TEXT_WHITE};
        white-space: nowrap;
    }}
    .fs-table td:first-child {{ 
        text-align: left;
        position: sticky;
        left: 0;
        z-index: 2;
        background: {DARK_BG};
        border-right: 2px solid {BORDER_COLOR};
    }}
    .fs-table tr:hover td {{ background: {GOLD_LIGHT} !important; }}
    .fs-table tr:hover td:first-child {{ background: {DARK_BG} !important; border-right: 2px solid {GOLD}; }}
    .fs-table .section-header td {{ background: {GOLD_LIGHT} !important; color: {GOLD}; font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; border-top: 2px solid {BORDER_COLOR}; }}
    .fs-table .section-header td:first-child {{ background: {GOLD_LIGHT} !important; }}
    .fs-table .subtotal td {{ color: {GOLD}; font-weight: 600; background: rgba(212, 165, 55, 0.05); }}
    .fs-table .subtotal td:first-child {{ background: rgba(212, 165, 55, 0.08); }}
    .fs-table .total td {{ color: {TEXT_WHITE}; font-weight: 700; background: rgba(212, 165, 55, 0.15) !important; border-top: 2px solid {GOLD}; border-bottom: 2px solid {GOLD}; font-size: 14px; }}
    .fs-table .total td:first-child {{ background: rgba(212, 165, 55, 0.2) !important; }}
    .fs-table .detail td {{ color: {TEXT_MUTED}; font-size: 12px; }}
    .fs-table .annual-total td {{ background: rgba(59, 130, 246, 0.1) !important; font-weight: 600; border-top: 2px solid {BLUE}; }}
    .fs-table .annual-total td:first-child {{ background: rgba(59, 130, 246, 0.15) !important; }}
    .fs-table .indent-1 td:first-child {{ padding-left: 25px; }}
    .fs-table .indent-2 td:first-child {{ padding-left: 45px; }}
    .fs-table .negative {{ color: {RED} !important; }}
    .fs-table .pct {{ color: {TEXT_MUTED}; font-size: 11px; margin-left: 4px; }}
    .fs-table .divider td {{ padding: 4px; border: none; background: transparent !important; }}
    .fs-table .divider td:first-child {{ background: {DARK_BG} !important; }}
    .fs-table .actual-cell {{ background: rgba(16, 185, 129, 0.05); }}
    .fs-table .forecast-cell {{ background: rgba(59, 130, 246, 0.05); }}
    </style>
    """


def build_table_header(periods: List[str], data: pd.DataFrame) -> str:
    """Build table header row."""
    html = '<div class="fs-container"><table class="fs-table"><thead><tr><th>Line Item</th>'
    
    for period in periods:
        period_data = data[data['period_label'] == period]
        if not period_data.empty:
            row = period_data.iloc[0]
            data_type = row.get('data_type', 'Forecast')
            
            if data_type == 'Actual':
                css_class = 'actual'
                label = f'{period}<br><span style="font-size:10px;color:{ACTUAL_COLOR}">Actual</span>'
            elif 'Mixed' in str(data_type):
                css_class = 'mixed'
                label = f'{period}<br><span style="font-size:10px;color:{MIXED_COLOR}">Mixed</span>'
            else:
                css_class = 'forecast'
                label = f'{period}<br><span style="font-size:10px;color:{FORECAST_COLOR}">Forecast</span>'
            
            html += f'<th class="{css_class}">{label}</th>'
        else:
            html += f'<th>{period}</th>'
    
    html += '</tr></thead><tbody>'
    return html


def build_table_body(sections: List[Dict], periods: List[str], data: pd.DataFrame) -> str:
    """Build table body rows."""
    html = ""
    
    for section in sections:
        row_type = section.get('type', 'detail')
        
        if row_type == 'section':
            html += f'<tr class="section-header"><td colspan="{len(periods) + 1}">{section["label"]}</td></tr>'
        elif row_type == 'divider':
            html += f'<tr class="divider"><td colspan="{len(periods) + 1}"></td></tr>'
        else:
            # Data row
            indent = section.get('indent', 0)
            css_class = f'{row_type} indent-{indent}' if indent else row_type
            is_annual = data.get('is_annual_total', pd.Series([False])).iloc[0] if not data.empty else False
            if is_annual:
                css_class += ' annual-total'
            
            html += f'<tr class="{css_class}"><td>{section["label"]}</td>'
            
            for period in periods:
                period_data = data[data['period_label'] == period]
                if period_data.empty:
                    html += '<td>-</td>'
                    continue
                
                row = period_data.iloc[0]
                field = section.get('field', '')
                value = row.get(field, 0) or 0
                
                # Format value
                if section.get('negative'):
                    value = -abs(value)
                
                if abs(value) >= 1_000_000:
                    formatted = f"{value/1_000_000:,.1f}M"
                elif abs(value) >= 1_000:
                    formatted = f"{value/1_000:,.0f}K"
                else:
                    formatted = f"{value:,.0f}"
                
                val_class = ""
                if value < 0:
                    val_class = "negative"
                    formatted = f"({formatted.replace('-', '')})"
                
                # Add cell background based on actual/forecast
                data_type = row.get('data_type', 'Forecast')
                if data_type == 'Actual':
                    val_class += " actual-cell"
                elif data_type == 'Forecast':
                    val_class += " forecast-cell"
                
                # Add percentage if requested
                pct_html = ""
                show_pct = section.get('show_pct')
                if show_pct and row.get(show_pct, 0) > 0:
                    pct = abs(value) / row[show_pct] * 100
                    pct_html = f'<span class="pct">({pct:.0f}%)</span>'
                
                html += f'<td class="{val_class}">{formatted}{pct_html}</td>'
            
            html += '</tr>'
    
    return html


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_financial_statements(
    forecast_result: Dict[str, Any],
    assumptions: Dict[str, Any],
    historic_data: Dict[str, Any] = None,
    db = None,
    scenario_id: str = None
):
    """Main render function for Financial Statements."""
    
    # Header
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 2px solid {GOLD};">
        <span style="font-size: 2rem;">📑</span>
        <div style="margin-left: 1rem;">
            <h2 style="color: {GOLD}; margin: 0; font-size: 1.5rem;">Financial Statements</h2>
            <p style="color: {TEXT_MUTED}; margin: 0; font-size: 0.85rem;">Income Statement • Balance Sheet • Cash Flow</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Legend
    st.markdown(f"""
    <div style="display: flex; gap: 1.5rem; margin-bottom: 1rem; font-size: 0.8rem; padding: 0.5rem 1rem; background: {DARK_BG}; border-radius: 4px;">
        <span><span style="color: {ACTUAL_COLOR};">●</span> Actual</span>
        <span><span style="color: {FORECAST_COLOR};">●</span> Forecast</span>
        <span><span style="color: {MIXED_COLOR};">●</span> Mixed (Actual + Forecast)</span>
        <span><span style="color: {GOLD};">■</span> Subtotal</span>
        <span><span style="color: {RED};">(123)</span> Expense/Outflow</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        view_mode = st.radio(
            "📅 Period View",
            options=["monthly", "annual"],
            format_func=lambda x: "Monthly" if x == "monthly" else "Annual",
            horizontal=True,
            key="fs_view_mode"
        )
    
    with col2:
        show_subtotals = st.checkbox("📊 Annual Subtotals", value=True, 
                                     disabled=view_mode == "annual", key="fs_subtotals")
    
    with col3:
        show_revenue_detail = st.checkbox("📈 Revenue Detail", value=True, key="fs_rev_detail")
    
    with col4:
        show_expense_detail = st.checkbox("💼 Expense Detail", value=True, key="fs_exp_detail")
    
    st.markdown("---")
    
    # Build data
    income_df, balance_df, cashflow_df = build_financial_data(
        forecast_result, assumptions, db, scenario_id
    )
    
    if income_df.empty:
        st.warning("⚠️ No forecast data available. Please run a forecast first.")
        return
    
    # Define value columns for aggregation
    is_value_cols = ['rev_wear_existing', 'rev_service_existing', 'rev_wear_prospect', 'rev_service_prospect',
                     'revenue_existing', 'revenue_prospect', 'total_revenue', 'total_cogs', 'total_gross_profit',
                     'opex_personnel', 'opex_facilities', 'opex_admin', 'opex_sales', 'opex_other', 'total_opex',
                     'ebitda', 'depreciation', 'ebit', 'interest_expense', 'ebt', 'tax_expense', 'net_income']
    
    bs_value_cols = ['cash_and_equivalents', 'accounts_receivable', 'inventory', 'total_current_assets',
                     'property_plant_equipment', 'accumulated_depreciation', 'net_ppe', 'total_non_current_assets',
                     'total_assets', 'accounts_payable', 'short_term_debt', 'total_current_liabilities',
                     'long_term_debt', 'total_non_current_liabilities', 'total_liabilities',
                     'share_capital', 'retained_earnings', 'total_equity', 'total_liabilities_and_equity']
    
    cf_value_cols = ['net_income', 'depreciation_amortization', 'change_in_receivables', 'change_in_inventory',
                     'change_in_payables', 'cash_from_operations', 'capital_expenditures', 'cash_from_investing',
                     'cash_from_financing', 'net_change_in_cash', 'beginning_cash', 'ending_cash']
    
    # Prepare display data
    if view_mode == "annual":
        is_display = aggregate_to_annual(income_df, is_value_cols)
        bs_display = balance_df.groupby('period_year').last().reset_index()  # Use end of year values
        bs_display['period_label'] = bs_display['period_year'].astype(str)
        cf_display = aggregate_to_annual(cashflow_df, cf_value_cols)
    elif show_subtotals:
        is_display = add_annual_subtotals(income_df, is_value_cols)
        bs_display = balance_df.copy()
        cf_display = add_annual_subtotals(cashflow_df, cf_value_cols)
    else:
        is_display = income_df.copy()
        bs_display = balance_df.copy()
        cf_display = cashflow_df.copy()
    
    # Show note for mixed years in annual view
    if view_mode == "annual":
        mixed_years = is_display[is_display.get('is_mixed', False) == True] if 'is_mixed' in is_display.columns else pd.DataFrame()
        if not mixed_years.empty:
            notes = []
            for _, row in mixed_years.iterrows():
                notes.append(f"**{int(row['period_year'])}**: {row.get('period_note', 'Contains actual and forecast data')}")
            
            st.markdown(f"""
            <div style="background: {MIXED_COLOR}20; border-left: 3px solid {MIXED_COLOR}; padding: 0.75rem; margin-bottom: 1rem; border-radius: 0 4px 4px 0;">
                <span style="color: {MIXED_COLOR}; font-weight: 600;">⚠️ Mixed Period Note</span><br>
                <span style="color: {TEXT_MUTED}; font-size: 0.9rem;">{"<br>".join(notes)}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # AI Assumptions Summary (if available)
    if AI_INTEGRATION_AVAILABLE and assumptions:
        with st.expander("🤖 AI Assumptions Status", expanded=False):
            # Extract assumption sources if available
            assumptions_summary = {}
            for key in ['revenue_growth', 'cogs_pct', 'opex_pct', 'capex_pct', 'ar_days', 'ap_days', 'inventory_days']:
                if key in assumptions:
                    assumptions_summary[key] = {
                        'value': assumptions[key],
                        'source': assumptions.get(f'{key}_source', 'default')
                    }
            if assumptions_summary:
                render_ai_assumptions_summary(assumptions_summary, show_details=True)
    
    # Tabs - NEW: Added Commentary tab
    tab_is, tab_bs, tab_cf, tab_download, tab_commentary = st.tabs([
        "📊 Income Statement", "📋 Balance Sheet", "💵 Cash Flow", "📥 Download", "📝 Commentary"
    ])
    
    with tab_is:
        show_detail = {'revenue': show_revenue_detail, 'opex': show_expense_detail}
        render_statement_table(is_display, 'income', show_detail, view_mode)
    
    with tab_bs:
        show_detail = {'assets': True, 'liabilities': True, 'equity': True}
        render_statement_table(bs_display, 'balance', show_detail, view_mode)
    
    with tab_cf:
        show_detail = {'working_capital': True}
        render_statement_table(cf_display, 'cashflow', show_detail, view_mode)
    
    with tab_download:
        st.markdown("### 📥 Download Financial Data")
        
        # CSV Downloads
        st.markdown("#### CSV Export")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = income_df.to_csv(index=False)
            st.download_button("📥 Income Statement (CSV)", csv, 
                             f"income_statement_{datetime.now():%Y%m%d}.csv", "text/csv",
                             use_container_width=True)
        
        with col2:
            csv = balance_df.to_csv(index=False)
            st.download_button("📥 Balance Sheet (CSV)", csv,
                             f"balance_sheet_{datetime.now():%Y%m%d}.csv", "text/csv",
                             use_container_width=True)
        
        with col3:
            csv = cashflow_df.to_csv(index=False)
            st.download_button("📥 Cash Flow (CSV)", csv,
                             f"cash_flow_{datetime.now():%Y%m%d}.csv", "text/csv",
                             use_container_width=True)
        
        st.markdown("---")
        
        # NEW: PDF Export
        st.markdown("#### PDF Export")
        try:
            from components.pdf_exporter import render_pdf_export_ui
            scenario_name = assumptions.get('scenario_name', 'Scenario')
            render_pdf_export_ui(income_df, balance_df, cashflow_df, scenario_name)
        except ImportError:
            st.info("PDF export requires reportlab. Install with: pip install reportlab")
        
        st.markdown("---")
        
        # NEW: Excel Export
        st.markdown("#### Excel Export")
        try:
            from components.excel_exporter import render_excel_export_ui
            scenario_name = assumptions.get('scenario_name', 'Scenario')
            render_excel_export_ui(income_df, balance_df, cashflow_df, scenario_name)
        except ImportError:
            st.info("Excel export requires openpyxl. Install with: pip install openpyxl")
    
    with tab_commentary:
        # NEW: Financial Commentary Generator
        st.markdown("### 📝 Financial Commentary")
        
        try:
            from components.commentary_generator import render_commentary_ui
            
            # Get forecast data for commentary
            forecast_data = {
                'summary': {
                    'total_revenue': income_df['total_revenue'].sum() if not income_df.empty else 0,
                    'total_ebit': income_df['ebit'].sum() if not income_df.empty else 0,
                }
            }
            
            scenario_name = assumptions.get('scenario_name', 'Scenario')
            render_commentary_ui(
                income_df, balance_df, cashflow_df,
                forecast_data, assumptions, scenario_name
            )
        except ImportError:
            st.info("Commentary generator is available. Ensure `components/commentary_generator.py` exists.")


# =============================================================================
# INTEGRATION FUNCTION
# =============================================================================

def render_three_statement_preview(
    forecast_result: Dict[str, Any],
    assumptions: Dict[str, Any],
    historic_data: Dict[str, Any] = None,
    db = None,
    scenario_id: str = None
):
    """Drop-in replacement for forecast_section.py integration."""
    render_financial_statements(
        forecast_result=forecast_result,
        assumptions=assumptions,
        historic_data=historic_data,
        db=db,
        scenario_id=scenario_id
    )


