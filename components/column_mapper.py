"""
Column Mapping Module
=====================
Reusable column mapping UI component for all CSV imports.
Provides consistent user experience across all import screens.

NEW: Supports both wide and long format for detailed line items.

Usage:
    from components.column_mapper import render_import_with_mapping, FIELD_CONFIGS
    
    # In your import screen:
    render_import_with_mapping(
        db=db,
        user_id=user_id,
        import_type='aged_debtors',
        scenario_id=scenario_id  # Optional, required for some imports
    )

Or use the lower-level components:
    from components.column_mapper import (
        render_column_mapper,
        validate_mapping,
        apply_mapping,
        FIELD_CONFIGS
    )
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional, List, Tuple, Callable, Any
from datetime import datetime
import re


# =============================================================================
# WIDE TO LONG FORMAT TRANSFORMATION (NEW)
# =============================================================================

def extract_period_from_column(column_name: str) -> Optional[str]:
    """
    Extract period date from column name.
    
    Handles formats like:
    - FY2024 -> 2024-12-01
    - FY2023_Restated -> 2023-12-01 (handles _Restated suffix)
    - YTD_Oct2025 -> 2025-10-01
    - 2024 -> 2024-12-01
    - 2024-12 -> 2024-12-01
    """
    col_upper = str(column_name).upper().strip()
    
    # Remove common suffixes like _Restated, _Restated_2, etc.
    col_clean = re.sub(r'_RESTATED.*$', '', col_upper)
    col_clean = re.sub(r'_CHANGE.*$', '', col_clean)
    col_clean = re.sub(r'_PCT.*$', '', col_clean)
    col_clean = re.sub(r'%$', '', col_clean).strip()
    
    # FY2024 format (with or without _Restated suffix)
    if col_clean.startswith('FY'):
        year_match = re.search(r'FY(\d{4})', col_clean)
        if year_match:
            year = int(year_match.group(1))
            return f"{year}-12-01"
    
    # YTD_Oct2025 format
    if 'YTD' in col_clean:
        # Try YTD_Oct2025 format
        month_match = re.search(r'YTD[_\s]*([A-Z]{3})(\d{4})', col_clean)
        if month_match:
            month_name = month_match.group(1)
            year = int(month_match.group(2))
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            month_num = month_map.get(month_name, 12)
            return f"{year}-{month_num:02d}-01"
    
    # YYYY-MM format
    date_match = re.search(r'(\d{4})-(\d{2})', col_clean)
    if date_match:
        year = int(date_match.group(1))
        month = int(date_match.group(2))
        return f"{year}-{month:02d}-01"
    
    # YYYY format (standalone year)
    year_match = re.search(r'^(\d{4})$', col_clean)
    if year_match:
        year = int(year_match.group(1))
        return f"{year}-12-01"
    
    return None


def transform_wide_to_long(df: pd.DataFrame, import_type: str) -> pd.DataFrame:
    """
    Transform wide format (line items as rows, periods as columns) to long format.
    
    Wide Format:
        Line Item, Category, FY2024, FY2023, ...
        Revenue, Revenue, 1000000, 900000, ...
        COGS, Cost of Sales, -600000, -540000, ...
    
    Long Format:
        period_date, line_item_name, category, amount
        2024-12-01, Revenue, Revenue, 1000000
        2023-12-01, Revenue, Revenue, 900000
        2024-12-01, COGS, Cost of Sales, -600000
    """
    if df.empty:
        return pd.DataFrame()
    
    # Identify columns (case-insensitive, handle variations)
    line_item_col = None
    category_col = None
    sub_category_col = None
    
    for col in df.columns:
        col_lower = str(col).lower().strip().replace(' ', '_').replace('-', '_')
        # Check for line item column (various formats)
        if ('line' in col_lower and 'item' in col_lower) or col_lower == 'line_item' or col_lower == 'lineitem':
            line_item_col = col
        # Check for category column
        elif ('category' in col_lower and 'sub' not in col_lower) or col_lower == 'category':
            category_col = col
        # Check for sub_category column (various formats: Sub_Category, Sub-Category, SubCategory)
        elif ('sub' in col_lower and 'category' in col_lower) or col_lower == 'sub_category' or col_lower == 'subcategory':
            sub_category_col = col
    
    if not line_item_col:
        # Try first column if it looks like line items (text data)
        if len(df.columns) > 0:
            first_col = df.columns[0]
            if df[first_col].dtype == 'object' and len(df[first_col].dropna()) > 0:
                # Check if first column contains text that looks like line items
                sample_values = df[first_col].dropna().head(5).astype(str)
                if any(len(str(v)) > 3 and not str(v).replace('.', '').replace('-', '').isdigit() for v in sample_values):
                    line_item_col = first_col
    
    if not line_item_col:
        st.error("Could not identify 'Line Item' column. Please ensure your file has a column with line item names.")
        return pd.DataFrame()
    
    # Identify period columns (exclude metadata columns)
    period_columns = []
    excluded_patterns = ['change', 'pct', '%', 'yoy', 'budget', 'variance', 'var', 'notes', 'py_ytd']  # Exclude these from period detection
    
    for col in df.columns:
        # Skip metadata columns
        if col in [line_item_col, category_col, sub_category_col]:
            continue
        
        col_lower = str(col).lower().strip()
        
        # Skip change/percentage/budget columns (unless they're period columns with suffixes)
        # First check if it's a period column (handles FY2023_Restated)
        period_date = extract_period_from_column(col)
        if period_date:
            # It's a period column, include it
            period_columns.append((col, period_date))
        elif any(pattern in col_lower for pattern in excluded_patterns):
            # It's a metadata column, skip it
            continue
        # If it doesn't match excluded patterns and isn't a period, it might be a period column we didn't recognize
        # Try to extract period anyway (handles edge cases)
        elif extract_period_from_column(col):
            period_columns.append((col, extract_period_from_column(col)))
    
    if not period_columns:
        st.error("Could not identify period columns. Expected formats: FY2024, FY2023_Restated, YTD_Oct2025, 2024-12, or 2024")
        st.info(f"Available columns: {', '.join(df.columns)}")
        return pd.DataFrame()
    
    # Transform to long format
    result_rows = []
    
    for _, row in df.iterrows():
        line_item_name = str(row[line_item_col]).strip() if pd.notna(row[line_item_col]) else ''
        
        # Skip empty rows, totals, margins
        if not line_item_name or len(line_item_name) < 2:
            continue
        if 'Total' in line_item_name and 'Total Cost' not in line_item_name:
            continue
        if '%' in line_item_name or 'Margin' in line_item_name:
            continue
        
        # Get category
        category = 'Other'
        if category_col and pd.notna(row[category_col]):
            category = str(row[category_col]).strip()
            if not category or category == '':
                category = 'Other'
        
        # Get sub_category if available (handle Sub_Category, Sub-Category, SubCategory)
        sub_category = None
        if sub_category_col and pd.notna(row[sub_category_col]):
            sub_category_val = str(row[sub_category_col]).strip()
            if sub_category_val and sub_category_val != '' and sub_category_val.lower() != 'nan':
                sub_category = sub_category_val
        
        # Process each period column
        for period_col, period_date in period_columns:
            value = row[period_col]
            
            # Skip empty values
            if pd.isna(value) or value == '' or str(value).strip() == '':
                continue
            
            # Parse value (handle accounting format with parentheses)
            try:
                if isinstance(value, str):
                    value_str = value.strip()
                    if value_str.startswith('(') and value_str.endswith(')'):
                        amount = -float(value_str.strip('()').replace(',', ''))
                    else:
                        amount = float(value_str.replace(',', ''))
                else:
                    amount = float(value)
            except:
                continue
            
            # Create long format row
            result_rows.append({
                'period_date': period_date,
                'line_item_name': line_item_name,
                'category': category,
                'sub_category': sub_category,
                'amount': amount
            })
    
    return pd.DataFrame(result_rows)


# =============================================================================
# FIELD CONFIGURATIONS FOR ALL IMPORT TYPES
# =============================================================================

FIELD_CONFIGS = {
    
    # -------------------------------------------------------------------------
    # WORKING CAPITAL
    # -------------------------------------------------------------------------
    
    'aged_debtors': {
        'display_name': 'Aged Debtors',
        'icon': '📈',
        'description': 'Outstanding customer invoices',
        'table': 'aged_debtors',
        'requires_scenario': False,
        'fields': {
            'customer_name': {
                'label': 'Customer Name',
                'required': True,
                'type': 'text',
                'description': 'Name of the customer/debtor',
                'hints': ['customer', 'debtor', 'client', 'name', 'account', 'cust', 'customer_name']
            },
            'amount_due': {
                'label': 'Amount Due',
                'required': True,
                'type': 'number',
                'description': 'Outstanding invoice amount',
                'hints': ['amount', 'value', 'balance', 'outstanding', 'total', 'due', 'amt', 'amount_due']
            },
            'invoice_date': {
                'label': 'Invoice Date',
                'required': True,
                'type': 'date',
                'description': 'Date the invoice was issued',
                'hints': ['invoice_date', 'inv_date', 'date', 'issued', 'doc_date']
            },
            'due_date': {
                'label': 'Due Date',
                'required': True,
                'type': 'date',
                'description': 'Payment due date',
                'hints': ['due_date', 'due', 'payment_date', 'payable', 'pay_date']
            },
            'invoice_ref': {
                'label': 'Invoice Reference',
                'required': False,
                'type': 'text',
                'description': 'Invoice number or reference',
                'hints': ['invoice', 'ref', 'reference', 'inv_no', 'number', 'doc', 'invoice_ref', 'invoice_number']
            },
            'payment_terms': {
                'label': 'Payment Terms (Days)',
                'required': False,
                'type': 'number',
                'description': 'Payment terms in days',
                'hints': ['terms', 'payment_terms', 'days', 'net']
            }
        }
    },
    
    'aged_creditors': {
        'display_name': 'Aged Creditors',
        'icon': '📉',
        'description': 'Outstanding supplier invoices',
        'table': 'aged_creditors',
        'requires_scenario': False,
        'fields': {
            'supplier_name': {
                'label': 'Supplier Name',
                'required': True,
                'type': 'text',
                'description': 'Name of the supplier/vendor',
                'hints': ['supplier', 'vendor', 'creditor', 'name', 'account', 'supplier_name']
            },
            'amount_due': {
                'label': 'Amount Due',
                'required': True,
                'type': 'number',
                'description': 'Outstanding invoice amount',
                'hints': ['amount', 'value', 'balance', 'outstanding', 'total', 'due', 'amt', 'amount_due']
            },
            'invoice_date': {
                'label': 'Invoice Date',
                'required': True,
                'type': 'date',
                'description': 'Date invoice was received',
                'hints': ['invoice_date', 'inv_date', 'date', 'issued', 'doc_date']
            },
            'due_date': {
                'label': 'Due Date',
                'required': True,
                'type': 'date',
                'description': 'Payment due date',
                'hints': ['due_date', 'due', 'payment_date', 'payable', 'pay_date']
            },
            'invoice_ref': {
                'label': 'Invoice/PO Reference',
                'required': False,
                'type': 'text',
                'description': 'Invoice or PO number',
                'hints': ['invoice', 'ref', 'reference', 'po', 'number', 'invoice_ref', 'invoice_number', 'po_number']
            },
            'currency': {
                'label': 'Currency',
                'required': False,
                'type': 'text',
                'description': 'Currency code (ZAR, USD)',
                'hints': ['currency', 'curr', 'ccy']
            }
        }
    },
    
    'suppliers': {
        'display_name': 'Suppliers',
        'icon': '🏭',
        'description': 'Supplier master data',
        'table': 'creditors',
        'requires_scenario': True,
        'fields': {
            'name': {
                'label': 'Supplier Name',
                'required': True,
                'type': 'text',
                'description': 'Name of the supplier',
                'hints': ['name', 'supplier', 'vendor', 'supplier_name']
            },
            'creditor_type': {
                'label': 'Type',
                'required': False,
                'type': 'text',
                'description': 'domestic, international, etc.',
                'hints': ['type', 'creditor_type', 'supplier_type']
            },
            'standard_payment_days': {
                'label': 'Payment Terms (Days)',
                'required': False,
                'type': 'number',
                'description': 'Standard payment terms',
                'hints': ['payment_terms', 'terms', 'days', 'standard_payment_days', 'payment_days']
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # CUSTOMERS
    # -------------------------------------------------------------------------
    
    'customers': {
        'display_name': 'Customers',
        'icon': '👥',
        'description': 'Customer master data',
        'table': 'customers',
        'requires_scenario': False,
        'fields': {
            'customer_code': {
                'label': 'Customer Code',
                'required': True,
                'type': 'text',
                'description': 'Unique customer identifier',
                'hints': ['customer_code', 'code', 'cust_code', 'account_code', 'id']
            },
            'customer_name': {
                'label': 'Customer Name',
                'required': True,
                'type': 'text',
                'description': 'Name of the customer',
                'hints': ['customer_name', 'name', 'customer', 'company_name']
            },
        }
    },
    
    # -------------------------------------------------------------------------
    # FLEET / INSTALLED BASE
    # -------------------------------------------------------------------------
    
    'installed_base': {
        'display_name': 'Installed Base',
        'icon': '🚛',
        'description': 'Fleet of installed machines',
        'table': 'installed_base',
        'requires_scenario': True,
        'fields': {
            'machine_id': {
                'label': 'Machine ID',
                'required': True,
                'type': 'text',
                'description': 'Unique machine identifier',
                'hints': ['machine_id', 'id', 'machine', 'equipment_id', 'unit_id']
            },
            'customer_name': {
                'label': 'Customer Name',
                'required': True,
                'type': 'text',
                'description': 'Customer who owns/operates the machine',
                'hints': ['customer', 'customer_name', 'client', 'name']
            },
            'site_name': {
                'label': 'Site Name',
                'required': False,
                'type': 'text',
                'description': 'Site/mine location',
                'hints': ['site', 'site_name', 'location', 'mine', 'facility']
            },
            'machine_model': {
                'label': 'Machine Model',
                'required': True,
                'type': 'text',
                'description': 'Machine model/type',
                'hints': ['model', 'machine_model', 'type', 'equipment_type', 'machine_type']
            },
            'commission_date': {
                'label': 'Commission Date',
                'required': True,
                'type': 'date',
                'description': 'Date machine was commissioned',
                'hints': ['commission_date', 'commission', 'start_date', 'install_date', 'date']
            }
        }
    },
    
    'wear_profiles': {
        'display_name': 'Wear Profiles',
        'icon': '⚙️',
        'description': 'Wear profile definitions for machine models',
        'table': 'wear_profiles',
        'requires_scenario': False,
        'fields': {
            'machine_model': {
                'label': 'Machine Model',
                'required': True,
                'type': 'text',
                'description': 'Machine model identifier',
                'hints': ['machine_model', 'model', 'type', 'equipment_type']
            },
            'liner_life_months': {
                'label': 'Liner Life (Months)',
                'required': True,
                'type': 'number',
                'description': 'Average liner life in months',
                'hints': ['liner_life', 'liner_life_months', 'life', 'months']
            },
            'refurb_interval_months': {
                'label': 'Refurb Interval (Months)',
                'required': True,
                'type': 'number',
                'description': 'Average refurbishment interval',
                'hints': ['refurb_interval', 'refurb_interval_months', 'refurb', 'interval']
            },
            'avg_consumable_revenue': {
                'label': 'Avg Consumable Revenue',
                'required': True,
                'type': 'number',
                'description': 'Average revenue per consumable sale',
                'hints': ['avg_consumable_revenue', 'consumable_revenue', 'revenue', 'price']
            },
            'avg_refurb_revenue': {
                'label': 'Avg Refurb Revenue',
                'required': True,
                'type': 'number',
                'description': 'Average revenue per refurbishment',
                'hints': ['avg_refurb_revenue', 'refurb_revenue', 'revenue']
            },
            'gross_margin_liner': {
                'label': 'Liner Gross Margin %',
                'required': False,
                'type': 'number',
                'description': 'Gross margin percentage for liners',
                'hints': ['gross_margin_liner', 'margin', 'margin_pct']
            },
            'gross_margin_refurb': {
                'label': 'Refurb Gross Margin %',
                'required': False,
                'type': 'number',
                'description': 'Gross margin percentage for refurbishments',
                'hints': ['gross_margin_refurb', 'margin']
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # PIPELINE / PROSPECTS
    # -------------------------------------------------------------------------
    
    'prospects': {
        'display_name': 'Prospects',
        'icon': '🎯',
        'description': 'Sales opportunities and pipeline',
        'table': 'prospects',
        'requires_scenario': True,
        'fields': {
            'prospect_name': {
                'label': 'Prospect Name',
                'required': True,
                'type': 'text',
                'description': 'Prospect/customer name',
                'hints': ['prospect', 'prospect_name', 'customer', 'client', 'name', 'opportunity']
            },
            'site_name': {
                'label': 'Site Name',
                'required': False,
                'type': 'text',
                'description': 'Site/mine location',
                'hints': ['site', 'site_name', 'location', 'mine']
            },
            'machine_model': {
                'label': 'Machine Model',
                'required': True,
                'type': 'text',
                'description': 'Machine model being sold',
                'hints': ['machine_model', 'model', 'equipment', 'type']
            },
            'expected_close_date': {
                'label': 'Expected Close Date',
                'required': True,
                'type': 'date',
                'description': 'Expected deal close date',
                'hints': ['expected_close_date', 'close_date', 'expected_date', 'date']
            },
            'confidence_pct': {
                'label': 'Probability %',
                'required': False,
                'type': 'number',
                'description': 'Win probability (0-100)',
                'hints': ['confidence', 'probability', 'prob', 'win_probability', 'chance', 'likelihood']
            },
            'annual_liner_value': {
                'label': 'Annual Liner Value',
                'required': False,
                'type': 'number',
                'description': 'Expected annual consumable revenue',
                'hints': ['annual_liner_value', 'liner_value', 'consumable_revenue', 'annual_revenue']
            },
            'refurb_value': {
                'label': 'Refurb Value',
                'required': False,
                'type': 'number',
                'description': 'Expected refurbishment revenue',
                'hints': ['refurb_value', 'refurb_revenue', 'refurbishment']
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # HISTORICAL FINANCIALS
    # -------------------------------------------------------------------------
    
    'historic_financials': {
        'display_name': 'Historic Financials',
        'icon': '📊',
        'description': 'Monthly P&L history',
        'table': 'historic_financials',
        'requires_scenario': True,
        'fields': {
            'month': {
                'label': 'Month',
                'required': True,
                'type': 'date',
                'description': 'Month (YYYY-MM-01)',
                'hints': ['month', 'date', 'period', 'month_year']
            },
            'revenue': {
                'label': 'Revenue',
                'required': True,
                'type': 'number',
                'description': 'Total revenue',
                'hints': ['revenue', 'sales', 'turnover', 'income']
            },
            'cogs': {
                'label': 'COGS',
                'required': False,
                'type': 'number',
                'description': 'Cost of goods sold',
                'hints': ['cogs', 'cost', 'cos', 'cost_of_sales', 'cost_of_goods']
            },
            'gross_profit': {
                'label': 'Gross Profit',
                'required': False,
                'type': 'number',
                'description': 'Gross profit',
                'hints': ['gross_profit', 'gp', 'gross']
            },
            'opex': {
                'label': 'Operating Expenses',
                'required': False,
                'type': 'number',
                'description': 'Operating expenses',
                'hints': ['opex', 'expenses', 'operating', 'overheads']
            },
            'ebit': {
                'label': 'EBIT',
                'required': False,
                'type': 'number',
                'description': 'Earnings before interest & tax',
                'hints': ['ebit', 'operating_profit', 'operating_income']
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # HISTORIC CUSTOMER REVENUE
    # -------------------------------------------------------------------------
    
    'historic_customer_revenue': {
        'display_name': 'Customer Revenue History',
        'icon': '💰',
        'description': 'Monthly revenue by customer',
        'table': 'historic_customer_revenue',
        'requires_scenario': True,
        'fields': {
            'month': {
                'label': 'Month',
                'required': True,
                'type': 'date',
                'description': 'Month (YYYY-MM-01)',
                'hints': ['month', 'date', 'period']
            },
            'customer_code': {
                'label': 'Customer Code',
                'required': True,
                'type': 'text',
                'description': 'Customer identifier',
                'hints': ['customer_code', 'code', 'cust_code', 'account']
            },
            'customer_name': {
                'label': 'Customer Name',
                'required': False,
                'type': 'text',
                'description': 'Customer name',
                'hints': ['customer_name', 'customer', 'name']
            },
            'revenue': {
                'label': 'Revenue',
                'required': True,
                'type': 'number',
                'description': 'Revenue amount',
                'hints': ['revenue', 'sales', 'amount', 'value']
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # HISTORIC BALANCE SHEET
    # -------------------------------------------------------------------------
    
    'historic_balance_sheet': {
        'display_name': 'Balance Sheet History',
        'icon': '📋',
        'description': 'Monthly or annual balance sheet snapshots',
        'table': 'historical_balance_sheet',
        'requires_scenario': True,
        'fields': {
            'month': {
                'label': 'Period',
                'required': True,
                'type': 'date',
                'description': 'Month or year end date (YYYY-MM-DD)',
                'hints': ['month', 'date', 'period', 'as_at', 'year_end']
            },
            # Current Assets
            'cash_and_equivalents': {
                'label': 'Cash & Equivalents',
                'required': False,
                'type': 'number',
                'description': 'Cash at bank and short-term investments',
                'hints': ['cash', 'cash_and_equivalents', 'bank', 'cash_at_bank']
            },
            'accounts_receivable': {
                'label': 'Accounts Receivable',
                'required': False,
                'type': 'number',
                'description': 'Trade debtors',
                'hints': ['accounts_receivable', 'receivables', 'debtors', 'trade_debtors', 'ar']
            },
            'inventory': {
                'label': 'Inventory',
                'required': False,
                'type': 'number',
                'description': 'Stock on hand',
                'hints': ['inventory', 'stock', 'inventories']
            },
            'prepaid_expenses': {
                'label': 'Prepaid Expenses',
                'required': False,
                'type': 'number',
                'description': 'Prepayments and deposits',
                'hints': ['prepaid', 'prepaid_expenses', 'prepayments', 'deposits']
            },
            'total_current_assets': {
                'label': 'Total Current Assets',
                'required': False,
                'type': 'number',
                'description': 'Sum of current assets',
                'hints': ['total_current_assets', 'current_assets']
            },
            # Non-current Assets
            'ppe_net': {
                'label': 'Property, Plant & Equipment (Net)',
                'required': False,
                'type': 'number',
                'description': 'PPE net of depreciation',
                'hints': ['ppe', 'ppe_net', 'fixed_assets', 'property_plant_equipment', 'plant_and_equipment']
            },
            'intangible_assets': {
                'label': 'Intangible Assets',
                'required': False,
                'type': 'number',
                'description': 'Goodwill, software, patents',
                'hints': ['intangible', 'intangible_assets', 'intangibles', 'goodwill']
            },
            'total_noncurrent_assets': {
                'label': 'Total Non-current Assets',
                'required': False,
                'type': 'number',
                'description': 'Sum of non-current assets',
                'hints': ['total_noncurrent_assets', 'noncurrent_assets', 'non_current_assets']
            },
            'total_assets': {
                'label': 'Total Assets',
                'required': False,
                'type': 'number',
                'description': 'Sum of all assets',
                'hints': ['total_assets', 'assets']
            },
            # Current Liabilities
            'accounts_payable': {
                'label': 'Accounts Payable',
                'required': False,
                'type': 'number',
                'description': 'Trade creditors',
                'hints': ['accounts_payable', 'payables', 'creditors', 'trade_creditors', 'ap']
            },
            'accrued_expenses': {
                'label': 'Accrued Expenses',
                'required': False,
                'type': 'number',
                'description': 'Accruals and provisions',
                'hints': ['accrued', 'accrued_expenses', 'accruals', 'provisions']
            },
            'short_term_debt': {
                'label': 'Short-term Debt',
                'required': False,
                'type': 'number',
                'description': 'Loans due within 12 months',
                'hints': ['short_term_debt', 'current_debt', 'short_term_loans', 'overdraft']
            },
            'total_current_liabilities': {
                'label': 'Total Current Liabilities',
                'required': False,
                'type': 'number',
                'description': 'Sum of current liabilities',
                'hints': ['total_current_liabilities', 'current_liabilities']
            },
            # Non-current Liabilities
            'long_term_debt': {
                'label': 'Long-term Debt',
                'required': False,
                'type': 'number',
                'description': 'Loans due after 12 months',
                'hints': ['long_term_debt', 'long_term_loans', 'term_loans', 'borrowings']
            },
            'total_noncurrent_liabilities': {
                'label': 'Total Non-current Liabilities',
                'required': False,
                'type': 'number',
                'description': 'Sum of non-current liabilities',
                'hints': ['total_noncurrent_liabilities', 'noncurrent_liabilities']
            },
            'total_liabilities': {
                'label': 'Total Liabilities',
                'required': False,
                'type': 'number',
                'description': 'Sum of all liabilities',
                'hints': ['total_liabilities', 'liabilities']
            },
            # Equity
            'share_capital': {
                'label': 'Share Capital',
                'required': False,
                'type': 'number',
                'description': 'Issued share capital',
                'hints': ['share_capital', 'capital', 'issued_capital', 'stated_capital']
            },
            'retained_earnings': {
                'label': 'Retained Earnings',
                'required': False,
                'type': 'number',
                'description': 'Accumulated profits',
                'hints': ['retained_earnings', 'retained', 'accumulated_profits', 'reserves']
            },
            'total_equity': {
                'label': 'Total Equity',
                'required': False,
                'type': 'number',
                'description': 'Total shareholders equity',
                'hints': ['total_equity', 'equity', 'shareholders_equity', 'net_assets']
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # HISTORIC CASH FLOW STATEMENT
    # -------------------------------------------------------------------------
    
    'historic_cashflow': {
        'display_name': 'Cash Flow History',
        'icon': '💵',
        'description': 'Monthly or annual cash flow statements',
        'table': 'historical_cashflow',
        'requires_scenario': True,
        'fields': {
            'month': {
                'label': 'Period',
                'required': True,
                'type': 'date',
                'description': 'Month or year end date (YYYY-MM-DD)',
                'hints': ['month', 'date', 'period', 'year_end']
            },
            # Operating Activities
            'net_income': {
                'label': 'Net Income',
                'required': False,
                'type': 'number',
                'description': 'Net profit/loss',
                'hints': ['net_income', 'net_profit', 'profit', 'loss']
            },
            'depreciation_amortization': {
                'label': 'Depreciation & Amortization',
                'required': False,
                'type': 'number',
                'description': 'Non-cash expense',
                'hints': ['depreciation', 'depreciation_amortization', 'd&a', 'amortization']
            },
            'change_in_receivables': {
                'label': 'Change in Receivables',
                'required': False,
                'type': 'number',
                'description': 'Increase (negative) or decrease (positive)',
                'hints': ['change_in_receivables', 'receivables_change', 'ar_change']
            },
            'change_in_inventory': {
                'label': 'Change in Inventory',
                'required': False,
                'type': 'number',
                'description': 'Increase (negative) or decrease (positive)',
                'hints': ['change_in_inventory', 'inventory_change', 'stock_change']
            },
            'change_in_payables': {
                'label': 'Change in Payables',
                'required': False,
                'type': 'number',
                'description': 'Increase (positive) or decrease (negative)',
                'hints': ['change_in_payables', 'payables_change', 'ap_change']
            },
            'change_in_accruals': {
                'label': 'Change in Accruals',
                'required': False,
                'type': 'number',
                'description': 'Increase (positive) or decrease (negative)',
                'hints': ['change_in_accruals', 'accruals_change']
            },
            'cash_from_operations': {
                'label': 'Cash from Operations',
                'required': False,
                'type': 'number',
                'description': 'Net cash from operating activities',
                'hints': ['cash_from_operations', 'cfo', 'operating_cash_flow']
            },
            # Investing Activities
            'capital_expenditure': {
                'label': 'Capital Expenditure',
                'required': False,
                'type': 'number',
                'description': 'CAPEX (negative)',
                'hints': ['capital_expenditure', 'capex', 'capital_exp', 'pp&e']
            },
            'asset_disposals': {
                'label': 'Asset Disposals',
                'required': False,
                'type': 'number',
                'description': 'Proceeds from asset sales (positive)',
                'hints': ['asset_disposals', 'disposals', 'asset_sales']
            },
            'cash_from_investing': {
                'label': 'Cash from Investing',
                'required': False,
                'type': 'number',
                'description': 'Net cash from investing activities',
                'hints': ['cash_from_investing', 'cfi', 'investing_cash_flow']
            },
            # Financing Activities
            'debt_proceeds': {
                'label': 'Debt Proceeds',
                'required': False,
                'type': 'number',
                'description': 'Loan proceeds (positive)',
                'hints': ['debt_proceeds', 'loan_proceeds', 'borrowings']
            },
            'debt_repayment': {
                'label': 'Debt Repayment',
                'required': False,
                'type': 'number',
                'description': 'Loan repayments (negative)',
                'hints': ['debt_repayment', 'loan_repayment', 'repayments']
            },
            'dividends_paid': {
                'label': 'Dividends Paid',
                'required': False,
                'type': 'number',
                'description': 'Dividend payments (negative)',
                'hints': ['dividends', 'dividends_paid', 'distributions']
            },
            'cash_from_financing': {
                'label': 'Cash from Financing',
                'required': False,
                'type': 'number',
                'description': 'Net cash from financing activities',
                'hints': ['cash_from_financing', 'cff', 'financing_cash_flow']
            },
            # Summary
            'net_change_in_cash': {
                'label': 'Net Change in Cash',
                'required': False,
                'type': 'number',
                'description': 'Total change in cash position',
                'hints': ['net_change', 'net_change_in_cash', 'cash_change', 'change_in_cash']
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # DETAILED LINE ITEMS (NEW - Granular Historical Data)
    # -------------------------------------------------------------------------
    
    'historical_income_statement_line_items': {
        'display_name': 'Income Statement Line Items (Detailed)',
        'icon': '📊',
        'description': 'Individual line items from income statements (e.g., Accounting Fees, Advertising, Depreciation). Supports both WIDE format (line items as rows, periods as columns) and LONG format (one row per line item per period).',
        'table': 'historical_income_statement_line_items',
        'requires_scenario': True,
        'fields': {
            'period_date': {
                'label': 'Period Date',
                'required': True,
                'type': 'date',
                'description': 'Period date (YYYY-MM-DD)',
                'hints': ['period_date', 'date', 'period', 'month', 'year_end']
            },
            'line_item_name': {
                'label': 'Line Item Name',
                'required': True,
                'type': 'text',
                'description': 'Name of the line item (e.g., "Accounting Fees", "Depreciation")',
                'hints': ['line_item_name', 'line_item', 'item_name', 'account_name', 'description']
            },
            'category': {
                'label': 'Category',
                'required': True,
                'type': 'text',
                'description': 'Category (e.g., "Operating Expenses", "Revenue", "Other Income")',
                'hints': ['category', 'type', 'classification', 'group']
            },
            'sub_category': {
                'label': 'Sub-Category',
                'required': False,
                'type': 'text',
                'description': 'Sub-category if available',
                'hints': ['sub_category', 'subcategory', 'sub_category', 'detail']
            },
            'amount': {
                'label': 'Amount',
                'required': True,
                'type': 'number',
                'description': 'Amount for this line item (negative for expenses, positive for income)',
                'hints': ['amount', 'value', 'balance', 'total']
            }
        }
    },
    
    'historical_balance_sheet_line_items': {
        'display_name': 'Balance Sheet Line Items (Detailed)',
        'icon': '📋',
        'description': 'Individual line items from balance sheets (e.g., Property Plant and Equipment, Right-of-Use Assets). Supports both WIDE and LONG formats.',
        'table': 'historical_balance_sheet_line_items',
        'requires_scenario': True,
        'fields': {
            'period_date': {
                'label': 'Period Date',
                'required': True,
                'type': 'date',
                'description': 'Period date (YYYY-MM-DD)',
                'hints': ['period_date', 'date', 'period', 'month', 'year_end', 'as_at']
            },
            'line_item_name': {
                'label': 'Line Item Name',
                'required': True,
                'type': 'text',
                'description': 'Name of the line item (e.g., "Property Plant and Equipment", "Trade Receivables")',
                'hints': ['line_item_name', 'line_item', 'item_name', 'account_name', 'description']
            },
            'category': {
                'label': 'Category',
                'required': True,
                'type': 'text',
                'description': 'Category (e.g., "Non-Current Assets", "Current Assets", "Equity")',
                'hints': ['category', 'type', 'classification', 'group']
            },
            'sub_category': {
                'label': 'Sub-Category',
                'required': False,
                'type': 'text',
                'description': 'Sub-category if available (e.g., "PPE", "Leases", "Intangibles")',
                'hints': ['sub_category', 'subcategory', 'sub_category', 'detail']
            },
            'amount': {
                'label': 'Amount',
                'required': True,
                'type': 'number',
                'description': 'Amount for this line item',
                'hints': ['amount', 'value', 'balance', 'total']
            }
        }
    },
    
    'historical_cashflow_line_items': {
        'display_name': 'Cash Flow Line Items (Detailed)',
        'icon': '💵',
        'description': 'Individual line items from cash flow statements (e.g., Depreciation and Amortisation, Increase in Inventories). Supports both WIDE and LONG formats.',
        'table': 'historical_cashflow_line_items',
        'requires_scenario': True,
        'fields': {
            'period_date': {
                'label': 'Period Date',
                'required': True,
                'type': 'date',
                'description': 'Period date (YYYY-MM-DD)',
                'hints': ['period_date', 'date', 'period', 'month', 'year_end']
            },
            'line_item_name': {
                'label': 'Line Item Name',
                'required': True,
                'type': 'text',
                'description': 'Name of the line item (e.g., "Depreciation and Amortisation", "Increase in Inventories")',
                'hints': ['line_item_name', 'line_item', 'item_name', 'account_name', 'description']
            },
            'category': {
                'label': 'Category',
                'required': True,
                'type': 'text',
                'description': 'Category (e.g., "Operating Activities", "Investing Activities", "Financing Activities")',
                'hints': ['category', 'type', 'classification', 'group', 'section']
            },
            'sub_category': {
                'label': 'Sub-Category',
                'required': False,
                'type': 'text',
                'description': 'Sub-category if available',
                'hints': ['sub_category', 'subcategory', 'sub_category', 'detail']
            },
            'amount': {
                'label': 'Amount',
                'required': True,
                'type': 'number',
                'description': 'Amount for this line item (negative for cash outflows, positive for cash inflows)',
                'hints': ['amount', 'value', 'balance', 'total']
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # HISTORIC TRIAL BALANCE
    # -------------------------------------------------------------------------
    
    'historic_trial_balance': {
        'display_name': 'Trial Balance History',
        'icon': '📒',
        'description': 'Monthly or annual trial balance with account detail',
        'table': 'historical_trial_balance',
        'requires_scenario': True,
        'fields': {
            'month': {
                'label': 'Period',
                'required': True,
                'type': 'date',
                'description': 'Month or year end date (YYYY-MM-DD)',
                'hints': ['month', 'date', 'period', 'as_at', 'year_end']
            },
            'account_code': {
                'label': 'Account Code',
                'required': True,
                'type': 'text',
                'description': 'GL account code',
                'hints': ['account_code', 'code', 'account_number', 'gl_code', 'acc_code']
            },
            'account_name': {
                'label': 'Account Name',
                'required': True,
                'type': 'text',
                'description': 'GL account description',
                'hints': ['account_name', 'account', 'description', 'name', 'account_description']
            },
            'account_type': {
                'label': 'Account Type',
                'required': False,
                'type': 'text',
                'description': 'asset, liability, equity, revenue, expense',
                'hints': ['account_type', 'type', 'category', 'class']
            },
            'account_category': {
                'label': 'Account Category',
                'required': False,
                'type': 'text',
                'description': 'current, non-current, operating, etc.',
                'hints': ['account_category', 'sub_type', 'subcategory', 'sub_category']
            },
            'debit': {
                'label': 'Debit',
                'required': False,
                'type': 'number',
                'description': 'Debit balance',
                'hints': ['debit', 'dr', 'debit_balance']
            },
            'credit': {
                'label': 'Credit',
                'required': False,
                'type': 'number',
                'description': 'Credit balance',
                'hints': ['credit', 'cr', 'credit_balance']
            },
            'balance': {
                'label': 'Balance',
                'required': False,
                'type': 'number',
                'description': 'Net balance (debit - credit)',
                'hints': ['balance', 'net', 'amount', 'value']
            }
        }
    }
}


# =============================================================================
# CSV TEMPLATES (for download)
# =============================================================================

CSV_TEMPLATES = {
    'aged_debtors': 'customer_name,invoice_date,due_date,amount_due,invoice_ref,payment_terms\nCustomer A,2024-01-15,2024-02-15,50000,INV-001,30',
    'aged_creditors': 'supplier_name,invoice_date,due_date,amount_due,invoice_ref,currency\nSupplier X,2024-01-10,2024-02-10,25000,INV-SUP-001,ZAR',
    'customers': 'customer_code,customer_name,customer_type\nCUST001,Customer A,mining',
    'installed_base': 'machine_id,customer_name,site_name,machine_model,commission_date\nM001,Customer A,Site 1,Model X,2023-01-15',
    'wear_profiles': 'machine_model,liner_life_months,refurb_interval_months,avg_consumable_revenue,avg_refurb_revenue,gross_margin_liner,gross_margin_refurb\nModel X,12,36,50000,200000,0.35,0.28',
    'prospects': 'customer_name,site_name,machine_model,expected_commission_date,probability,annual_liner_value,refurb_value\nProspect A,Site 1,Model X,2025-06-01,75,60000,250000',
    'historic_financials': 'month,revenue,cogs,gross_profit,opex,ebit\n2024-01-01,1000000,600000,400000,200000,200000',
    'historic_customer_revenue': 'month,customer_code,customer_name,revenue\n2024-01-01,CUST001,Customer A,50000',
    'historic_balance_sheet': 'month,cash_and_equivalents,accounts_receivable,inventory,total_assets,accounts_payable,total_liabilities,share_capital,retained_earnings,total_equity\n2024-12-31,500000,300000,200000,5000000,150000,2000000,2000000,1000000,3000000',
    'historic_cashflow': 'month,net_income,depreciation_amortization,change_in_receivables,change_in_inventory,change_in_payables,cash_from_operations,capital_expenditure,cash_from_investing,debt_proceeds,debt_repayment,cash_from_financing,net_change_in_cash\n2024-01-01,150000,50000,-10000,-5000,8000,183000,-50000,-50000,0,0,0,83000',
    'historical_income_statement_line_items': 'period_date,line_item_name,category,sub_category,amount\n2024-12-01,Revenue,Revenue,,1000000\n2024-12-01,Accounting Fees,Operating Expenses,,-50000',
    'historical_balance_sheet_line_items': 'period_date,line_item_name,category,sub_category,amount\n2024-12-31,Property Plant and Equipment,Non-Current Assets,PPE,5000000',
    'historical_cashflow_line_items': 'period_date,line_item_name,category,sub_category,amount\n2024-12-01,Depreciation and Amortisation,Operating Activities,,50000'
}


# =============================================================================
# COLUMN MAPPING UI
# =============================================================================

def render_column_mapper(
    csv_columns: List[str], 
    field_config: Dict,
    key_prefix: str = "",
    sample_data: pd.DataFrame = None
) -> Dict[str, str]:
    """
    Render column mapping UI.
    
    Args:
        csv_columns: List of column names from CSV
        field_config: Field configuration from FIELD_CONFIGS
        key_prefix: Prefix for Streamlit keys
        sample_data: Optional sample data for preview
    
    Returns:
        Mapping dict: {db_field: csv_column}
    """
    st.markdown("### 🔗 Column Mapping")
    st.caption("Map your CSV columns to the required fields")
    
    mapping = {}
    fields = field_config.get('fields', {})
    
    # Group fields by required/optional
    required_fields = {k: v for k, v in fields.items() if v.get('required', False)}
    optional_fields = {k: v for k, v in fields.items() if not v.get('required', False)}
    
    # Show required fields first
    if required_fields:
        st.markdown("#### Required Fields")
        for field_key, field_def in required_fields.items():
            label = field_def.get('label', field_key)
            hints = field_def.get('hints', [])
            
            # Find best match
            best_match = None
            best_score = 0
            
            for csv_col in csv_columns:
                csv_col_lower = str(csv_col).lower()
                field_lower = field_key.lower()
                
                # Exact match
                if csv_col_lower == field_lower:
                    best_match = csv_col
                    best_score = 100
                    break
                
                # Check hints
                for hint in hints:
                    if hint.lower() in csv_col_lower or csv_col_lower in hint.lower():
                        score = len(hint) / len(csv_col)
                        if score > best_score:
                            best_match = csv_col
                            best_score = score
                
                # Partial match
                if field_lower in csv_col_lower or csv_col_lower in field_lower:
                    score = len(field_lower) / max(len(csv_col_lower), len(field_lower))
                    if score > best_score:
                        best_match = csv_col
                        best_score = score
            
            mapping[field_key] = st.selectbox(
                f"{label} *",
                options=[''] + csv_columns,
                index=1 + csv_columns.index(best_match) if best_match and best_match in csv_columns else 0,
                key=f"{key_prefix}_map_{field_key}",
                help=field_def.get('description', '')
            )
    
    # Show optional fields
    if optional_fields:
        st.markdown("#### Optional Fields")
        for field_key, field_def in optional_fields.items():
            label = field_def.get('label', field_key)
            hints = field_def.get('hints', [])
            
            # Find best match
            best_match = None
            for csv_col in csv_columns:
                csv_col_lower = str(csv_col).lower()
                if field_key.lower() in csv_col_lower or csv_col_lower in field_key.lower():
                    best_match = csv_col
                    break
                for hint in hints:
                    if hint.lower() in csv_col_lower:
                        best_match = csv_col
                        break
            
            mapping[field_key] = st.selectbox(
                label,
                options=[''] + csv_columns,
                index=1 + csv_columns.index(best_match) if best_match and best_match in csv_columns else 0,
                key=f"{key_prefix}_map_{field_key}",
                help=field_def.get('description', '')
            )
    
    return mapping


def validate_mapping(mapping: Dict, field_config: Dict) -> Tuple[bool, List[str]]:
    """Validate that all required fields are mapped."""
    errors = []
    fields = field_config.get('fields', {})
    
    for field_key, field_def in fields.items():
        if field_def.get('required', False):
            if not mapping.get(field_key):
                errors.append(f"Required field '{field_def.get('label', field_key)}' is not mapped")
    
    return len(errors) == 0, errors


def apply_mapping(df: pd.DataFrame, mapping: Dict) -> pd.DataFrame:
    """Apply column mapping to standardize DataFrame."""
    result = pd.DataFrame()
    for db_field, csv_col in mapping.items():
        if csv_col and csv_col in df.columns:
            result[db_field] = df[csv_col]
    return result


# =============================================================================
# DATA TRANSFORMATION HELPERS
# =============================================================================

def parse_date(value, default=None) -> Optional[str]:
    """Parse a date value to ISO format string."""
    if pd.isna(value):
        return default or datetime.now().strftime('%Y-%m-%d')
    try:
        return pd.to_datetime(value).strftime('%Y-%m-%d')
    except:
        return default or datetime.now().strftime('%Y-%m-%d')


def parse_number(value, default=0) -> float:
    """Parse a numeric value, handling currency formatting and accounting parentheses."""
    if pd.isna(value):
        return default
    try:
        # Convert to string and clean
        cleaned = str(value).strip()

        # Strip common quoting
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()
        cleaned = cleaned.strip()

        # Handle placeholders
        if cleaned in ("", "-", "–", "—", "N/A", "na", "NaN", "nan"):
            return default
        
        # Handle accounting format: (12345) means -12345
        # Some files include spaces inside parentheses: "( 123 )"
        cleaned_no_space = cleaned.replace(" ", "")
        is_negative = cleaned_no_space.startswith('(') and cleaned_no_space.endswith(')')
        if is_negative:
            cleaned = cleaned_no_space[1:-1]  # Remove parentheses
        else:
            cleaned = cleaned_no_space
        
        # Remove common currency formatting
        cleaned = cleaned.replace(',', '').replace('R', '').replace('$', '').replace('€', '').replace('£', '')
        
        # After cleanup, still empty/placeholder
        if cleaned in ("", "-", "–", "—"):
            return default

        result = float(cleaned)
        
        # Apply negative if it was in parentheses
        if is_negative:
            result = -result
        
        return result
    except:
        return default


def parse_int(value, default=0) -> int:
    """Parse an integer value."""
    if pd.isna(value):
        return default
    try:
        return int(float(str(value).replace(',', '')))
    except:
        return default


def clean_string(value, max_length=None) -> Optional[str]:
    """Clean and optionally truncate a string value."""
    if pd.isna(value):
        return None
    result = str(value).strip()
    if max_length and len(result) > max_length:
        result = result[:max_length]
    return result if result else None


# =============================================================================
# GENERIC IMPORT FUNCTION
# =============================================================================

def process_import(
    db,
    user_id: str,
    df: pd.DataFrame,
    import_type: str,
    scenario_id: str = None,
    clear_existing: bool = True,
    chunk_size: int = 1000
) -> Dict[str, int]:
    """
    Process an import using the mapped DataFrame.
    
    Args:
        db: Database handler
        user_id: User ID
        df: Mapped DataFrame with standardized columns
        import_type: Key from FIELD_CONFIGS
        scenario_id: Scenario ID (required for some imports)
        clear_existing: Whether to clear existing data first
    
    Returns:
        Stats dict with success, failed, skipped counts
    """
    config = FIELD_CONFIGS.get(import_type)
    if not config:
        st.error(f"Unknown import type: {import_type}")
        return {'success': 0, 'failed': 0, 'skipped': 0}
    
    table = config['table']
    
    # Check scenario requirement
    if config['requires_scenario'] and not scenario_id:
        st.error("Scenario ID is required for this import")
        return {'success': 0, 'failed': 0, 'skipped': 0}
    
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    if df.empty:
        return stats
    
    # Clear existing data if requested
    if clear_existing:
        try:
            # Tables that only use scenario_id (no user_id column)
            scenario_only_tables = [
                'historic_financials', 'historic_customer_revenue',
                'historical_balance_sheet', 'historical_cashflow', 'historical_trial_balance',
                'historical_income_statement_line_items', 'historical_balance_sheet_line_items',
                'historical_cashflow_line_items'
            ]
            
            # Check if table exists first (for new tables that might not be migrated yet)
            try:
                # Try to query the table to see if it exists
                test_query = db.client.table(table).select('id').limit(1)
                if table in scenario_only_tables and scenario_id:
                    test_query = test_query.eq('scenario_id', scenario_id)
                test_query.execute()
            except Exception as table_check_error:
                # Table doesn't exist - likely migration not run
                error_msg = str(table_check_error)
                if 'PGRST205' in error_msg or 'not found' in error_msg.lower() or 'schema cache' in error_msg.lower():
                    st.error(f"""
                    ❌ **Table Not Found: {table}**
                    
                    The database table does not exist. Please run the migration first:
                    
                    ```bash
                    ./run_migration_psql.sh migrations_add_detailed_line_items.sql
                    ```
                    
                    This will create the required tables for detailed line item imports.
                    """)
                    return {'success': 0, 'failed': 0, 'skipped': 0}
                else:
                    # Other error, continue with delete attempt
                    pass
            
            query = db.client.table(table).delete()
            
            if table in scenario_only_tables:
                # These tables only have scenario_id, not user_id
                if scenario_id:
                    query = query.eq('scenario_id', scenario_id)
            else:
                # Standard tables have user_id
                query = query.eq('user_id', user_id)
                if scenario_id and config['requires_scenario']:
                    query = query.eq('scenario_id', scenario_id)
            
            query.execute()
        except Exception as e:
            error_msg = str(e)
            # Check if it's a "table not found" error
            if 'PGRST205' in error_msg or 'not found' in error_msg.lower() or 'schema cache' in error_msg.lower():
                st.error(f"""
                ❌ **Table Not Found: {table}**
                
                The database table does not exist. Please run the migration first:
                
                ```bash
                ./run_migration_psql.sh migrations_add_detailed_line_items.sql
                ```
                
                This will create the required tables for detailed line item imports.
                """)
                return {'success': 0, 'failed': 0, 'skipped': 0}
            else:
                st.warning(f"Could not clear existing data: {e}")
    
    # -------------------------------------------------------------------------
    # Fast path: build records, then chunked batch upsert/insert
    # -------------------------------------------------------------------------
    progress = st.progress(0)
    status = st.empty()
    error_log = []  # Collect errors for display (first few)

    # Determine conflict columns for tables that support it
    line_item_tables = [
        'historical_income_statement_line_items',
        'historical_balance_sheet_line_items',
        'historical_cashflow_line_items'
    ]
    on_conflict: Optional[str] = None
    if table in line_item_tables:
        on_conflict = 'scenario_id,period_date,line_item_name'
    elif table == 'expense_assumptions':
        on_conflict = 'scenario_id,expense_code'
    else:
        # If the incoming data includes a month field, use scenario_id,month
        if 'month' in df.columns:
            on_conflict = 'scenario_id,month'

    # Build all records first (no DB calls yet)
    total_rows = len(df)
    built_records: List[Tuple[int, Dict[str, Any]]] = []
    for i, (_idx, row) in enumerate(df.iterrows(), start=1):
        if i == 1 or i % 200 == 0 or i == total_rows:
            progress.progress(min(i / max(total_rows, 1), 1.0))
            status.text(f"Staging {i:,} of {total_rows:,}...")

        record = build_record(row, import_type, user_id, scenario_id, config)
        if record is None:
            stats['skipped'] += 1
            continue
        built_records.append((i, record))

    # Chunked write
    total_to_write = len(built_records)
    written = 0
    if total_to_write == 0:
        progress.empty()
        status.empty()
        return stats

    def _write_chunk(records_chunk: List[Tuple[int, Dict[str, Any]]]) -> None:
        """Write a chunk; raise on failure."""
        payload = [r for (_row_num, r) in records_chunk]
        if on_conflict:
            db.client.table(table).upsert(payload, on_conflict=on_conflict).execute()
        else:
            # Default: insert (fastest). If duplicates occur, fallback is handled outside.
            db.client.table(table).insert(payload).execute()

    def _write_chunk_with_backoff(records_chunk: List[Tuple[int, Dict[str, Any]]]) -> None:
        """
        Write a chunk, splitting automatically on request-size/time issues.

        Supabase/PostgREST can reject large payloads (413) or time out depending on network.
        We keep default chunk_size=1000 but automatically split a failing chunk to isolate/fit limits.
        """
        try:
            _write_chunk(records_chunk)
            return
        except Exception as e:
            msg = str(e).lower()
            # Heuristics for payload-size / request issues
            maybe_too_large = any(s in msg for s in ["413", "payload", "request entity too large", "too large", "timeout"])
            if maybe_too_large and len(records_chunk) > 1:
                mid = len(records_chunk) // 2
                _write_chunk_with_backoff(records_chunk[:mid])
                _write_chunk_with_backoff(records_chunk[mid:])
                return
            raise

    # Iterate chunks
    for start in range(0, total_to_write, max(int(chunk_size), 1)):
        end = min(start + max(int(chunk_size), 1), total_to_write)
        chunk = built_records[start:end]

        status.text(f"Uploading {written + 1:,}–{end:,} of {total_to_write:,}...")
        progress.progress(min(end / max(total_to_write, 1), 1.0))

        try:
            _write_chunk_with_backoff(chunk)
            stats['success'] += len(chunk)
            written += len(chunk)
            continue
        except Exception as chunk_err:
            # If insert failed due to duplicates, try upsert without conflict columns (PK-based) as fallback
            err_msg = str(chunk_err)
            if (not on_conflict) and ('duplicate' in err_msg.lower() or '23505' in err_msg):
                try:
                    db.client.table(table).upsert([r for (_n, r) in chunk]).execute()
                    stats['success'] += len(chunk)
                    written += len(chunk)
                    continue
                except Exception as chunk_err2:
                    chunk_err = chunk_err2

            # Slow fallback: attempt row-by-row within this chunk to isolate failing rows
            for row_num, record in chunk:
                try:
                    if on_conflict:
                        db.client.table(table).upsert(record, on_conflict=on_conflict).execute()
                    else:
                        try:
                            db.client.table(table).insert(record).execute()
                        except Exception as insert_err:
                            if 'duplicate' in str(insert_err).lower() or '23505' in str(insert_err):
                                db.client.table(table).upsert(record).execute()
                            else:
                                raise
                    stats['success'] += 1
                except Exception as row_err:
                    stats['failed'] += 1
                    if len(error_log) < 5:
                        error_log.append({
                            'row': row_num,
                            'record': record,
                            'error': str(row_err)
                        })
    
    progress.empty()
    status.empty()
    
    # Display errors if any - make them persistent and visible
    if error_log:
        st.markdown("---")
        st.error(f"❌ **Import Failed: {stats['failed']} rows failed** (showing first {len(error_log)} errors)")
        
        # Store errors in session state so they persist
        error_key = f"import_errors_{import_type}_{scenario_id}"
        st.session_state[error_key] = error_log
        
        # Display errors in a prominent, persistent way
        for i, err in enumerate(error_log):
            with st.container():
                st.markdown(f"### Error {i+1}: Row {err['row']}")
                st.error(f"**Error Message:**\n```\n{err['error']}\n```")
                
                # Show record data
                st.markdown("**Record Data:**")
                st.json(err['record'])
                
                # Show the problematic values
                st.markdown("**Key Values:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Period Date:** {err['record'].get('period_date', 'N/A')}")
                with col2:
                    st.write(f"**Line Item:** {err['record'].get('line_item_name', 'N/A')}")
                with col3:
                    st.write(f"**Amount:** {err['record'].get('amount', 'N/A')}")
                
                st.markdown("---")
        
        if stats['failed'] > len(error_log):
            st.info(f"⚠️ {stats['failed'] - len(error_log)} more errors occurred. Check the error pattern above.")
        
        # Don't rerun automatically - let user see the errors
        st.stop()
    
    return stats


def build_record(
    row: pd.Series,
    import_type: str,
    user_id: str,
    scenario_id: str,
    config: Dict
) -> Optional[Dict]:
    """
    Build a database record from a DataFrame row.
    Returns None if row should be skipped.
    """
    
    # Import-specific builders
    builders = {
        'aged_debtors': build_aged_debtor_record,
        'aged_creditors': build_aged_creditor_record,
        'suppliers': build_supplier_record,
        'customers': build_customer_record,
        'installed_base': build_installed_base_record,
        'wear_profiles': build_wear_profile_record,
        'prospects': build_prospect_record,
        'historic_financials': build_historic_financials_record,
        'historic_customer_revenue': build_historic_customer_revenue_record,
        'historic_balance_sheet': build_historic_balance_sheet_record,
        'historic_cashflow': build_historic_cashflow_record,
        'historic_trial_balance': build_historic_trial_balance_record,
        'historical_income_statement_line_items': build_historical_line_item_record,
        'historical_balance_sheet_line_items': build_historical_line_item_record,
        'historical_cashflow_line_items': build_historical_line_item_record,
    }
    
    builder = builders.get(import_type)
    if builder:
        # Pass import_type for historical line items so they can set statement_type correctly
        if import_type in ['historical_income_statement_line_items', 
                          'historical_balance_sheet_line_items', 
                          'historical_cashflow_line_items']:
            return builder(row, user_id, scenario_id, import_type)
        else:
            return builder(row, user_id, scenario_id)
    
    return None


# =============================================================================
# RECORD BUILDERS FOR EACH IMPORT TYPE
# =============================================================================

def build_aged_debtor_record(row: pd.Series, user_id: str, scenario_id: str) -> Optional[Dict]:
    """Build aged_debtors record."""
    customer_name = clean_string(row.get('customer_name'))
    if not customer_name:
        return None
    
    amount_due = parse_number(row.get('amount_due'))
    if amount_due <= 0:
        return None
    
    invoice_date = parse_date(row.get('invoice_date'))
    due_date = parse_date(row.get('due_date'))
    
    if not due_date:
        terms = parse_int(row.get('payment_terms'), 30)
        due_date = (pd.to_datetime(invoice_date) + pd.Timedelta(days=terms)).strftime('%Y-%m-%d')
    
    payment_terms = parse_int(row.get('payment_terms'))
    if not payment_terms:
        try:
            payment_terms = (pd.to_datetime(due_date) - pd.to_datetime(invoice_date)).days
        except:
            payment_terms = 30
    
    return {
        'user_id': user_id,
        'customer_name': customer_name,
        'invoice_date': invoice_date,
        'due_date': due_date,
        'amount_due': amount_due,
        'payment_terms': payment_terms,
        'invoice_ref': clean_string(row.get('invoice_ref'), 50)
    }


def build_aged_creditor_record(row: pd.Series, user_id: str, scenario_id: str) -> Optional[Dict]:
    """Build aged_creditors record."""
    supplier_name = clean_string(row.get('supplier_name'))
    if not supplier_name:
        return None
    
    amount_due = parse_number(row.get('amount_due'))
    if amount_due <= 0:
        return None
    
    invoice_date = parse_date(row.get('invoice_date'))
    due_date = parse_date(row.get('due_date'))
    
    if not due_date:
        due_date = (pd.to_datetime(invoice_date) + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    
    return {
        'user_id': user_id,
        'supplier_name': supplier_name,
        'invoice_date': invoice_date,
        'due_date': due_date,
        'amount_due': amount_due,
        'currency': clean_string(row.get('currency'), 3) or 'ZAR',
        'invoice_ref': clean_string(row.get('invoice_ref'), 50)
    }


def build_supplier_record(row: pd.Series, user_id: str, scenario_id: str) -> Optional[Dict]:
    """Build creditors (suppliers) record."""
    name = clean_string(row.get('name'))
    if not name:
        return None
    
    cred_type = clean_string(row.get('creditor_type')) or 'domestic'
    
    return {
        'scenario_id': scenario_id,
        'user_id': user_id,
        'name': name,
        'creditor_type': cred_type,
        'standard_payment_days': parse_int(
            row.get('standard_payment_days') if row.get('standard_payment_days') is not None else row.get('payment_terms'),
            30
        ),
    }


def build_customer_record(row: pd.Series, user_id: str, scenario_id: str) -> Optional[Dict]:
    """Build customers record."""
    customer_code = clean_string(row.get('customer_code'))
    customer_name = clean_string(row.get('customer_name'))
    
    if not customer_code or not customer_name:
        return None
    
    return {
        'user_id': user_id,
        'customer_code': customer_code,
        'customer_name': customer_name
    }


def build_installed_base_record(row: pd.Series, user_id: str, scenario_id: str) -> Optional[Dict]:
    """Build installed_base record."""
    machine_id = clean_string(row.get('machine_id'))
    customer_name = clean_string(row.get('customer_name'))
    machine_model = clean_string(row.get('machine_model'))
    commission_date = parse_date(row.get('commission_date'))
    
    if not machine_id or not customer_name or not machine_model or not commission_date:
        return None
    
    return {
        'scenario_id': scenario_id,
        'user_id': user_id,
        'machine_id': machine_id,
        'customer_name': customer_name,
        'site_name': clean_string(row.get('site_name')),
        'machine_model': machine_model,
        'commission_date': commission_date,
        'status': 'Active'
    }


def build_wear_profile_record(row: pd.Series, user_id: str, scenario_id: str) -> Optional[Dict]:
    """Build wear_profiles record."""
    machine_model = clean_string(row.get('machine_model'))
    if not machine_model:
        return None
    
    return {
        'user_id': user_id,
        'machine_model': machine_model,
        'liner_life_months': parse_int(row.get('liner_life_months'), 12),
        'refurb_interval_months': parse_int(row.get('refurb_interval_months'), 36),
        'avg_consumable_revenue': parse_number(row.get('avg_consumable_revenue'), 0),
        'avg_refurb_revenue': parse_number(row.get('avg_refurb_revenue'), 0),
        'gross_margin_liner': parse_number(row.get('gross_margin_liner'), 0.35),
        'gross_margin_refurb': parse_number(row.get('gross_margin_refurb'), 0.28)
    }


def build_prospect_record(row: pd.Series, user_id: str, scenario_id: str) -> Optional[Dict]:
    """Build prospects record."""
    prospect_name = clean_string(row.get('prospect_name'))
    machine_model_name = clean_string(row.get('machine_model'))
    expected_close_date = parse_date(row.get('expected_close_date'))
    
    if not prospect_name or not machine_model_name or not expected_close_date:
        return None
    
    return {
        'scenario_id': scenario_id,
        'user_id': user_id,
        'prospect_name': prospect_name,
        'site_name': clean_string(row.get('site_name')),
        'machine_model_name': machine_model_name,
        'expected_close_date': expected_close_date,
        'confidence_pct': parse_number(row.get('confidence_pct'), 50) / 100 if parse_number(row.get('confidence_pct'), 0) > 1 else parse_number(row.get('confidence_pct'), 0.5),
        'machine_count': parse_int(row.get('machine_count'), 1),
        'annual_liner_value': parse_number(row.get('annual_liner_value'), 0),
        'refurb_value': parse_number(row.get('refurb_value'), 0)
    }


def build_historic_financials_record(row: pd.Series, user_id: str, scenario_id: str) -> Optional[Dict]:
    """Build historic_financials record."""
    month = parse_date(row.get('month'))
    revenue = parse_number(row.get('revenue'))
    
    if not month or revenue == 0:
        return None
    
    # Ensure month is first of month
    month_dt = pd.to_datetime(month)
    month = month_dt.replace(day=1).strftime('%Y-%m-%d')
    
    return {
        'scenario_id': scenario_id,
        'month': month,
        'revenue': revenue,
        'cogs': parse_number(row.get('cogs')),
        'gross_profit': parse_number(row.get('gross_profit')),
        'opex': parse_number(row.get('opex')),
        'ebit': parse_number(row.get('ebit'))
    }


def build_historic_customer_revenue_record(row: pd.Series, user_id: str, scenario_id: str) -> Optional[Dict]:
    """Build historic_customer_revenue record."""
    month = parse_date(row.get('month'))
    customer_code = clean_string(row.get('customer_code'))
    revenue = parse_number(row.get('revenue'))
    
    if not month or not customer_code or revenue == 0:
        return None
    
    month_dt = pd.to_datetime(month)
    month = month_dt.replace(day=1).strftime('%Y-%m-%d')
    
    return {
        'scenario_id': scenario_id,
        'month': month,
        'customer_code': customer_code,
        'customer_name': clean_string(row.get('customer_name')),
        'revenue': revenue
    }


def build_historic_balance_sheet_record(row: pd.Series, user_id: str, scenario_id: str) -> Optional[Dict]:
    """Build historic_balance_sheet record."""
    month = parse_date(row.get('month'))
    if not month:
        return None
    
    month_dt = pd.to_datetime(month)
    month = month_dt.strftime('%Y-%m-%d')
    
    return {
        'scenario_id': scenario_id,
        'month': month,
        'cash_and_equivalents': parse_number(row.get('cash_and_equivalents')),
        'accounts_receivable': parse_number(row.get('accounts_receivable')),
        'inventory': parse_number(row.get('inventory')),
        'prepaid_expenses': parse_number(row.get('prepaid_expenses')),
        'total_current_assets': parse_number(row.get('total_current_assets')),
        'ppe_net': parse_number(row.get('ppe_net')),
        'intangible_assets': parse_number(row.get('intangible_assets')),
        'total_noncurrent_assets': parse_number(row.get('total_noncurrent_assets')),
        'total_assets': parse_number(row.get('total_assets')),
        'accounts_payable': parse_number(row.get('accounts_payable')),
        'accrued_expenses': parse_number(row.get('accrued_expenses')),
        'short_term_debt': parse_number(row.get('short_term_debt')),
        'total_current_liabilities': parse_number(row.get('total_current_liabilities')),
        'long_term_debt': parse_number(row.get('long_term_debt')),
        'total_noncurrent_liabilities': parse_number(row.get('total_noncurrent_liabilities')),
        'total_liabilities': parse_number(row.get('total_liabilities')),
        'share_capital': parse_number(row.get('share_capital')),
        'retained_earnings': parse_number(row.get('retained_earnings')),
        'total_equity': parse_number(row.get('total_equity'))
    }


def build_historic_cashflow_record(row: pd.Series, user_id: str, scenario_id: str) -> Optional[Dict]:
    """Build historic_cashflow record."""
    month = parse_date(row.get('month'))
    if not month:
        return None
    
    month_dt = pd.to_datetime(month)
    month = month_dt.strftime('%Y-%m-%d')
    
    return {
        'scenario_id': scenario_id,
        'month': month,
        'net_income': parse_number(row.get('net_income')),
        'depreciation_amortization': parse_number(row.get('depreciation_amortization')),
        'change_in_receivables': parse_number(row.get('change_in_receivables')),
        'change_in_inventory': parse_number(row.get('change_in_inventory')),
        'change_in_payables': parse_number(row.get('change_in_payables')),
        'change_in_accruals': parse_number(row.get('change_in_accruals')),
        'cash_from_operations': parse_number(row.get('cash_from_operations')),
        'capital_expenditure': parse_number(row.get('capital_expenditure')),
        'asset_disposals': parse_number(row.get('asset_disposals')),
        'cash_from_investing': parse_number(row.get('cash_from_investing')),
        'debt_proceeds': parse_number(row.get('debt_proceeds')),
        'debt_repayment': parse_number(row.get('debt_repayment')),
        'dividends_paid': parse_number(row.get('dividends_paid')),
        'cash_from_financing': parse_number(row.get('cash_from_financing')),
        'net_change_in_cash': parse_number(row.get('net_change_in_cash'))
    }


def build_historic_trial_balance_record(row: pd.Series, user_id: str, scenario_id: str) -> Optional[Dict]:
    """Build historic_trial_balance record."""
    month = parse_date(row.get('month'))
    account_code = clean_string(row.get('account_code'))
    account_name = clean_string(row.get('account_name'))
    
    if not month or not account_code or not account_name:
        return None
    
    # Ensure month is first of month
    month_dt = pd.to_datetime(month)
    month = month_dt.replace(day=1).strftime('%Y-%m-%d')
    
    debit = parse_number(row.get('debit')) or 0
    credit = parse_number(row.get('credit')) or 0
    balance = parse_number(row.get('balance'))
    
    # If balance not provided, calculate from debit/credit
    if balance is None:
        balance = debit - credit
    
    return {
        'scenario_id': scenario_id,
        'month': month,
        'account_code': account_code,
        'account_name': account_name,
        'account_type': clean_string(row.get('account_type')),
        'account_category': clean_string(row.get('account_category')),
        'debit': debit,
        'credit': credit,
        'balance': balance
    }


def build_historical_line_item_record(row: pd.Series, user_id: str, scenario_id: str, import_type: str = None) -> Optional[Dict]:
    """
    Build historical line item record for income statement, balance sheet, or cash flow.
    Works for all three detailed line item tables.
    """
    # Validate required inputs
    if not scenario_id:
        return None
    if not user_id:
        return None
    
    period_date = parse_date(row.get('period_date'))
    line_item_name = clean_string(row.get('line_item_name'))
    category = clean_string(row.get('category'))
    amount = parse_number(row.get('amount'))
    
    if not period_date or not line_item_name or not category:
        return None
    
    # Ensure period_date is in correct format (normalize to month-start for monthly models)
    try:
        if isinstance(period_date, str):
            period_date_dt = pd.to_datetime(period_date)
        else:
            period_date_dt = period_date

        # Normalize to first day of month for all statement line-item imports
        period_date_dt = period_date_dt.replace(day=1)
        period_date_str = period_date_dt.strftime('%Y-%m-%d')
    except Exception:
        return None
    
    # Get sub_category if available
    sub_category = clean_string(row.get('sub_category'))
    
    # Determine statement_type from import_type
    statement_type = 'income_statement'  # default
    if import_type:
        if 'balance_sheet' in import_type:
            statement_type = 'balance_sheet'
        elif 'cashflow' in import_type or 'cash_flow' in import_type:
            statement_type = 'cash_flow'
        elif 'income_statement' in import_type:
            statement_type = 'income_statement'
    
    return {
        'scenario_id': scenario_id,
        'user_id': user_id,
        'period_date': period_date_str,
        'line_item_name': line_item_name,
        'category': category,
        'sub_category': sub_category if sub_category else None,
        'amount': float(amount) if amount is not None else 0.0,
        'statement_type': statement_type
    }


# =============================================================================
# HIGH-LEVEL IMPORT UI COMPONENT
# =============================================================================

def render_import_with_mapping(
    db,
    user_id: str,
    import_type: str,
    scenario_id: str = None,
    on_success: Callable = None
):
    """
    Render a complete import UI with column mapping.
    
    NEW: Automatically detects and converts WIDE format (transposed) to LONG format
    for detailed line item imports.
    
    Args:
        db: Database handler
        user_id: User ID
        import_type: Key from FIELD_CONFIGS
        scenario_id: Scenario ID (required for some imports)
        on_success: Optional callback after successful import
    
    Usage:
        render_import_with_mapping(db, user_id, 'aged_debtors')
    """
    config = FIELD_CONFIGS.get(import_type)
    if not config:
        st.error(f"Unknown import type: {import_type}")
        return
    
    st.markdown(f"### {config['icon']} Import {config['display_name']}")
    st.caption(config['description'])
    
    # Check scenario requirement
    if config['requires_scenario'] and not scenario_id:
        st.warning("⚠️ Please select a scenario first")
        return
    
    # Template download
    template = CSV_TEMPLATES.get(import_type)
    if template:
        st.download_button(
            "📥 Download Template",
            data=template,
            file_name=f"{import_type}_template.csv",
            mime="text/csv"
        )
    
    # File upload
    uploaded = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        key=f"upload_{import_type}"
    )
    
    if not uploaded:
        return
    
    try:
        raw_df = pd.read_csv(uploaded)
        # Normalize column names (CE Imports files often include whitespace like " amount ")
        raw_df.columns = [str(c).strip() for c in raw_df.columns]
        # Drop blank/unnamed columns created by trailing commas
        drop_cols = [c for c in raw_df.columns if c == "" or c.lower().startswith("unnamed")]
        if drop_cols:
            raw_df = raw_df.drop(columns=drop_cols, errors='ignore')
        st.success(f"✅ Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns")
        
        # NEW: Detect and handle wide format (transposed) for detailed line items
        is_wide_format = False
        if import_type in ['historical_income_statement_line_items', 
                          'historical_balance_sheet_line_items', 
                          'historical_cashflow_line_items']:
                # Check if this looks like wide format (line items as rows, periods as columns)
                # Wide format typically has: Line Item, Category, FY2024, FY2023_Restated, etc.
                has_line_item_col = any('line' in str(col).lower() and 'item' in str(col).lower() 
                                       for col in raw_df.columns)
                has_category_col = any('category' in str(col).lower() and 'sub' not in str(col).lower() 
                                      for col in raw_df.columns)
                
                # Check for period columns (FY2024, FY2023_Restated, YTD_Oct2025, etc.)
                period_cols_found = []
                for col in raw_df.columns:
                    col_lower = str(col).lower()
                    # Skip metadata columns
                    if 'line' in col_lower and 'item' in col_lower:
                        continue
                    if 'category' in col_lower:
                        continue
                    if 'change' in col_lower or 'pct' in col_lower or '%' in col_lower or 'yoy' in col_lower:
                        continue
                    if 'budget' in col_lower or 'variance' in col_lower or 'var' in col_lower:
                        continue
                    if 'notes' in col_lower or 'py' in col_lower:
                        continue
                    # Check if it's a period column
                    if extract_period_from_column(col):
                        period_cols_found.append(col)
                
                has_period_cols = len(period_cols_found) > 0
                
                if has_line_item_col and has_period_cols:
                    is_wide_format = True
                    st.info("""
                    📊 **Wide Format Detected**
                    
                    Your file appears to be in "wide" format (line items as rows, periods as columns).
                    The system will automatically convert this to the required "long" format during import.
                    """)
                    
                    # Show format detection
                    with st.expander("🔍 Format Detection Details", expanded=False):
                        st.write("**Detected Format:** Wide (transposed)")
                        line_item_cols = [col for col in raw_df.columns if 'line' in str(col).lower() and 'item' in str(col).lower()]
                        category_cols = [col for col in raw_df.columns if 'category' in str(col).lower() and 'sub' not in str(col).lower()]
                        st.write(f"- Line Item Column: {line_item_cols}")
                        st.write(f"- Category Column: {category_cols}")
                        st.write(f"- Period Columns Found: {period_cols_found}")
                        st.write(f"- Total Periods: {len(period_cols_found)}")
                        st.write("**Will Convert To:** Long format (one row per line item per period)")
        
        # Preview raw data
        with st.expander("👁️ Preview Raw Data", expanded=True):
            st.dataframe(raw_df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # If wide format, show transformation preview
        if is_wide_format:
            transformed_df = transform_wide_to_long(raw_df, import_type)
            if not transformed_df.empty:
                with st.expander("🔄 Transformed Data Preview (Long Format)", expanded=True):
                    st.dataframe(transformed_df.head(20), use_container_width=True)
                    st.caption(f"✅ Converted {len(raw_df)} rows × {len([c for c in raw_df.columns if c.upper().startswith('FY') or 'YTD' in c.upper()])} periods = {len(transformed_df)} records")
                st.markdown("---")
                # Use transformed data for mapping
                raw_df = transformed_df

        # ---------------------------------------------------------------------
        # NEW: Period selection + YTD controls for statement line-item imports
        # ---------------------------------------------------------------------
        import_settings = None
        if import_type in [
            'historical_income_statement_line_items',
            'historical_balance_sheet_line_items',
            'historical_cashflow_line_items',
        ] and 'period_date' in raw_df.columns:
            try:
                _tmp = raw_df.copy()
                _tmp['period_date'] = pd.to_datetime(_tmp['period_date'], errors='coerce')
                _tmp = _tmp.dropna(subset=['period_date'])
                # Normalize to month-start for selection consistency
                _tmp['period_date'] = _tmp['period_date'].dt.to_period('M').dt.to_timestamp()

                period_vals = sorted(_tmp['period_date'].dropna().unique().tolist())
                period_labels = [pd.to_datetime(p).strftime('%Y-%m') for p in period_vals]
                label_to_period = {lbl: period_vals[i] for i, lbl in enumerate(period_labels)}

                with st.expander("⚙️ Import Options (Periods + YTD)", expanded=False):
                    st.caption("Select which periods to import. Use YTD controls only if the latest period is a year-to-date cumulative total.")

                    fy_end_month = st.selectbox(
                        "Financial year-end month",
                        options=list(range(1, 13)),
                        index=1,  # default Feb (CE Africa typical FY end)
                        format_func=lambda m: datetime(2000, m, 1).strftime('%b'),
                        key=f"{import_type}_fy_end_month",
                        help="Used to group months into financial years (FY). Example: FY end Feb → FY2026 runs Mar-2025 to Feb-2026."
                    )

                    def _fy_end_year(dt: pd.Timestamp, year_end_month: int) -> int:
                        # FY end year is the year in which the FY ends.
                        return int(dt.year) if int(dt.month) <= int(year_end_month) else int(dt.year) + 1

                    # Show fiscal years available and allow selection by FY instead of raw months
                    fy_map = {}
                    for p in period_vals:
                        fy = _fy_end_year(pd.to_datetime(p), fy_end_month)
                        fy_map.setdefault(fy, []).append(pd.to_datetime(p))
                    fy_years = sorted(fy_map.keys())
                    fy_labels = [f"FY{y}" for y in fy_years]
                    fy_label_to_year = {f"FY{y}": y for y in fy_years}

                    selected_fy_labels = st.multiselect(
                        "Financial years to import",
                        options=fy_labels,
                        default=fy_labels,
                        key=f"{import_type}_fy_select",
                    )

                    # Convert FY selection back into concrete period list
                    selected_periods = []
                    for lbl in selected_fy_labels:
                        y = fy_label_to_year.get(lbl)
                        if y in fy_map:
                            selected_periods.extend(fy_map[y])
                    selected_periods = sorted(set(selected_periods))
                    selected_labels = [pd.to_datetime(p).strftime('%Y-%m') for p in selected_periods]

                    selected_labels = st.multiselect(
                        "Periods to import (optional override)",
                        options=period_labels,
                        default=selected_labels if selected_labels else period_labels,
                        key=f"{import_type}_period_select",
                        help="Defaults to all months in the selected financial years above."
                    )
                    if not selected_labels:
                        st.warning("No periods selected — import will be disabled.")

                    selected_periods = [label_to_period[lbl] for lbl in selected_labels]
                    selected_years = sorted({_fy_end_year(pd.to_datetime(p), fy_end_month) for p in selected_periods})
                    latest_period = max(selected_periods) if selected_periods else None
                    latest_year = _fy_end_year(pd.to_datetime(latest_period), fy_end_month) if latest_period is not None else None

                    period_mode = st.radio(
                        "Most recent selected period represents",
                        options=["Full year / normal month", "YTD (partial year cumulative)"],
                        horizontal=True,
                        key=f"{import_type}_ytd_mode",
                    )

                    ytd_cfg = None
                    if period_mode == "YTD (partial year cumulative)" and latest_year is not None:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            ytd_year = st.selectbox(
                                "YTD financial year",
                                options=selected_years,
                                index=selected_years.index(latest_year) if latest_year in selected_years else 0,
                                key=f"{import_type}_ytd_year",
                            )
                            ytd_end_month = st.selectbox(
                                "YTD ends at month (calendar month)",
                                options=list(range(1, 13)),
                                index=9,  # default Oct
                                format_func=lambda m: datetime(2000, m, 1).strftime('%b'),
                                key=f"{import_type}_ytd_end_month",
                                help="This is the calendar month in which the YTD statement is cut off (e.g., Oct)."
                            )
                        with col_b:
                            # Build default fill months as remaining FY months after YTD end
                            # Determine FY start/end dates for the selected FY
                            fy_end = pd.Timestamp(int(ytd_year), int(fy_end_month), 1)
                            fy_start = (fy_end - pd.DateOffset(months=11)).to_period('M').to_timestamp()
                            fy_months = pd.date_range(start=fy_start, periods=12, freq='MS')
                            # YTD end is a calendar month inside the FY range; find its index
                            ytd_end_ts = pd.Timestamp(int(ytd_year) if ytd_end_month <= fy_end_month else int(ytd_year) - 1, int(ytd_end_month), 1)
                            # If the chosen month isn't inside the FY, fall back to latest selected period
                            if ytd_end_ts not in set(fy_months):
                                ytd_end_ts = pd.to_datetime(latest_period).to_period('M').to_timestamp()
                            ytd_idx = int(list(fy_months).index(ytd_end_ts)) if ytd_end_ts in set(fy_months) else 11
                            default_fill = list(range(ytd_idx + 2, 13))  # FY month indices (1..12)
                            fill_months = st.multiselect(
                                "Months to forecast/fill (within FY, as FY-month index)",
                                options=list(range(1, 13)),
                                default=default_fill,
                                format_func=lambda i: f"FY month {i} ({fy_months[i-1].strftime('%b')})",
                                key=f"{import_type}_ytd_fill_months",
                            )
                            st.caption("These months will be treated as forecasted fill when annualizing YTD totals.")

                        ytd_cfg = {
                            "year": int(ytd_year),
                            "year_end_month": int(fy_end_month),
                            "ytd_end_calendar_month": int(ytd_end_month),
                            "fill_fy_months": [int(m) for m in fill_months],
                        }

                    import_settings = {
                        "import_type": import_type,
                        "selected_periods": selected_labels,
                        "fiscal_year_end_month": int(fy_end_month),
                        "ytd": ytd_cfg,
                    }

                # Apply period filter to raw_df before mapping/import
                if selected_labels:
                    raw_df = raw_df.copy()
                    raw_df['period_date'] = pd.to_datetime(raw_df['period_date'], errors='coerce')
                    raw_df = raw_df.dropna(subset=['period_date'])
                    raw_df['period_date'] = raw_df['period_date'].dt.to_period('M').dt.to_timestamp()
                    raw_df = raw_df[raw_df['period_date'].isin(selected_periods)]
                    # Keep period_date as string-like for mapping (apply_mapping handles it)
                    raw_df['period_date'] = raw_df['period_date'].dt.strftime('%Y-%m-%d')
            except Exception as _e:
                import_settings = None
        
        # Column mapping
        mapping = render_column_mapper(
            csv_columns=list(raw_df.columns),
            field_config=config,
            key_prefix=import_type,
            sample_data=raw_df
        )
        
        # Validation
        is_valid, errors = validate_mapping(mapping, config)
        
        st.markdown("---")
        
        if errors:
            st.error("**Map all required fields:**")
            for err in errors:
                st.warning(f"⚠️ {err}")
        
        # Mapped preview
        if mapping:
            with st.expander("👁️ Preview Mapped Data", expanded=True):
                mapped_df = apply_mapping(raw_df, mapping)
                st.dataframe(mapped_df.head(10), use_container_width=True)
        
        # Import button
        col1, col2 = st.columns([1, 3])
        with col1:
            import_btn = st.button(
                f"🚀 Import {config['display_name']}",
                type="primary",
                key=f"import_{import_type}_btn",
                disabled=not is_valid or (import_settings is not None and not import_settings.get("selected_periods"))
            )
        
        if import_btn and is_valid:
            mapped_df = apply_mapping(raw_df, mapping)

            # Persist import settings (period selection + YTD config) into assumptions for this scenario
            try:
                if import_settings and scenario_id and hasattr(db, "update_assumptions"):
                    existing = {}
                    if hasattr(db, "get_scenario_assumptions"):
                        try:
                            existing = db.get_scenario_assumptions(scenario_id, user_id) or {}
                        except Exception:
                            existing = {}
                    existing = existing or {}
                    existing.setdefault("import_period_settings", {})
                    existing["import_period_settings"][import_type] = import_settings
                    db.update_assumptions(scenario_id, user_id, existing)
            except Exception:
                # Non-blocking: import should still proceed
                pass
            
            with st.spinner("Importing..."):
                stats = process_import(
                    db=db,
                    user_id=user_id,
                    df=mapped_df,
                    import_type=import_type,
                    scenario_id=scenario_id,
                    clear_existing=True
                )
            
            # Only show success message if there were no failures
            if stats['failed'] == 0:
                if stats['success'] > 0:
                    st.success(f"✅ Imported {stats['success']} records")
                if stats['skipped'] > 0:
                    st.info(f"ℹ️ Skipped {stats['skipped']} invalid rows")

                if on_success:
                    on_success()

                # Only rerun on success - errors will stop() to keep them visible
                st.rerun()
            else:
                # Errors are already displayed by process_import (which calls st.stop())
                # Just show summary
                if stats['success'] > 0:
                    st.info(f"ℹ️ {stats['success']} records imported successfully")
                if stats['skipped'] > 0:
                    st.info(f"ℹ️ Skipped {stats['skipped']} invalid rows")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Column Mapper Test", layout="wide")
    st.title("Column Mapper Module")
    
    st.markdown("### Available Import Types")
    for key, config in FIELD_CONFIGS.items():
        st.markdown(f"- `{key}`: {config['icon']} {config['display_name']}")
    
    st.markdown("### Usage")
    st.code("""
from components.column_mapper import render_import_with_mapping

# Simple usage
render_import_with_mapping(db, user_id, 'aged_debtors')

# With scenario
render_import_with_mapping(db, user_id, 'installed_base', scenario_id)
""")
