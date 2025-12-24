#!/usr/bin/env python3
"""
Transform AFS Extract Files to Model Format
===========================================
Converts vertical format (line items as rows) to horizontal format (periods as rows)
for Income Statement, Balance Sheet, and Cash Flow statements.

Usage:
    python scripts/transform_afs_extract.py
"""

import pandas as pd
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Base directory
BASE_DIR = Path("/Users/gregmorris/Development Projects/CE Africa/CE Africa Files/AFS Extract")
OUTPUT_DIR = BASE_DIR / "transformed"

# Field mappings
INCOME_STATEMENT_MAP = {
    'Revenue': 'revenue',
    'Cost of Sales': 'cogs',
    'Cost of Goods Sold': 'cogs',  # Alternative name
    'Gross Profit': 'gross_profit',
    # Operating expenses will be summed
}

BALANCE_SHEET_MAP = {
    'Cash and Cash Equivalents': 'cash_and_equivalents',
    'Trade Receivables Net': 'accounts_receivable',
    'Total Inventories': 'inventory',
    'Total Current Assets': 'total_current_assets',
    'Property Plant and Equipment': 'ppe_net',
    'Total Intangible Assets': 'intangible_assets',
    'Total Non-Current Assets': 'total_noncurrent_assets',
    'Total Assets': 'total_assets',
    'Total Current Liabilities': 'total_current_liabilities',
    'Total Non-Current Liabilities': 'total_noncurrent_liabilities',
    'Total Liabilities': 'total_liabilities',
    'Stated Capital': 'share_capital',
    'Retained Income/(Accumulated Loss)': 'retained_earnings',
    'Total Equity': 'total_equity',
}

CASH_FLOW_MAP = {
    'Loss/Profit Before Taxation': 'net_income',
    'Depreciation and Amortisation': 'depreciation_amortization',
    'Increase in Trade and Other Receivables': 'change_in_receivables',
    'Increase in Inventories': 'change_in_inventory',
    'Increase in Trade and Other Payables': 'change_in_payables',
    'Net Cash From Operating Activities': 'cash_from_operations',
    'Net Cash From Investing Activities': 'cash_from_investing',
    'Net Cash From Financing Activities': 'cash_from_financing',
    'Total Cash Movement for the Year': 'net_change_in_cash',
}


def extract_period_from_filename(filename: str) -> Optional[str]:
    """Extract period date from filename.
    
    Examples:
        CE_Africa_Income_Statement_FY2024.csv -> 2024-12-01
        CE_Africa_Balance_Sheet_YTD_Oct2025.csv -> 2025-10-01
    """
    # Match FY2024, FY2023, etc.
    fy_match = re.search(r'FY(\d{4})', filename)
    if fy_match:
        year = int(fy_match.group(1))
        return f"{year}-12-01"  # Year-end
    
    # Match YTD_Oct2025, YTD_Jan2024, etc.
    ytd_match = re.search(r'YTD_(\w+)(\d{4})', filename)
    if ytd_match:
        month_name = ytd_match.group(1)
        year = int(ytd_match.group(2))
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        month = month_map.get(month_name, 12)
        return f"{year}-{month:02d}-01"
    
    return None


def transform_income_statement(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Transform income statement from vertical to horizontal format."""
    result = {'month': period}
    
    # Find period column (case-insensitive, handle variations)
    period_cols = []
    for c in df.columns:
        c_lower = c.upper()
        if (c_lower.startswith('FY') or 'YTD' in c_lower or 'FY202' in c_lower or 'FY202' in c_lower) and \
           c != 'Line Item' and 'Category' not in c and 'Notes' not in c:
            period_cols.append(c)
    
    if not period_cols:
        # Try to find any numeric column (excluding Line Item, Category, Notes)
        for c in df.columns:
            if c not in ['Line Item', 'Category', 'Notes', 'Sub_Category']:
                try:
                    # Check if column has numeric values
                    numeric_vals = pd.to_numeric(df[c], errors='coerce').dropna()
                    if len(numeric_vals) > 0:
                        period_cols.append(c)
                except:
                    pass
    
    if not period_cols:
        print(f"  ⚠️  Could not find period column. Available: {', '.join(df.columns)}")
        return pd.DataFrame([result])
    
    period_col = period_cols[0]
    print(f"  ℹ️  Using period column: {period_col}")
    
    # Map direct fields
    for afs_name, model_name in INCOME_STATEMENT_MAP.items():
        row = None
        
        # Special handling for COGS - prioritize "Total Cost of Sales"
        if model_name == 'cogs':
            # First, try to find "Total Cost of Sales" (exact or partial)
            total_cogs_row = df[df['Line Item'].str.contains('Total Cost of Sales|Total.*Cost.*Sales', case=False, na=False, regex=True)]
            if not total_cogs_row.empty:
                # Check if the value is not empty (handle parentheses and strings)
                val = total_cogs_row[period_col].iloc[0]
                val_str = str(val).strip() if pd.notna(val) else ''
                # Value is valid if it's not empty, not just whitespace, and not just a dash or comma
                if val_str and val_str not in ['', '-', ',', 'nan', 'None']:
                    row = total_cogs_row
                    print(f"    ✓ Found Total Cost of Sales: {val_str}")
            else:
                # Try exact match
                row = df[df['Line Item'] == afs_name]
                if row.empty:
                    # Try partial match (case-insensitive)
                    row = df[df['Line Item'].str.contains(afs_name, case=False, na=False)]
                
                # If still empty or multiple matches, sum all individual COGS line items
                if row.empty or len(row) > 1:
                    cogs_rows = df[df['Line Item'].str.contains('Cost of Goods|Cost of Sales', case=False, na=False) &
                                  ~df['Line Item'].str.contains('Total|Margin|%|Gross', case=False, na=False)]
                    if len(cogs_rows) > 0:
                        # Sum all COGS line items
                        values = []
                        for idx, r in cogs_rows.iterrows():
                            val = r[period_col]
                            if pd.isna(val) or val == '' or val == 0:
                                continue
                            if isinstance(val, str):
                                val = val.strip()
                                if not val or val == '':
                                    continue
                                if val.startswith('(') and val.endswith(')'):
                                    val = abs(float(val.strip('()').replace(',', '')))
                                else:
                                    val = abs(float(val.replace(',', '')))
                            elif pd.notna(val):
                                val = abs(float(val))
                            else:
                                continue
                            values.append(val)
                        if values:
                            result[model_name] = sum(values)
                            continue
        else:
            # For other fields, try exact match first
            row = df[df['Line Item'] == afs_name]
            if row.empty:
                # Try partial match (case-insensitive)
                row = df[df['Line Item'].str.contains(afs_name, case=False, na=False)]
        
        if not row.empty:
            value = row[period_col].iloc[0]
            
            # Handle accounting format: parentheses = negative, strings with commas
            if isinstance(value, str):
                value = value.strip()
                if value.startswith('(') and value.endswith(')'):
                    value = -float(value.strip('()').replace(',', ''))
                else:
                    value = float(value.replace(',', ''))
            elif pd.notna(value):
                value = float(value)
            else:
                value = 0
            
            # Remove negative sign if present (COGS is negative in AFS)
            if model_name == 'cogs' and value < 0:
                value = abs(value)
            result[model_name] = value
        else:
            result[model_name] = 0
    
    # Sum all operating expenses
    category_col = None
    for col in df.columns:
        if 'category' in col.lower():
            category_col = col
            break
    
    if category_col:
        opex_rows = df[df[category_col].str.contains('Operating Expenses', case=False, na=False)]
    else:
        # Fallback: look for expense line items
        opex_rows = df[df['Line Item'].str.contains('Expense|Cost|Fee|Charge', case=False, na=False) & 
                      ~df['Line Item'].str.contains('Total|Gross|Revenue|Income', case=False, na=False)]
    
    if not opex_rows.empty:
        # Convert to numeric, handling parentheses and commas
        def parse_value(v):
            if pd.isna(v):
                return 0
            if isinstance(v, str):
                v = v.strip()
                if v.startswith('(') and v.endswith(')'):
                    return abs(float(v.strip('()').replace(',', '')))
                return abs(float(v.replace(',', '')))
            return abs(float(v))
        
        opex_values = opex_rows[period_col].apply(parse_value)
        opex_total = opex_values.sum()
        result['opex'] = float(opex_total) if pd.notna(opex_total) else 0
    else:
        result['opex'] = 0
    
    # Calculate EBIT
    gross_profit = result.get('gross_profit', 0)
    opex = result.get('opex', 0)
    result['ebit'] = gross_profit - opex
    
    return pd.DataFrame([result])


def transform_balance_sheet(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Transform balance sheet from vertical to horizontal format."""
    result = {'month': period}
    
    # Find period column (case-insensitive, handle variations)
    period_cols = []
    for c in df.columns:
        c_lower = c.upper()
        if (c_lower.startswith('FY') or 'YTD' in c_lower or 'FY202' in c_lower) and \
           c != 'Line Item' and 'Category' not in c and 'Notes' not in c and 'Sub_Category' not in c:
            period_cols.append(c)
    
    if not period_cols:
        # Try to find any numeric column
        for c in df.columns:
            if c not in ['Line Item', 'Category', 'Notes', 'Sub_Category', 'Change', 'Change_Pct', 'YoY_Change', 'YoY_Pct']:
                try:
                    numeric_vals = pd.to_numeric(df[c], errors='coerce').dropna()
                    if len(numeric_vals) > 0:
                        period_cols.append(c)
                except:
                    pass
    
    if not period_cols:
        print(f"  ⚠️  Could not find period column. Available: {', '.join(df.columns)}")
        return pd.DataFrame([result])
    
    period_col = period_cols[0]
    print(f"  ℹ️  Using period column: {period_col}")
    
    # Map direct fields (use contains for flexible matching)
    for afs_name, model_name in BALANCE_SHEET_MAP.items():
        # Try exact match first, then partial match
        row = df[df['Line Item'] == afs_name]
        if row.empty:
            # Try partial match (e.g., "Total Current Assets" matches "Current Assets")
            search_term = re.escape(afs_name.replace('Total ', '').replace('Net', '').strip())
            row = df[df['Line Item'].str.contains(search_term, case=False, na=False, regex=True)]
        
        if not row.empty:
            value = row[period_col].iloc[0]
            
            # Handle accounting format: parentheses = negative, strings with commas
            if isinstance(value, str):
                value = value.strip()
                if value.startswith('(') and value.endswith(')'):
                    value = -float(value.strip('()').replace(',', ''))
                else:
                    value = float(value.replace(',', ''))
            elif pd.notna(value):
                value = float(value)
            else:
                value = 0
            
            result[model_name] = value
        else:
            result[model_name] = 0
    
    # Handle accounts payable (may need to sum multiple line items)
    category_col = None
    for col in df.columns:
        if 'category' in col.lower():
            category_col = col
            break
    
    if category_col:
        payables_rows = df[
            (df[category_col].str.contains('Current Liabilities', case=False, na=False)) & 
            (df['Line Item'].str.contains('Payable|Creditor|Trade', case=False, na=False))
        ]
    else:
        payables_rows = df[df['Line Item'].str.contains('Payable|Creditor|Trade', case=False, na=False)]
    
    if not payables_rows.empty:
        def parse_value(v):
            if pd.isna(v):
                return 0
            if isinstance(v, str):
                v = v.strip()
                if v.startswith('(') and v.endswith(')'):
                    return abs(float(v.strip('()').replace(',', '')))
                return abs(float(v.replace(',', '')))
            return abs(float(v))
        
        ap_values = payables_rows[period_col].apply(parse_value)
        ap_total = ap_values.sum()
        result['accounts_payable'] = float(ap_total) if pd.notna(ap_total) else 0
    else:
        result['accounts_payable'] = 0
    
    # Handle short-term debt (lease liabilities current)
    lease_current = df[df['Line Item'].str.contains('Lease.*Current|Current.*Lease', case=False, na=False)]
    if not lease_current.empty:
        result['short_term_debt'] = float(lease_current[period_col].iloc[0]) if pd.notna(lease_current[period_col].iloc[0]) else 0
    else:
        result['short_term_debt'] = 0
    
    # Handle long-term debt (lease liabilities non-current)
    lease_noncurrent = df[df['Line Item'].str.contains('Lease.*Non.*Current|Non.*Current.*Lease', case=False, na=False)]
    if not lease_noncurrent.empty:
        result['long_term_debt'] = float(lease_noncurrent[period_col].iloc[0]) if pd.notna(lease_noncurrent[period_col].iloc[0]) else 0
    else:
        result['long_term_debt'] = 0
    
    return pd.DataFrame([result])


def transform_cash_flow(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Transform cash flow from vertical to horizontal format."""
    result = {'month': period}
    
    # Find period column (case-insensitive, handle variations)
    period_cols = []
    for c in df.columns:
        c_lower = c.upper()
        if (c_lower.startswith('FY') or 'YTD' in c_lower or 'FY202' in c_lower) and \
           c != 'Line Item' and 'Category' not in c and 'Notes' not in c and 'Sub_Category' not in c:
            period_cols.append(c)
    
    if not period_cols:
        # Try to find any numeric column
        for c in df.columns:
            if c not in ['Line Item', 'Category', 'Notes', 'Sub_Category', 'Change', 'Change_Pct', 'YoY_Change', 'YoY_Pct']:
                try:
                    numeric_vals = pd.to_numeric(df[c], errors='coerce').dropna()
                    if len(numeric_vals) > 0:
                        period_cols.append(c)
                except:
                    pass
    
    if not period_cols:
        print(f"  ⚠️  Could not find period column. Available: {', '.join(df.columns)}")
        return pd.DataFrame([result])
    
    period_col = period_cols[0]
    print(f"  ℹ️  Using period column: {period_col}")
    
    # Map direct fields (use flexible matching)
    for afs_name, model_name in CASH_FLOW_MAP.items():
        # Try exact match first
        row = df[df['Line Item'] == afs_name]
        if row.empty:
            # Try partial match (remove common words)
            search_term = afs_name.replace('Loss/', '').replace('Profit/', '').replace(' and ', ' ').strip()
            row = df[df['Line Item'].str.contains(search_term, case=False, na=False)]
        
        if not row.empty:
            value = row[period_col].iloc[0]
            
            # Handle accounting format: parentheses = negative
            if isinstance(value, str):
                value = value.strip()
                if value.startswith('(') and value.endswith(')'):
                    # Remove parentheses and make negative
                    value = -float(value.strip('()').replace(',', ''))
                else:
                    # Remove commas and convert
                    value = float(value.replace(',', ''))
            elif pd.notna(value):
                value = float(value)
            else:
                value = 0
            
            # Handle sign adjustments for changes
            if 'change_in' in model_name:
                # Increases in assets are negative, increases in liabilities are positive
                if 'receivables' in model_name or 'inventory' in model_name:
                    # Increase in receivables/inventory = negative change
                    result[model_name] = -abs(value) if pd.notna(value) else 0
                elif 'payables' in model_name:
                    # Increase in payables = positive change
                    result[model_name] = abs(value) if pd.notna(value) else 0
            else:
                result[model_name] = value if pd.notna(value) else 0
        else:
            result[model_name] = 0
    
    # Add missing fields with defaults
    result['capital_expenditure'] = 0  # May need to extract from investing activities
    result['debt_repayment'] = 0  # May need to extract from financing activities
    result['dividends_paid'] = 0  # May need to extract from financing activities
    
    return pd.DataFrame([result])


def process_file(filepath: Path) -> Optional[pd.DataFrame]:
    """Process a single AFS Extract file."""
    print(f"Processing: {filepath.name}")
    
    # Determine file type
    if 'Income_Statement' in filepath.name:
        file_type = 'income_statement'
    elif 'Balance_Sheet' in filepath.name:
        file_type = 'balance_sheet'
    elif 'Cash_Flow' in filepath.name:
        file_type = 'cash_flow'
    else:
        print(f"  ⚠️  Unknown file type: {filepath.name}")
        return None
    
    # Extract period
    period = extract_period_from_filename(filepath.name)
    if not period:
        print(f"  ⚠️  Could not extract period from filename")
        return None
    
    # Read CSV with error handling
    try:
        # Try reading with different encodings and error handling
        try:
            df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip', skipinitialspace=True, 
                           skip_blank_lines=True, comment=None)
        except Exception as e1:
            try:
                df = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip', skipinitialspace=True,
                               skip_blank_lines=True)
            except Exception as e2:
                try:
                    df = pd.read_csv(filepath, encoding='cp1252', on_bad_lines='skip', skipinitialspace=True,
                                   skip_blank_lines=True)
                except Exception as e3:
                    # Last resort: try with error_bad_lines=False (pandas < 2.0) or on_bad_lines='warn'
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8', error_bad_lines=False, warn_bad_lines=False,
                                       skipinitialspace=True, skip_blank_lines=True)
                    except:
                        print(f"  ❌ Error reading file: {e1}, {e2}, {e3}")
                        return None
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return None
    
    # Normalize column names (handle variations and case)
    df.columns = df.columns.str.strip()
    
    # Check for 'Line Item' column with variations (case-insensitive)
    line_item_col = None
    for col in df.columns:
        col_lower = col.lower().replace(' ', '').replace('_', '')
        if 'lineitem' in col_lower or col_lower == 'lineitem':
            line_item_col = col
            break
    
    if line_item_col is None:
        # Try first column if it looks like line items
        if len(df.columns) > 0:
            first_col = df.columns[0]
            # Check if first column contains text (likely line items)
            if df[first_col].dtype == 'object' and len(df[first_col].dropna()) > 0:
                line_item_col = first_col
                print(f"  ℹ️  Using first column '{line_item_col}' as Line Item")
            else:
                print(f"  ⚠️  Could not find 'Line Item' column. Available columns: {', '.join(df.columns)}")
                return None
        else:
            print(f"  ⚠️  No columns found in file")
            return None
    
    # Rename to standard name for processing
    if line_item_col != 'Line Item':
        df = df.rename(columns={line_item_col: 'Line Item'})
    
    # Remove empty rows and rows that are just headers/notes
    df = df[df['Line Item'].notna() & (df['Line Item'].str.strip() != '')]
    # Remove rows that are clearly notes or headers (all caps, contain "NOTE", "METHOD", etc.)
    df = df[~df['Line Item'].str.contains('^NOTE|^METHOD|^KEY|^RECONCIL|^ALTERNATIVE', case=False, na=False)]
    # Remove rows where Line Item is just commas or special characters
    df = df[df['Line Item'].str.strip().str.len() > 1]
    
    # Transform based on type
    if file_type == 'income_statement':
        return transform_income_statement(df, period)
    elif file_type == 'balance_sheet':
        return transform_balance_sheet(df, period)
    elif file_type == 'cash_flow':
        return transform_cash_flow(df, period)
    
    return None


def transform_to_line_items(df: pd.DataFrame, period: str, statement_type: str) -> pd.DataFrame:
    """Transform vertical format to detailed line items format."""
    line_items = []
    
    # Find period column
    period_cols = []
    for c in df.columns:
        c_lower = c.upper()
        if (c_lower.startswith('FY') or 'YTD' in c_lower or 'FY202' in c_lower) and \
           c != 'Line Item' and 'Category' not in c and 'Notes' not in c and 'Sub_Category' not in c:
            period_cols.append(c)
    
    if not period_cols:
        for c in df.columns:
            if c not in ['Line Item', 'Category', 'Notes', 'Sub_Category', 'Change', 'Change_Pct', 'YoY_Change', 'YoY_Pct', 'Budget', 'Variance', 'Var%', 'PY_YTD']:
                try:
                    numeric_vals = pd.to_numeric(df[c], errors='coerce').dropna()
                    if len(numeric_vals) > 0:
                        period_cols.append(c)
                except:
                    pass
    
    if not period_cols:
        return pd.DataFrame()
    
    period_col = period_cols[0]
    
    # Get category column
    category_col = None
    for col in df.columns:
        if 'category' in col.lower():
            category_col = col
            break
    
    # Get sub_category column if available
    sub_category_col = None
    for col in df.columns:
        if 'sub_category' in col.lower() or 'subcategory' in col.lower():
            sub_category_col = col
            break
    
    # Process each line item
    for idx, row in df.iterrows():
        line_item_name = str(row['Line Item']).strip() if pd.notna(row['Line Item']) else ''
        
        # Skip empty rows, totals, margins, and notes
        if not line_item_name or line_item_name == '' or len(line_item_name) < 2:
            continue
        if 'Total' in line_item_name and 'Total Cost' not in line_item_name:  # Keep "Total Cost of Sales" for COGS
            continue
        if '%' in line_item_name or 'Margin' in line_item_name:
            continue
        if line_item_name.startswith('NOTE') or line_item_name.startswith('METHOD') or line_item_name.startswith('KEY'):
            continue
        
        # Get amount
        value = row[period_col]
        if pd.isna(value) or value == '' or str(value).strip() == '':
            continue
        
        # Parse accounting format
        if isinstance(value, str):
            value = value.strip()
            if value.startswith('(') and value.endswith(')'):
                amount = -float(value.strip('()').replace(',', ''))
            else:
                try:
                    amount = float(value.replace(',', ''))
                except:
                    continue
        elif pd.notna(value):
            amount = float(value)
        else:
            continue
        
        # Get category
        category = str(row[category_col]).strip() if category_col and pd.notna(row[category_col]) else 'Other'
        
        # Get sub_category
        sub_category = str(row[sub_category_col]).strip() if sub_category_col and pd.notna(row[sub_category_col]) else None
        if sub_category == '' or sub_category == 'nan':
            sub_category = None
        
        line_items.append({
            'period_date': period,
            'line_item_name': line_item_name,
            'category': category,
            'sub_category': sub_category,
            'amount': amount,
            'statement_type': statement_type
        })
    
    return pd.DataFrame(line_items)


def main():
    """Main transformation function."""
    print("=" * 60)
    print("AFS Extract Format Transformer")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Process all files
    income_statements = []
    balance_sheets = []
    cash_flows = []
    
    # Detailed line items
    is_line_items = []
    bs_line_items = []
    cf_line_items = []
    
    for filepath in BASE_DIR.glob("*.csv"):
        result = process_file(filepath)
        if result is not None:
            if 'Income_Statement' in filepath.name:
                income_statements.append(result)
                # Also extract detailed line items
                period = extract_period_from_filename(filepath.name)
                if period:
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip', skipinitialspace=True, skip_blank_lines=True)
                        df.columns = df.columns.str.strip()
                        line_item_col = None
                        for col in df.columns:
                            col_lower = col.lower().replace(' ', '').replace('_', '')
                            if 'lineitem' in col_lower:
                                line_item_col = col
                                break
                        if line_item_col is None and len(df.columns) > 0:
                            line_item_col = df.columns[0]
                        if line_item_col and line_item_col != 'Line Item':
                            df = df.rename(columns={line_item_col: 'Line Item'})
                        df = df[df['Line Item'].notna() & (df['Line Item'].str.strip() != '')]
                        line_items_df = transform_to_line_items(df, period, 'income_statement')
                        if not line_items_df.empty:
                            is_line_items.append(line_items_df)
                    except Exception as e:
                        print(f"  ⚠️  Could not extract line items: {e}")
            elif 'Balance_Sheet' in filepath.name:
                balance_sheets.append(result)
                # Extract detailed line items
                period = extract_period_from_filename(filepath.name)
                if period:
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip', skipinitialspace=True, skip_blank_lines=True)
                        df.columns = df.columns.str.strip()
                        line_item_col = None
                        for col in df.columns:
                            col_lower = col.lower().replace(' ', '').replace('_', '')
                            if 'lineitem' in col_lower:
                                line_item_col = col
                                break
                        if line_item_col is None and len(df.columns) > 0:
                            line_item_col = df.columns[0]
                        if line_item_col and line_item_col != 'Line Item':
                            df = df.rename(columns={line_item_col: 'Line Item'})
                        df = df[df['Line Item'].notna() & (df['Line Item'].str.strip() != '')]
                        line_items_df = transform_to_line_items(df, period, 'balance_sheet')
                        if not line_items_df.empty:
                            bs_line_items.append(line_items_df)
                    except Exception as e:
                        print(f"  ⚠️  Could not extract line items: {e}")
            elif 'Cash_Flow' in filepath.name:
                cash_flows.append(result)
                # Extract detailed line items
                period = extract_period_from_filename(filepath.name)
                if period:
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip', skipinitialspace=True, skip_blank_lines=True)
                        df.columns = df.columns.str.strip()
                        line_item_col = None
                        for col in df.columns:
                            col_lower = col.lower().replace(' ', '').replace('_', '')
                            if 'lineitem' in col_lower:
                                line_item_col = col
                                break
                        if line_item_col is None and len(df.columns) > 0:
                            line_item_col = df.columns[0]
                        if line_item_col and line_item_col != 'Line Item':
                            df = df.rename(columns={line_item_col: 'Line Item'})
                        df = df[df['Line Item'].notna() & (df['Line Item'].str.strip() != '')]
                        df = df[~df['Line Item'].str.contains('^NOTE|^METHOD|^KEY|^RECONCIL|^ALTERNATIVE', case=False, na=False)]
                        line_items_df = transform_to_line_items(df, period, 'cash_flow')
                        if not line_items_df.empty:
                            cf_line_items.append(line_items_df)
                    except Exception as e:
                        print(f"  ⚠️  Could not extract line items: {e}")
    
    # Combine and save SUMMARY files
    if income_statements:
        is_df = pd.concat(income_statements, ignore_index=True)
        is_df = is_df.sort_values('month')
        output_path = OUTPUT_DIR / "historic_financials.csv"
        is_df.to_csv(output_path, index=False)
        print(f"\n✅ Income Statements (Summary): {len(income_statements)} periods → {output_path}")
        print(f"   Columns: {', '.join(is_df.columns)}")
    
    if balance_sheets:
        bs_df = pd.concat(balance_sheets, ignore_index=True)
        bs_df = bs_df.sort_values('month')
        output_path = OUTPUT_DIR / "historical_balance_sheet.csv"
        bs_df.to_csv(output_path, index=False)
        print(f"\n✅ Balance Sheets (Summary): {len(balance_sheets)} periods → {output_path}")
        print(f"   Columns: {', '.join(bs_df.columns)}")
    
    if cash_flows:
        cf_df = pd.concat(cash_flows, ignore_index=True)
        cf_df = cf_df.sort_values('month')
        output_path = OUTPUT_DIR / "historical_cashflow.csv"
        cf_df.to_csv(output_path, index=False)
        print(f"\n✅ Cash Flows (Summary): {len(cash_flows)} periods → {output_path}")
        print(f"   Columns: {', '.join(cf_df.columns)}")
    
    # Combine and save DETAILED LINE ITEMS files
    if is_line_items:
        is_li_df = pd.concat(is_line_items, ignore_index=True)
        is_li_df = is_li_df.sort_values('period_date')
        output_path = OUTPUT_DIR / "historical_income_statement_line_items.csv"
        is_li_df.to_csv(output_path, index=False)
        print(f"\n✅ Income Statement Line Items (Detailed): {len(is_li_df)} line items across {is_li_df['period_date'].nunique()} periods → {output_path}")
        print(f"   Columns: {', '.join(is_li_df.columns)}")
        print(f"   Sample line items: {', '.join(is_li_df['line_item_name'].unique()[:10])}")
    
    if bs_line_items:
        bs_li_df = pd.concat(bs_line_items, ignore_index=True)
        bs_li_df = bs_li_df.sort_values('period_date')
        output_path = OUTPUT_DIR / "historical_balance_sheet_line_items.csv"
        bs_li_df.to_csv(output_path, index=False)
        print(f"\n✅ Balance Sheet Line Items (Detailed): {len(bs_li_df)} line items across {bs_li_df['period_date'].nunique()} periods → {output_path}")
        print(f"   Columns: {', '.join(bs_li_df.columns)}")
        print(f"   Sample line items: {', '.join(bs_li_df['line_item_name'].unique()[:10])}")
    
    if cf_line_items:
        cf_li_df = pd.concat(cf_line_items, ignore_index=True)
        cf_li_df = cf_li_df.sort_values('period_date')
        output_path = OUTPUT_DIR / "historical_cashflow_line_items.csv"
        cf_li_df.to_csv(output_path, index=False)
        print(f"\n✅ Cash Flow Line Items (Detailed): {len(cf_li_df)} line items across {cf_li_df['period_date'].nunique()} periods → {output_path}")
        print(f"   Columns: {', '.join(cf_li_df.columns)}")
        print(f"   Sample line items: {', '.join(cf_li_df['line_item_name'].unique()[:10])}")
    
    print("\n" + "=" * 60)
    print("Transformation complete!")
    print(f"Output files saved to: {OUTPUT_DIR}")
    print("\n📊 FILES GENERATED:")
    print("   Summary files (for quick import):")
    print("   - historic_financials.csv")
    print("   - historical_balance_sheet.csv")
    print("   - historical_cashflow.csv")
    print("\n   Detailed line item files (for complete data):")
    print("   - historical_income_statement_line_items.csv")
    print("   - historical_balance_sheet_line_items.csv")
    print("   - historical_cashflow_line_items.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
