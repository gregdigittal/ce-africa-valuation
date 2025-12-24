"""
Trial Balance Processor
=======================
Automatically extracts Income Statements and Balance Sheets from trial balances,
then calculates Cash Flow Statements.

Sprint 23: Trial Balance to Financial Statements conversion.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import re


# Account classification patterns
INCOME_STATEMENT_ACCOUNTS = {
    # Revenue accounts
    'revenue': [
        r'revenue', r'sales', r'income', r'turnover', r'fees',
        r'4\d{3}',  # Account codes starting with 4 (common revenue codes)
    ],
    # Cost of goods sold
    'cogs': [
        r'cost.*sales', r'cogs', r'cost.*goods', r'direct.*cost',
        r'5\d{3}',  # Account codes starting with 5 (common COGS codes)
    ],
    # Operating expenses
    'opex': [
        r'operating.*expense', r'opex', r'admin', r'overhead',
        r'salaries', r'wages', r'rent', r'utilities', r'insurance',
        r'personnel', r'facilities', r'administrative', r'sales.*marketing',
        r'other.*operating', r'general.*admin', r'ga', r'sga',
        r'6\d{3}',  # Account codes starting with 6 (common expense codes)
    ],
    # Other income
    'other_income': [
        r'other.*income', r'interest.*income', r'dividend.*income',
    ],
    # Other expenses
    'other_expense': [
        r'other.*expense', r'interest.*expense', r'finance.*cost',
    ],
    # Depreciation
    'depreciation': [
        r'depreciation', r'amortization', r'amortisation',
        r'depreciation.*total', r'total.*depreciation',  # Handle "Depreciation - Total" variations
        r'depreciation.*net', r'net.*depreciation',  # Handle "Net Depreciation" variations
    ],
    # Tax
    'tax': [
        r'tax', r'income.*tax', r'corporation.*tax',
        r'tax.*expense', r'tax.*provision', r'tax.*payable',  # Handle tax variations
        r'total.*tax', r'tax.*total',  # Handle "Tax - Total" variations
    ],
}

BALANCE_SHEET_ACCOUNTS = {
    # Assets
    'current_assets': [
        r'cash', r'bank', r'debtors', r'receivables', r'inventory', r'stock',
        r'prepaid', r'current.*asset',
        r'1\d{3}',  # Account codes starting with 1 (common asset codes)
    ],
    'fixed_assets': [
        r'fixed.*asset', r'ppe', r'property', r'plant', r'equipment',
        r'machinery', r'vehicles', r'accumulated.*depreciation',
        r'2\d{3}',  # Account codes starting with 2 (common fixed asset codes)
    ],
    'intangible_assets': [
        r'intangible', r'goodwill', r'patents', r'trademarks',
    ],
    # Liabilities
    'current_liabilities': [
        r'creditors', r'payables', r'short.*term.*debt', r'overdraft',
        r'accruals', r'current.*liability', r'trade.*payable',
        r'3\d{3}',  # Account codes starting with 3 (common liability codes)
    ],
    'long_term_liabilities': [
        r'long.*term.*debt', r'loan', r'bond', r'non.*current.*liability',
    ],
    # Equity
    'equity': [
        r'equity', r'share.*capital', r'retained.*earnings', r'reserves',
        r'capital', r'3\d{3}',  # Some equity codes start with 3
    ],
}


class TrialBalanceProcessor:
    """
    Process trial balances to extract financial statements.
    """
    
    def __init__(self):
        """Initialize processor."""
        self.account_classifications = {}
    
    def classify_account(
        self,
        account_name: str,
        account_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify an account as Income Statement or Balance Sheet.
        
        Args:
            account_name: Account name/description
            account_code: Optional account code
        
        Returns:
            Dictionary with classification:
            {
                'statement': 'income_statement' | 'balance_sheet',
                'category': 'revenue' | 'cogs' | 'opex' | 'current_assets' | etc.,
                'confidence': float (0-1)
            }
        """
        account_lower = str(account_name).lower()
        code_str = str(account_code) if account_code else ""
        
        # Check Income Statement patterns
        is_score = 0.0
        is_category = None
        
        for category, patterns in INCOME_STATEMENT_ACCOUNTS.items():
            for pattern in patterns:
                if re.search(pattern, account_lower, re.IGNORECASE):
                    is_score = max(is_score, 0.8)
                    is_category = category
                    break
                if code_str and re.match(pattern, code_str):
                    is_score = max(is_score, 0.9)
                    is_category = category
                    break
        
        # Check Balance Sheet patterns
        bs_score = 0.0
        bs_category = None
        
        for category, patterns in BALANCE_SHEET_ACCOUNTS.items():
            for pattern in patterns:
                if re.search(pattern, account_lower, re.IGNORECASE):
                    bs_score = max(bs_score, 0.8)
                    bs_category = category
                    break
                if code_str and re.match(pattern, code_str):
                    bs_score = max(bs_score, 0.9)
                    bs_category = category
                    break
        
        # Determine classification
        if is_score > bs_score and is_score > 0.5:
            return {
                'statement': 'income_statement',
                'category': is_category or 'other',
                'confidence': is_score
            }
        elif bs_score > is_score and bs_score > 0.5:
            return {
                'statement': 'balance_sheet',
                'category': bs_category or 'other',
                'confidence': bs_score
            }
        else:
            # Default: try to infer from account type
            # Revenue/expense accounts are typically IS
            # Asset/liability/equity accounts are typically BS
            if any(word in account_lower for word in ['revenue', 'income', 'expense', 'cost', 'profit', 'loss']):
                return {
                    'statement': 'income_statement',
                    'category': 'other',
                    'confidence': 0.6
                }
            elif any(word in account_lower for word in ['asset', 'liability', 'equity', 'capital', 'debt', 'loan']):
                return {
                    'statement': 'balance_sheet',
                    'category': 'other',
                    'confidence': 0.6
                }
            else:
                # Unknown - default to balance sheet (trial balance items are often BS)
                return {
                    'statement': 'balance_sheet',
                    'category': 'other',
                    'confidence': 0.4
                }
    
    def process_trial_balance(
        self,
        tb_df: pd.DataFrame,
        period_column: str = 'period',
        account_column: str = 'account_name',
        account_code_column: Optional[str] = 'account_code',
        debit_column: str = 'debit',
        credit_column: str = 'credit'
    ) -> Dict[str, pd.DataFrame]:
        """
        Process trial balance DataFrame and extract financial statements.
        
        Args:
            tb_df: Trial balance DataFrame with columns:
                - period/period_date: Period identifier
                - account_name: Account name
                - account_code: Optional account code
                - debit: Debit amount
                - credit: Credit amount
            period_column: Name of period column
            account_column: Name of account name column
            account_code_column: Name of account code column (optional)
            debit_column: Name of debit column
            credit_column: Name of credit column
        
        Returns:
            Dictionary with:
            {
                'income_statement': DataFrame,
                'balance_sheet': DataFrame,
                'cash_flow': DataFrame,
                'classifications': DataFrame (account classifications)
            }
        """
        # Validate required columns
        required_cols = [period_column, account_column, debit_column, credit_column]
        missing_cols = [col for col in required_cols if col not in tb_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Classify all accounts
        classifications = []
        for idx, row in tb_df.iterrows():
            account_name = row[account_column]
            account_code = row.get(account_code_column) if account_code_column else None
            
            classification = self.classify_account(account_name, account_code)
            classifications.append({
                'account_name': account_name,
                'account_code': account_code,
                'statement': classification['statement'],
                'category': classification['category'],
                'confidence': classification['confidence']
            })
        
        classifications_df = pd.DataFrame(classifications)
        
        # Normalize debit and credit values to positive
        # Some accounting systems store credits as negative, or use different conventions
        # We'll use absolute values and rely on which column has the value to determine the sign
        tb_df = tb_df.copy()
        tb_df['debit'] = pd.to_numeric(tb_df[debit_column], errors='coerce').fillna(0).abs()
        tb_df['credit'] = pd.to_numeric(tb_df[credit_column], errors='coerce').fillna(0).abs()
        
        # Merge classifications (handle case where account_code_column might not exist)
        merge_cols = [account_column]
        if account_code_column and account_code_column in tb_df.columns:
            merge_cols.append(account_code_column)
        
        tb_df = tb_df.merge(classifications_df, on=merge_cols, how='left')
        
        # Extract Income Statement
        is_df = self._extract_income_statement(tb_df, period_column, account_column, classifications_df)
        
        # Extract Balance Sheet
        bs_df = self._extract_balance_sheet(tb_df, period_column, account_column, classifications_df)
        
        # Calculate Cash Flow Statement
        cf_df = self._calculate_cash_flow(is_df, bs_df, period_column)
        
        return {
            'income_statement': is_df,
            'balance_sheet': bs_df,
            'cash_flow': cf_df,
            'classifications': classifications_df
        }
    
    def _extract_income_statement(
        self,
        tb_df: pd.DataFrame,
        period_column: str,
        account_column: str,
        classifications_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract income statement from trial balance."""
        # Filter to income statement accounts
        is_accounts = classifications_df[classifications_df['statement'] == 'income_statement']
        if not is_accounts.empty and account_column in is_accounts.columns:
            is_tb = tb_df[tb_df[account_column].isin(is_accounts[account_column])]
        else:
            is_tb = pd.DataFrame()
        
        # Group by period and category
        is_data = []
        if is_tb.empty:
            return pd.DataFrame(columns=[period_column, 'revenue', 'cogs', 'gross_profit', 'opex', 'depreciation', 'other_income', 'other_expense', 'ebit', 'tax', 'net_profit'])
        
        for period in tb_df[period_column].unique():
            period_data = is_tb[is_tb[period_column] == period]
            
            # Calculate net amounts by category
            # Revenue: credit - debit (positive)
            # Expenses: debit - credit (positive)
            revenue = 0
            cogs = 0
            opex = 0
            depreciation = 0
            other_income = 0
            other_expense = 0
            tax = 0
            
            # Track account-level details for debugging
            account_details = []
            
            for _, row in period_data.iterrows():
                category = row.get('category', 'other')
                # Values are already normalized to positive in the DataFrame processing
                # Convert to float, ensuring positive values
                try:
                    debit = abs(float(row.get('debit', 0) or 0))
                    credit = abs(float(row.get('credit', 0) or 0))
                except (ValueError, TypeError):
                    debit = 0.0
                    credit = 0.0
                
                account_name = str(row.get(account_column, ''))
                
                # Auto-correct classification based on balance direction
                # Revenue accounts should have credit > debit (credit normal)
                # Expense accounts should have debit > credit (debit normal)
                
                credit_balance = credit > debit
                debit_balance = debit > credit
                
                # Check if classification seems wrong based on balance
                if category == 'revenue' and debit_balance and not credit_balance:
                    # Revenue account with debit balance - might be misclassified expense
                    # Check account name
                    account_lower = account_name.lower()
                    if any(word in account_lower for word in ['cost', 'expense', 'cogs', 'direct']):
                        # Reclassify as COGS
                        category = 'cogs'
                elif category == 'cogs' and credit_balance and not debit_balance:
                    # COGS account with credit balance - might be misclassified revenue
                    account_lower = account_name.lower()
                    if any(word in account_lower for word in ['revenue', 'sales', 'income']):
                        # Reclassify as revenue
                        category = 'revenue'
                
                # INCOME STATEMENT CALCULATION (based on accounting standards):
                # Revenue/Income: ALWAYS credit normal → amount = credit - debit
                #   - Normal: credit > debit → positive revenue ✓
                #   - Unusual: debit > credit → negative revenue (returns exceed sales)
                # Expenses: ALWAYS debit normal → amount = debit - credit
                #   - Normal: debit > credit → positive expense ✓
                #   - Unusual: credit > debit → negative expense (credits/refunds)
                
                if category == 'revenue':
                    # Revenue: ALWAYS credit - debit (credit normal)
                    # Ensure we get positive value for normal revenue accounts
                    amount = credit - debit
                    # If amount is negative but credit > debit, there might be a data issue
                    # Log for debugging but use the calculated value
                    revenue += amount
                elif category == 'cogs':
                    # COGS: ALWAYS debit - credit (debit normal)
                    amount = debit - credit
                    cogs += amount
                elif category == 'opex':
                    # OPEX: ALWAYS debit - credit (debit normal)
                    amount = debit - credit
                    opex += amount
                elif category == 'depreciation':
                    # Depreciation: ALWAYS debit - credit (debit normal)
                    amount = debit - credit
                    depreciation += amount
                elif category == 'other_income':
                    # Other Income: ALWAYS credit - debit (credit normal)
                    amount = credit - debit
                    other_income += amount
                elif category == 'other_expense':
                    # Other Expense: ALWAYS debit - credit (debit normal)
                    amount = debit - credit
                    other_expense += amount
                elif category == 'tax':
                    # Tax: ALWAYS debit - credit (debit normal)
                    amount = debit - credit
                    tax += amount
                # If category is 'other' but it's an income statement account, try to infer
                elif category == 'other':
                    account_lower = account_name.lower()
                    # If credit > debit, likely revenue/income
                    if credit_balance:
                        # Check if it sounds like an expense account
                        if any(word in account_lower for word in ['cost', 'expense', 'charge', 'fee', 'cogs']):
                            opex += debit - credit  # Expense with credit balance (unusual, but possible)
                        else:
                            revenue += credit - debit
                    # If debit > credit, likely expense
                    elif debit_balance:
                        # Check if it sounds like a revenue account
                        if any(word in account_lower for word in ['revenue', 'sales', 'income', 'turnover']):
                            revenue += credit - debit  # Revenue with debit balance (returns/refunds)
                        else:
                            opex += debit - credit
            
            # Calculate derived metrics
            gross_profit = revenue - cogs
            operating_profit = gross_profit - opex - depreciation
            ebit = operating_profit + other_income - other_expense
            net_profit = ebit - tax
            
            is_data.append({
                period_column: period,
                'revenue': revenue,
                'cogs': cogs,
                'gross_profit': gross_profit,
                'opex': opex,
                'depreciation': depreciation,
                'other_income': other_income,
                'other_expense': other_expense,
                'ebit': ebit,
                'tax': tax,
                'net_profit': net_profit
            })
        
        return pd.DataFrame(is_data)
    
    def _extract_balance_sheet(
        self,
        tb_df: pd.DataFrame,
        period_column: str,
        account_column: str,
        classifications_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract balance sheet from trial balance."""
        # Filter to balance sheet accounts
        bs_accounts = classifications_df[classifications_df['statement'] == 'balance_sheet']
        if not bs_accounts.empty and account_column in bs_accounts.columns:
            bs_tb = tb_df[tb_df[account_column].isin(bs_accounts[account_column])]
        else:
            bs_tb = pd.DataFrame()
        
        # Group by period and category
        bs_data = []
        if bs_tb.empty:
            return pd.DataFrame(columns=[period_column, 'current_assets', 'fixed_assets', 'intangible_assets', 'total_assets', 'current_liabilities', 'long_term_liabilities', 'total_liabilities', 'equity', 'total_equity', 'total_liabilities_and_equity'])
        
        for period in tb_df[period_column].unique():
            period_data = bs_tb[bs_tb[period_column] == period]
            
            # Calculate balances by category
            # Assets: debit - credit (positive)
            # Liabilities/Equity: credit - debit (positive)
            current_assets = 0
            fixed_assets = 0
            intangible_assets = 0
            current_liabilities = 0
            long_term_liabilities = 0
            equity = 0
            
            for _, row in period_data.iterrows():
                category = row.get('category', 'other')
                debit = float(row.get('debit', 0) or 0)
                credit = float(row.get('credit', 0) or 0)
                
                # BALANCE SHEET: All values should be positive
                # Assets: Debit normal → debit - credit (positive if debit > credit)
                # Liabilities/Equity: Credit normal → credit - debit (positive if credit > debit)
                
                if category in ['current_assets']:
                    amount = debit - credit
                    amount = abs(amount) if amount < 0 else amount  # Ensure positive
                    current_assets += amount
                elif category in ['fixed_assets']:
                    amount = debit - credit
                    amount = abs(amount) if amount < 0 else amount
                    fixed_assets += amount
                elif category in ['intangible_assets']:
                    amount = debit - credit
                    amount = abs(amount) if amount < 0 else amount
                    intangible_assets += amount
                elif category in ['current_liabilities']:
                    amount = credit - debit
                    amount = abs(amount) if amount < 0 else amount  # Ensure positive
                    current_liabilities += amount
                elif category in ['long_term_liabilities']:
                    amount = credit - debit
                    amount = abs(amount) if amount < 0 else amount
                    long_term_liabilities += amount
                elif category in ['equity']:
                    amount = credit - debit
                    amount = abs(amount) if amount < 0 else amount
                    equity += amount
            
            # Calculate totals
            total_assets = current_assets + fixed_assets + intangible_assets
            total_liabilities = current_liabilities + long_term_liabilities
            total_equity = equity
            total_liabilities_and_equity = total_liabilities + total_equity
            
            bs_data.append({
                period_column: period,
                'current_assets': current_assets,
                'fixed_assets': fixed_assets,
                'intangible_assets': intangible_assets,
                'total_assets': total_assets,
                'current_liabilities': current_liabilities,
                'long_term_liabilities': long_term_liabilities,
                'total_liabilities': total_liabilities,
                'equity': equity,
                'total_equity': equity,
                'total_liabilities_and_equity': total_liabilities_and_equity
            })
        
        return pd.DataFrame(bs_data)
    
    def _calculate_cash_flow(
        self,
        is_df: pd.DataFrame,
        bs_df: pd.DataFrame,
        period_column: str
    ) -> pd.DataFrame:
        """
        Calculate cash flow statement from income statement and balance sheet.
        
        Uses indirect method:
        - Operating: Net income + non-cash items + working capital changes
        - Investing: Changes in fixed assets
        - Financing: Changes in debt and equity
        """
        # Merge IS and BS by period
        merged = is_df.merge(bs_df, on=period_column, how='outer', suffixes=('_is', '_bs'))
        merged = merged.sort_values(period_column)
        
        cf_data = []
        
        for i, row in merged.iterrows():
            period = row[period_column]
            
            # Operating Cash Flow (Indirect Method)
            net_income = row.get('net_profit', 0)
            depreciation = row.get('depreciation', 0)
            
            # Working capital changes (need previous period)
            if i > 0:
                prev_row = merged.iloc[i - 1]
                
                # Change in current assets (negative impact on cash)
                delta_current_assets = (
                    row.get('current_assets', 0) - prev_row.get('current_assets', 0)
                )
                
                # Change in current liabilities (positive impact on cash)
                delta_current_liabilities = (
                    row.get('current_liabilities', 0) - prev_row.get('current_liabilities', 0)
                )
                
                # Net working capital change
                delta_wc = delta_current_liabilities - delta_current_assets
            else:
                delta_wc = 0
            
            cash_from_operations = net_income + depreciation + delta_wc
            
            # Investing Cash Flow
            if i > 0:
                prev_row = merged.iloc[i - 1]
                # Change in fixed assets (negative = investment, positive = disposal)
                delta_fixed_assets = (
                    row.get('fixed_assets', 0) - prev_row.get('fixed_assets', 0)
                )
                # Add back depreciation (already included in fixed assets change)
                cash_from_investing = -(delta_fixed_assets - depreciation)
            else:
                cash_from_investing = 0
            
            # Financing Cash Flow
            if i > 0:
                prev_row = merged.iloc[i - 1]
                # Change in debt
                delta_debt = (
                    (row.get('current_liabilities', 0) + row.get('long_term_liabilities', 0)) -
                    (prev_row.get('current_liabilities', 0) + prev_row.get('long_term_liabilities', 0))
                )
                # Change in equity
                delta_equity = (
                    row.get('equity', 0) - prev_row.get('equity', 0)
                )
                cash_from_financing = delta_debt + delta_equity
            else:
                cash_from_financing = 0
            
            # Net change in cash
            net_cash_flow = cash_from_operations + cash_from_investing + cash_from_financing
            
            # Beginning cash (from previous period's ending cash)
            if i > 0:
                prev_cf = cf_data[-1]
                beginning_cash = prev_cf.get('ending_cash', 0)
            else:
                beginning_cash = row.get('current_assets', 0)  # Assume cash is in current assets
            
            ending_cash = beginning_cash + net_cash_flow
            
            cf_data.append({
                period_column: period,
                'cash_from_operations': cash_from_operations,
                'cash_from_investing': cash_from_investing,
                'cash_from_financing': cash_from_financing,
                'net_cash_flow': net_cash_flow,
                'beginning_cash': beginning_cash,
                'ending_cash': ending_cash
            })
        
        return pd.DataFrame(cf_data)


def _save_extracted_statements(
    db,
    scenario_id: str,
    user_id: str,
    is_df: pd.DataFrame,
    bs_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    period_column: str
) -> Dict[str, Any]:
    """Save extracted financial statements to database."""
    saved = {
        'trial_balance': False,
        'income_statement': False,
        'balance_sheet': False,
        'cash_flow': False
    }
    
    try:
        # Note: The trial balance is already imported and saved to the database.
        saved['trial_balance'] = True  # Already exists in DB
        
        # Save income statement
        if not is_df.empty:
            is_records = []
            for _, row in is_df.iterrows():
                record = {
                    'scenario_id': scenario_id,
                    'revenue': row.get('revenue', 0),
                    'cogs': row.get('cogs', 0),
                    'gross_profit': row.get('gross_profit', 0),
                    'opex': row.get('opex', 0),
                    'ebit': row.get('ebit', 0),
                }
                
                # Map period to month format
                period = row[period_column]
                if isinstance(period, str):
                    try:
                        period_date = pd.to_datetime(period)
                    except:
                        period_date = datetime.now()
                elif hasattr(period, 'strftime'):
                    period_date = period
                else:
                    period_date = datetime.now()
                
                record['month'] = period_date.strftime('%Y-%m-%d') if hasattr(period_date, 'strftime') else str(period_date)
                is_records.append(record)
            
            if is_records:
                try:
                    db.client.table('historic_financials').upsert(is_records).execute()
                except:
                    try:
                        db.client.table('historic_financials').upsert(
                            is_records, on_conflict='scenario_id,month'
                        ).execute()
                    except:
                        for record in is_records:
                            db.client.table('historic_financials').delete().eq(
                                'scenario_id', scenario_id
                            ).eq('month', record['month']).execute()
                        db.client.table('historic_financials').insert(is_records).execute()
                saved['income_statement'] = True
        
        # Save balance sheet
        if not bs_df.empty:
            bs_records = []
            for _, row in bs_df.iterrows():
                current_assets = row.get('current_assets', 0)
                fixed_assets = row.get('fixed_assets', 0)
                intangible_assets = row.get('intangible_assets', 0)
                current_liabilities = row.get('current_liabilities', 0)
                long_term_liabilities = row.get('long_term_liabilities', 0)
                equity = row.get('equity', 0)
                
                period = row[period_column]
                if isinstance(period, str):
                    try:
                        period_date = pd.to_datetime(period)
                    except:
                        period_date = datetime.now()
                elif hasattr(period, 'strftime'):
                    period_date = period
                elif hasattr(period, 'year'):
                    period_date = period
                else:
                    period_date = datetime.now()
                
                if not hasattr(period_date, 'year'):
                    period_date = pd.to_datetime(period_date)
                
                period_date_str = period_date.replace(day=1).strftime('%Y-%m-%d') if hasattr(period_date, 'replace') else pd.to_datetime(period_date).replace(day=1).strftime('%Y-%m-%d')
                
                record = {
                    'scenario_id': scenario_id,
                    'period_date': period_date_str,
                    'period_year': period_date.year,
                    'period_month': period_date.month,
                    'is_actual': True,
                    'total_current_assets': current_assets,
                    'net_ppe': fixed_assets,
                    'intangible_assets': intangible_assets,
                    'total_non_current_assets': fixed_assets + intangible_assets,
                    'total_assets': row.get('total_assets', 0),
                    'total_current_liabilities': current_liabilities,
                    'long_term_debt': long_term_liabilities,
                    'total_non_current_liabilities': long_term_liabilities,
                    'total_liabilities': row.get('total_liabilities', 0),
                    'share_capital': equity * 0.5,
                    'retained_earnings': equity * 0.5,
                    'total_equity': equity,
                }
                bs_records.append(record)
            
            if bs_records:
                try:
                    db.client.table('historical_balance_sheet').upsert(bs_records).execute()
                except:
                    try:
                        db.client.table('historical_balance_sheet').upsert(
                            bs_records, on_conflict='scenario_id,period_date'
                        ).execute()
                    except:
                        for record in bs_records:
                            db.client.table('historical_balance_sheet').delete().eq(
                                'scenario_id', scenario_id
                            ).eq('period_date', record['period_date']).execute()
                        db.client.table('historical_balance_sheet').insert(bs_records).execute()
                saved['balance_sheet'] = True
        
        # Save cash flow
        if not cf_df.empty:
            cf_records = []
            for _, row in cf_df.iterrows():
                record = {
                    'scenario_id': scenario_id,
                    'cash_from_operations': row.get('cash_from_operations', 0),
                    'cash_from_investing': row.get('cash_from_investing', 0),
                    'cash_from_financing': row.get('cash_from_financing', 0),
                }
                
                period = row[period_column]
                if isinstance(period, str):
                    try:
                        period_date = pd.to_datetime(period)
                    except:
                        period_date = datetime.now()
                elif hasattr(period, 'strftime'):
                    period_date = period
                else:
                    period_date = datetime.now()
                
                record['month'] = period_date.strftime('%Y-%m-%d') if hasattr(period_date, 'strftime') else str(period_date)
                cf_records.append(record)
            
            if cf_records:
                try:
                    db.client.table('historical_cashflow').upsert(cf_records).execute()
                except:
                    try:
                        db.client.table('historical_cashflow').upsert(
                            cf_records, on_conflict='scenario_id,month'
                        ).execute()
                    except:
                        for record in cf_records:
                            db.client.table('historical_cashflow').delete().eq(
                                'scenario_id', scenario_id
                            ).eq('month', record['month']).execute()
                        db.client.table('historical_cashflow').insert(cf_records).execute()
                saved['cash_flow'] = True
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'saved': saved
        }
    
    return {
        'success': True,
        'saved': saved
    }


def process_trial_balance_import(
    db,
    scenario_id: str,
    user_id: str,
    tb_df: pd.DataFrame,
    period_column: str = 'period',
    account_column: str = 'account_name',
    account_code_column: Optional[str] = 'account_code',
    debit_column: str = 'debit',
    credit_column: str = 'credit'
) -> Dict[str, Any]:
    """
    Process trial balance import and save extracted financial statements.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        user_id: User ID
        tb_df: Trial balance DataFrame
        period_column: Name of period column
        account_column: Name of account name column
        account_code_column: Name of account code column
        debit_column: Name of debit column
        credit_column: Name of credit column
    
    Returns:
        Dictionary with processing results and saved data
    """
    processor = TrialBalanceProcessor()
    
    # Validate trial balance data
    if tb_df.empty:
        return {
            'success': False,
            'error': 'Trial balance DataFrame is empty',
            'saved': {'trial_balance': False, 'income_statement': False, 'balance_sheet': False, 'cash_flow': False},
            'results': {}
        }
    
    # Check required columns exist
    required_cols = [period_column, account_column, debit_column, credit_column]
    missing_cols = [col for col in required_cols if col not in tb_df.columns]
    if missing_cols:
        return {
            'success': False,
            'error': f'Missing required columns: {missing_cols}',
            'saved': {'trial_balance': False, 'income_statement': False, 'balance_sheet': False, 'cash_flow': False},
            'results': {}
        }
    
    # Process trial balance
    try:
        results = processor.process_trial_balance(
            tb_df,
            period_column,
            account_column,
            account_code_column,
            debit_column,
            credit_column
        )
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': f'Error processing trial balance: {str(e)}',
            'traceback': traceback.format_exc(),
            'saved': {'trial_balance': False, 'income_statement': False, 'balance_sheet': False, 'cash_flow': False},
            'results': {}
        }
    
    is_df = results['income_statement']
    bs_df = results['balance_sheet']
    cf_df = results['cash_flow']
    classifications_df = results['classifications']
    
    # Validate extraction results
    if is_df.empty and bs_df.empty:
        return {
            'success': False,
            'error': 'No financial statements could be extracted. Check account classifications.',
            'results': results,
            'saved': {'trial_balance': False, 'income_statement': False, 'balance_sheet': False, 'cash_flow': False}
        }
    
    # Save to database
    save_result = _save_extracted_statements(
        db, scenario_id, user_id,
        is_df, bs_df, cf_df,
        period_column
    )
    
    return {
        'success': save_result['success'],
        'error': save_result.get('error'),
        'saved': save_result.get('saved', {}),
        'results': results,
        'income_statement_rows': len(is_df),
        'balance_sheet_rows': len(bs_df),
        'cash_flow_rows': len(cf_df)
    }


def _save_extracted_statements(
    db,
    scenario_id: str,
    user_id: str,
    is_df: pd.DataFrame,
    bs_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    period_column: str
) -> Dict[str, Any]:
    """Save extracted financial statements to database."""
    saved = {
        'trial_balance': False,
        'income_statement': False,
        'balance_sheet': False,
        'cash_flow': False
    }
    
    try:
        # Note: The trial balance is already imported and saved to the database.
        # We don't need to save it again - we're just extracting financial statements from it.
        # The tb_df passed in is for processing only, not for saving.
        saved['trial_balance'] = True  # Already exists in DB
        
        # Save income statement
        # Note: historic_financials table only has scenario_id (no user_id)
        if not is_df.empty:
            is_records = []
            for _, row in is_df.iterrows():
                record = {
                    'scenario_id': scenario_id,
                    'revenue': row.get('revenue', 0),
                    'cogs': row.get('cogs', 0),
                    'gross_profit': row.get('gross_profit', 0),
                    'opex': row.get('opex', 0),
                    'depreciation': row.get('depreciation', 0),  # Include depreciation
                    'other_income': row.get('other_income', 0),  # Include other_income
                    'other_expense': row.get('other_expense', 0),  # Include other_expense (may contain interest)
                    'ebit': row.get('ebit', 0),
                    'tax': row.get('tax', 0),  # Include tax
                    'net_profit': row.get('net_profit', 0),  # Include net_profit
                }
                
                # Extract interest_expense from other_expense if available
                # (In many trial balances, interest is the main component of other_expense)
                if 'interest_expense' in row:
                    record['interest_expense'] = row.get('interest_expense', 0)
                elif row.get('other_expense', 0) != 0:
                    # Use other_expense as interest_expense (common case where interest is the main other expense)
                    record['interest_expense'] = row.get('other_expense', 0)
                else:
                    record['interest_expense'] = 0
                
                # Map period to month format
                period = row[period_column]
                if isinstance(period, str):
                    try:
                        period_date = pd.to_datetime(period)
                    except:
                        period_date = datetime.now()
                elif hasattr(period, 'strftime'):
                    period_date = period
                else:
                    period_date = datetime.now()
                
                record['month'] = period_date.strftime('%Y-%m-%d') if hasattr(period_date, 'strftime') else str(period_date)
                is_records.append(record)
            
            # Save to historic_financials (P&L format)
            if is_records:
                # Filter records to only include columns that exist in the database
                # This prevents errors if the schema hasn't been migrated yet
                safe_records = []
                for record in is_records:
                    # Only include columns that are known to exist in the base schema
                    # Additional columns (depreciation, interest_expense, tax) will be added via migration
                    safe_record = {
                        'scenario_id': record.get('scenario_id'),
                        'month': record.get('month'),
                        'revenue': record.get('revenue', 0),
                        'cogs': record.get('cogs', 0),
                        'gross_profit': record.get('gross_profit', 0),
                        'opex': record.get('opex', 0),
                        'ebit': record.get('ebit', 0),
                    }
                    # Add optional columns only if they exist (after migration)
                    # Try to include them - if they fail, we'll catch the error
                    optional_cols = ['depreciation', 'interest_expense', 'tax', 'other_income', 'other_expense', 'net_profit']
                    for col in optional_cols:
                        if col in record:
                            safe_record[col] = record[col]
                    safe_records.append(safe_record)
                
                # Try upsert - if unique constraint doesn't match, delete and insert
                try:
                    db.client.table('historic_financials').upsert(
                        safe_records
                    ).execute()
                    saved['income_statement'] = True
                except Exception as e:
                    error_msg = str(e)
                    # If error is about missing columns, try without optional columns
                    if 'column' in error_msg.lower() and ('depreciation' in error_msg.lower() or 'interest' in error_msg.lower() or 'tax' in error_msg.lower()):
                        st.warning(f"⚠️ Database schema needs migration. Some columns may be missing. Error: {error_msg}")
                        st.info("💡 Please run the migration: `migrations_add_depreciation_interest_tax.sql` to add depreciation, interest_expense, and tax columns.")
                        # Try saving without optional columns
                        minimal_records = []
                        for record in safe_records:
                            minimal_record = {
                                'scenario_id': record.get('scenario_id'),
                                'month': record.get('month'),
                                'revenue': record.get('revenue', 0),
                                'cogs': record.get('cogs', 0),
                                'gross_profit': record.get('gross_profit', 0),
                                'opex': record.get('opex', 0),
                                'ebit': record.get('ebit', 0),
                            }
                            minimal_records.append(minimal_record)
                        try:
                            db.client.table('historic_financials').upsert(
                                minimal_records
                            ).execute()
                            saved['income_statement'] = True
                            st.warning("✅ Saved basic financial data. Re-run extraction after migration to include depreciation, interest, and tax.")
                        except Exception as e2:
                            st.error(f"❌ Failed to save even basic financial data: {str(e2)}")
                    else:
                        # Other error - try with explicit on_conflict
                        try:
                            db.client.table('historic_financials').upsert(
                                safe_records,
                                on_conflict='scenario_id,month'
                            ).execute()
                            saved['income_statement'] = True
                        except:
                            # Last resort: delete existing and insert
                            try:
                                for record in safe_records:
                                    db.client.table('historic_financials').delete().eq(
                                        'scenario_id', scenario_id
                                    ).eq('month', record['month']).execute()
                                db.client.table('historic_financials').insert(safe_records).execute()
                                saved['income_statement'] = True
                            except Exception as e3:
                                st.error(f"❌ Failed to save financial statements: {str(e3)}")
                                saved['income_statement'] = False
        
        # Save balance sheet
        # Note: historical_balance_sheet table only has scenario_id (no user_id)
        # Map extracted columns to database column names (matching build_historic_balance_sheet_record)
        if not bs_df.empty:
            bs_records = []
            for _, row in bs_df.iterrows():
                # Map extracted values to database column names
                current_assets = row.get('current_assets', 0)
                fixed_assets = row.get('fixed_assets', 0)
                intangible_assets = row.get('intangible_assets', 0)
                current_liabilities = row.get('current_liabilities', 0)
                long_term_liabilities = row.get('long_term_liabilities', 0)
                equity = row.get('equity', 0)
                
                # Map period to period_date, period_year, and period_month
                period = row[period_column]
                if isinstance(period, str):
                    try:
                        period_date = pd.to_datetime(period)
                    except:
                        period_date = datetime.now()
                elif hasattr(period, 'strftime'):
                    period_date = period
                elif hasattr(period, 'year'):  # Already a datetime-like object
                    period_date = period
                else:
                    period_date = datetime.now()
                
                # Ensure it's a datetime object
                if not hasattr(period_date, 'year'):
                    period_date = pd.to_datetime(period_date)
                
                # Format period_date as YYYY-MM-DD (first of month)
                period_date_str = period_date.replace(day=1).strftime('%Y-%m-%d') if hasattr(period_date, 'replace') else pd.to_datetime(period_date).replace(day=1).strftime('%Y-%m-%d')
                
                record = {
                    'scenario_id': scenario_id,
                    'period_date': period_date_str,
                    'period_year': period_date.year,
                    'period_month': period_date.month,
                    'is_actual': True,
                    # Current Assets - map to database columns
                    'total_current_assets': current_assets,
                    # Non-current Assets - use net_ppe (not ppe_net)
                    'net_ppe': fixed_assets,  # Property, Plant & Equipment (net)
                    'intangible_assets': intangible_assets,
                    'total_non_current_assets': fixed_assets + intangible_assets,  # Note: total_non_current_assets (with underscore)
                    'total_assets': row.get('total_assets', 0),
                    # Current Liabilities
                    'total_current_liabilities': current_liabilities,
                    # Non-current Liabilities
                    'long_term_debt': long_term_liabilities,
                    'total_non_current_liabilities': long_term_liabilities,  # Note: total_non_current_liabilities (with underscore)
                    'total_liabilities': row.get('total_liabilities', 0),
                    # Equity - split into components
                    'share_capital': equity * 0.5,  # Estimate split (can be refined)
                    'retained_earnings': equity * 0.5,  # Estimate split
                    'total_equity': equity,
                }
                
                bs_records.append(record)
            
            if bs_records:
                # Try upsert without on_conflict first, or use period_date if that's the unique constraint
                # If that fails, we'll delete and insert
                try:
                    db.client.table('historical_balance_sheet').upsert(
                        bs_records
                    ).execute()
                except Exception as e:
                    # If upsert fails, try with period_date as unique constraint
                    try:
                        db.client.table('historical_balance_sheet').upsert(
                            bs_records,
                            on_conflict='scenario_id,period_date'
                        ).execute()
                    except:
                        # Last resort: delete existing and insert
                        # Delete existing records for this scenario and period
                        for record in bs_records:
                            db.client.table('historical_balance_sheet').delete().eq(
                                'scenario_id', scenario_id
                            ).eq('period_date', record['period_date']).execute()
                        # Then insert
                        db.client.table('historical_balance_sheet').insert(bs_records).execute()
                saved['balance_sheet'] = True
        
        # Save cash flow
        # Note: historical_cashflow table only has scenario_id (no user_id)
        # Map extracted columns to database column names
        if not cf_df.empty:
            cf_records = []
            for _, row in cf_df.iterrows():
                cash_from_operations = row.get('cash_from_operations', 0)
                cash_from_investing = row.get('cash_from_investing', 0)
                cash_from_financing = row.get('cash_from_financing', 0)
                
                record = {
                    'scenario_id': scenario_id,
                    # Operating Activities
                    'cash_from_operations': cash_from_operations,
                    # Investing Activities
                    'cash_from_investing': cash_from_investing,
                    # Financing Activities
                    'cash_from_financing': cash_from_financing,
                    # Net change (calculated field - may not exist in DB, but include for completeness)
                    # Note: beginning_cash and ending_cash may not be in the schema
                    # Only include fields that exist in the database schema
                }
                
                # Map period to month format
                period = row[period_column]
                if isinstance(period, str):
                    try:
                        period_date = pd.to_datetime(period)
                    except:
                        period_date = datetime.now()
                elif hasattr(period, 'strftime'):
                    period_date = period
                else:
                    period_date = datetime.now()
                
                record['month'] = period_date.strftime('%Y-%m-%d') if hasattr(period_date, 'strftime') else str(period_date)
                cf_records.append(record)
            
            if cf_records:
                # Try upsert - if unique constraint doesn't match, delete and insert
                try:
                    db.client.table('historical_cashflow').upsert(
                        cf_records
                    ).execute()
                except Exception as e:
                    # If upsert fails, try with explicit on_conflict
                    try:
                        db.client.table('historical_cashflow').upsert(
                            cf_records,
                            on_conflict='scenario_id,month'
                        ).execute()
                    except:
                        # Last resort: delete existing and insert
                        for record in cf_records:
                            db.client.table('historical_cashflow').delete().eq(
                                'scenario_id', scenario_id
                            ).eq('month', record['month']).execute()
                        db.client.table('historical_cashflow').insert(cf_records).execute()
                saved['cash_flow'] = True
        
    except Exception as e:
        st.error(f"Error saving extracted financial statements: {e}")
        return {
            'success': False,
            'error': str(e),
            'saved': saved
        }
    
    return {
        'success': True,
        'saved': saved
    }
