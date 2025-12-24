"""
Trial Balance Account Mapper
============================
UI component for mapping trial balance accounts to financial statement categories
and selecting debit/credit calculation method.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime


# Financial statement categories
INCOME_STATEMENT_CATEGORIES = [
    'revenue',
    'cogs',
    'opex',
    'depreciation',
    'other_income',
    'other_expense',
    'tax'
]

BALANCE_SHEET_CATEGORIES = [
    'current_assets',
    'fixed_assets',
    'intangible_assets',
    'current_liabilities',
    'long_term_liabilities',
    'equity'
]

ALL_CATEGORIES = INCOME_STATEMENT_CATEGORIES + BALANCE_SHEET_CATEGORIES + ['other', 'exclude']


def render_account_mapper(
    tb_df: pd.DataFrame,
    period_column: str,
    account_column: str,
    account_code_column: Optional[str],
    scenario_id: str
) -> Dict[str, Dict[str, Any]]:
    """
    Render UI for mapping accounts to categories and selecting debit/credit.
    
    Args:
        tb_df: Trial balance DataFrame
        period_column: Name of period column
        account_column: Name of account name column
        account_code_column: Name of account code column (optional)
        scenario_id: Scenario ID for saving mappings
    
    Returns:
        Dictionary mapping account_name (or account_code) to:
        {
            'category': str,
            'statement': 'income_statement' | 'balance_sheet',
            'use_debit': bool,  # True to use debit, False to use credit
            'multiplier': float  # 1.0 or -1.0 to flip sign if needed
        }
    """
    if tb_df.empty:
        st.warning("No trial balance data to map")
        return {}
    
    st.markdown("### 📋 Account Mapping & Classification")
    st.caption("Map each account to a financial statement category and select calculation method")
    
    # Get unique accounts
    if account_code_column and account_code_column in tb_df.columns:
        # Group by account code and name
        accounts = tb_df[[account_column, account_code_column]].drop_duplicates()
        accounts = accounts.sort_values([account_code_column, account_column])
        key_column = account_code_column
        display_cols = [account_code_column, account_column]
    else:
        accounts = tb_df[[account_column]].drop_duplicates()
        accounts = accounts.sort_values(account_column)
        key_column = account_column
        display_cols = [account_column]
    
    # Load saved mappings from session state
    mapping_key = f'trial_balance_mappings_{scenario_id}'
    if mapping_key not in st.session_state:
        st.session_state[mapping_key] = {}
    
    saved_mappings = st.session_state[mapping_key]
    
    # Create mapping interface
    st.markdown("#### Account Classifications")
    
    # Show summary of current mappings
    if saved_mappings:
        mapped_count = len(saved_mappings)
        total_count = len(accounts)
        st.info(f"📊 {mapped_count} of {total_count} accounts mapped")
    
    # AI-powered classification
    with st.expander("🤖 AI-Powered Classification", expanded=True):
        st.markdown("**Let AI classify accounts based on accounting standards:**")
        st.caption("Uses LLM to intelligently classify accounts by analyzing names, codes, and balance patterns")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 Classify All with AI", type="primary", use_container_width=True):
                try:
                    from components.ai_account_classifier import classify_accounts_with_ai
                    
                    debit_col = 'debit' if 'debit' in tb_df.columns else 'dr'
                    credit_col = 'credit' if 'credit' in tb_df.columns else 'cr'
                    
                    ai_classifications = classify_accounts_with_ai(
                        tb_df,
                        account_column,  # Use the function parameter name
                        account_code_column,  # Use the function parameter name
                        debit_col,
                        credit_col
                    )
                    
                    if ai_classifications:
                        # Apply AI classifications to saved_mappings
                        applied_count = 0
                        for account_key, classification in ai_classifications.items():
                            # Find the account in accounts DataFrame
                            # Try by account code first, then by name
                            matching = None
                            if account_code_column and account_code_column in accounts.columns:
                                # Try matching by code
                                matching = accounts[accounts[account_code_column].astype(str) == str(account_key)]
                                if matching.empty:
                                    # Try matching by name
                                    matching = accounts[accounts[account_column].astype(str) == str(account_key)]
                            else:
                                # Match by name only
                                matching = accounts[accounts[account_column].astype(str) == str(account_key)]
                            
                            if not matching.empty:
                                row = matching.iloc[0]
                                # Use account_code as key if available, otherwise account_name
                                map_key = row.get(account_code_column) if account_code_column and account_code_column in row else row[account_column]
                                
                                saved_mappings[map_key] = {
                                    'category': classification['category'],
                                    'statement': classification['statement'],
                                    'use_debit': classification['use_debit'],
                                    'account_name': row[account_column],
                                    'account_code': row.get(account_code_column) if account_code_column else None,
                                    'confidence': classification.get('confidence', 0.8),
                                    'reasoning': classification.get('reasoning', 'AI classified')
                                }
                                applied_count += 1
                        
                        st.session_state[mapping_key] = saved_mappings
                        if applied_count > 0:
                            st.success(f"✅ AI classified {applied_count} accounts!")
                            
                            # Show summary of classifications
                            with st.expander("📊 AI Classification Summary", expanded=True):
                                summary_data = []
                                for key, mapping in saved_mappings.items():
                                    if 'reasoning' in mapping and 'AI' in mapping.get('reasoning', ''):
                                        summary_data.append({
                                            'Account': mapping.get('account_name', key),
                                            'Category': mapping.get('category', ''),
                                            'Statement': mapping.get('statement', ''),
                                            'Confidence': f"{mapping.get('confidence', 0):.1%}",
                                            'Reasoning': mapping.get('reasoning', '')[:100]
                                        })
                                
                                if summary_data:
                                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
                        else:
                            st.warning("⚠️ No accounts were matched. Check account codes/names.")
                        st.rerun()
                    else:
                        st.warning("⚠️ AI classification returned no results. Check API keys in secrets.toml")
                except Exception as e:
                    st.error(f"❌ AI classification failed: {str(e)}")
                    import traceback
                    error_trace = traceback.format_exc()
                    st.error(f"Full error: {error_trace}")
                    with st.expander("🔍 Error Details"):
                        st.code(error_trace)
                    
                    # Show available variables for debugging
                    with st.expander("🔍 Debug Info"):
                        st.write(f"account_column: {account_column}")
                        st.write(f"account_code_column: {account_code_column}")
                        st.write(f"tb_df columns: {list(tb_df.columns)}")
        
        with col2:
            st.info("""
            **Requirements:**
            - OpenAI API key in `secrets.toml` under `[openai]` → `api_key`
            - OR Anthropic API key under `[anthropic]` → `api_key`
            
            The AI will analyze account names, codes, and balance patterns to classify accounts according to standard accounting principles.
            """)
    
    # Pattern-based quick mapping (fallback)
    with st.expander("🔍 Quick Map by Pattern", expanded=False):
        st.markdown("**Auto-map accounts using patterns (fallback):**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Auto-map Revenue", use_container_width=True):
                _auto_map_by_pattern(accounts, account_column, account_code_column, 'revenue', saved_mappings)
                st.rerun()
            if st.button("Auto-map COGS", use_container_width=True):
                _auto_map_by_pattern(accounts, account_column, account_code_column, 'cogs', saved_mappings)
                st.rerun()
            if st.button("Auto-map OPEX", use_container_width=True):
                _auto_map_by_pattern(accounts, account_column, account_code_column, 'opex', saved_mappings)
                st.rerun()
        with col2:
            if st.button("Auto-map Assets", use_container_width=True):
                _auto_map_by_pattern(accounts, account_column, account_code_column, 'current_assets', saved_mappings)
                st.rerun()
            if st.button("Auto-map Liabilities", use_container_width=True):
                _auto_map_by_pattern(accounts, account_column, account_code_column, 'current_liabilities', saved_mappings)
                st.rerun()
            if st.button("Clear All Mappings", use_container_width=True):
                saved_mappings.clear()
                st.rerun()
    
    st.markdown("---")
    
    # Show accounts in a table with mapping controls
    st.markdown("#### Map Individual Accounts")
    
    # Create a DataFrame for editing
    mapping_data = []
    for idx, row in accounts.iterrows():
        account_name = row[account_column]
        account_code = row.get(account_code_column) if account_code_column else None
        key = account_code if account_code else account_name
        
        # Get saved mapping or defaults
        mapping = saved_mappings.get(key, {})
        
        # Calculate sample balance to help user decide
        account_rows = tb_df[tb_df[account_column] == account_name]
        if account_code_column and account_code:
            account_rows = account_rows[account_rows[account_code_column] == account_code]
        
        total_debit = account_rows[account_rows.columns[account_rows.columns.str.contains('debit|dr', case=False, na=False)]].iloc[:, 0].sum() if len(account_rows) > 0 else 0
        total_credit = account_rows[account_rows.columns[account_rows.columns.str.contains('credit|cr', case=False, na=False)]].iloc[:, 0].sum() if len(account_rows) > 0 else 0
        
        mapping_data.append({
            'Account Code': account_code if account_code else '',
            'Account Name': account_name,
            'Category': mapping.get('category', 'other'),
            'Statement': mapping.get('statement', 'income_statement'),
            'Use Debit': mapping.get('use_debit', True),
            'Confidence': f"{mapping.get('confidence', 0.8):.1%}" if 'confidence' in mapping else '',
            'Reasoning': mapping.get('reasoning', '')[:50] if 'reasoning' in mapping else '',  # Truncate for display
            'Sample Debit': f"{total_debit:,.2f}",
            'Sample Credit': f"{total_credit:,.2f}",
            '_key': key
        })
    
    mapping_df = pd.DataFrame(mapping_data)
    
    # Use data editor for mapping
    display_cols = ['Account Code', 'Account Name', 'Category', 'Statement', 'Use Debit']
    if 'Confidence' in mapping_df.columns and mapping_df['Confidence'].notna().any():
        display_cols.append('Confidence')
    if 'Reasoning' in mapping_df.columns and mapping_df['Reasoning'].notna().any():
        display_cols.append('Reasoning')
    display_cols.extend(['Sample Debit', 'Sample Credit'])
    
    edited_df = st.data_editor(
        mapping_df[display_cols],
        column_config={
            'Account Code': st.column_config.TextColumn('Account Code', disabled=True),
            'Account Name': st.column_config.TextColumn('Account Name', disabled=True),
            'Category': st.column_config.SelectboxColumn(
                'Category',
                options=ALL_CATEGORIES,
                required=True
            ),
            'Statement': st.column_config.SelectboxColumn(
                'Statement',
                options=['income_statement', 'balance_sheet', 'exclude'],
                required=True
            ),
            'Use Debit': st.column_config.CheckboxColumn('Use Debit'),
            'Confidence': st.column_config.TextColumn('Confidence', disabled=True, help="AI confidence score"),
            'Reasoning': st.column_config.TextColumn('Reasoning', disabled=True, help="AI classification reasoning"),
            'Sample Debit': st.column_config.TextColumn('Sample Debit', disabled=True),
            'Sample Credit': st.column_config.TextColumn('Sample Credit', disabled=True),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key=f"account_mapper_{scenario_id}"
    )
    
    # Save mappings
    if st.button("💾 Save Mappings", type="primary", use_container_width=True):
        for idx, row in edited_df.iterrows():
            key = mapping_df.iloc[idx]['_key']
            saved_mappings[key] = {
                'category': row['Category'],
                'statement': row['Statement'],
                'use_debit': row['Use Debit'],
                'account_name': row['Account Name'],
                'account_code': row['Account Code'] if 'Account Code' in row else None
            }
        
        st.session_state[mapping_key] = saved_mappings
        st.success(f"✅ Saved mappings for {len(saved_mappings)} accounts")
        st.rerun()
    
    return saved_mappings


def _auto_map_by_pattern(
    accounts: pd.DataFrame,
    account_column: str,
    account_code_column: Optional[str],
    category: str,
    saved_mappings: Dict[str, Dict[str, Any]]
):
    """Auto-map accounts based on patterns."""
    import re
    
    patterns = {
        'revenue': [r'revenue', r'sales', r'income', r'turnover', r'4\d{3}'],
        'cogs': [r'cost.*sales', r'cogs', r'cost.*goods', r'direct.*cost', r'5\d{3}'],
        'opex': [r'operating.*expense', r'opex', r'admin', r'overhead', r'salaries', r'6\d{3}'],
        'current_assets': [r'cash', r'bank', r'debtors', r'receivables', r'inventory', r'1\d{3}'],
        'current_liabilities': [r'creditors', r'payables', r'short.*term.*debt', r'3\d{3}'],
    }
    
    pattern_list = patterns.get(category, [])
    statement = 'income_statement' if category in ['revenue', 'cogs', 'opex'] else 'balance_sheet'
    
    for idx, row in accounts.iterrows():
        account_name = str(row[account_column]).lower()
        account_code = str(row.get(account_code_column, '')).lower() if account_code_column else ''
        key = row.get(account_code_column) if account_code_column else row[account_column]
        
        # Check if matches pattern
        matches = False
        for pattern in pattern_list:
            if re.search(pattern, account_name, re.IGNORECASE) or (account_code and re.match(pattern, account_code)):
                matches = True
                break
        
        if matches:
            saved_mappings[key] = {
                'category': category,
                'statement': statement,
                'use_debit': category in ['cogs', 'opex', 'current_assets', 'fixed_assets'],  # Debit normal
                'account_name': row[account_column],
                'account_code': row.get(account_code_column) if account_code_column else None
            }


def apply_account_mappings(
    tb_df: pd.DataFrame,
    mappings: Dict[str, Dict[str, Any]],
    period_column: str,
    account_column: str,
    account_code_column: Optional[str],
    debit_column: str,
    credit_column: str
) -> Dict[str, pd.DataFrame]:
    """
    Apply user-defined account mappings to extract financial statements.
    
    Returns:
        Dictionary with 'income_statement', 'balance_sheet', 'cash_flow' DataFrames
    """
    from components.trial_balance_processor import TrialBalanceProcessor
    
    # Create classifications DataFrame from mappings
    classifications = []
    for key, mapping in mappings.items():
        if mapping.get('statement') == 'exclude':
            continue
        
        # Find matching rows in trial balance
        if account_code_column and account_code_column in tb_df.columns:
            mask = tb_df[account_code_column] == key
        else:
            mask = tb_df[account_column] == key
        
        if not mask.any():
            # Try by account name if code didn't match
            mask = tb_df[account_column] == mapping.get('account_name', key)
        
        if mask.any():
            classifications.append({
                'account_name': mapping.get('account_name', key),
                'account_code': mapping.get('account_code'),
                'statement': mapping['statement'],
                'category': mapping['category'],
                'use_debit': mapping.get('use_debit', True),
                'confidence': 1.0  # User-defined, so 100% confidence
            })
    
    if not classifications:
        return {
            'income_statement': pd.DataFrame(),
            'balance_sheet': pd.DataFrame(),
            'cash_flow': pd.DataFrame()
        }
    
    classifications_df = pd.DataFrame(classifications)
    
    # Process using custom logic with mappings
    return _extract_with_mappings(
        tb_df,
        classifications_df,
        period_column,
        account_column,
        account_code_column,
        debit_column,
        credit_column
    )


def _extract_with_mappings(
    tb_df: pd.DataFrame,
    classifications_df: pd.DataFrame,
    period_column: str,
    account_column: str,
    account_code_column: Optional[str],
    debit_column: str,
    credit_column: str
) -> Dict[str, pd.DataFrame]:
    """Extract financial statements using user-defined mappings."""
    
    # Merge classifications with trial balance
    merge_cols = [account_column]
    if account_code_column and account_code_column in tb_df.columns:
        merge_cols.append(account_code_column)
    
    tb_df = tb_df.copy()
    # Convert to numeric, handling various formats
    tb_df['debit'] = pd.to_numeric(tb_df[debit_column], errors='coerce').fillna(0)
    tb_df['credit'] = pd.to_numeric(tb_df[credit_column], errors='coerce').fillna(0)
    
    # Normalize: Ensure all debit and credit values are positive
    # Some accounting systems store credits as negative, or use different conventions
    # We'll use absolute values and rely on which column has the value to determine the sign
    tb_df['debit'] = tb_df['debit'].abs()
    tb_df['credit'] = tb_df['credit'].abs()
    
    # Merge classifications
    tb_classified = tb_df.merge(classifications_df, on=merge_cols, how='inner')
    
    # Extract Income Statement
    is_data = []
    is_accounts = tb_classified[tb_classified['statement'] == 'income_statement']
    
    for period in tb_classified[period_column].unique():
        period_data = is_accounts[is_accounts[period_column] == period]
        
        revenue = 0
        cogs = 0
        opex = 0
        depreciation = 0
        other_income = 0
        other_expense = 0
        tax = 0
        
        for _, row in period_data.iterrows():
            category = row['category']
            # Values are already normalized to positive in the DataFrame
            debit = float(row['debit'] or 0)
            credit = float(row['credit'] or 0)
            
            # INCOME STATEMENT CALCULATION RULES (based on accounting standards):
            # Revenue/Income: ALWAYS credit normal → amount = credit - debit
            #   - Normal case: credit > debit → positive revenue ✓
            #   - Unusual case: debit > credit → negative revenue (returns exceed sales)
            # Expenses: ALWAYS debit normal → amount = debit - credit
            #   - Normal case: debit > credit → positive expense ✓
            #   - Unusual case: credit > debit → negative expense (credits/refunds)
            
            if category in ['revenue', 'other_income']:
                # Revenue/Income: ALWAYS use credit - debit (credit normal)
                # Both values are positive, so calculation is straightforward
                amount = credit - debit
                if category == 'revenue':
                    revenue += amount
                else:
                    other_income += amount
                    
            elif category in ['cogs', 'opex', 'depreciation', 'other_expense', 'tax']:
                # Expenses: ALWAYS use debit - credit (debit normal, regardless of use_debit flag)
                amount = debit - credit
                
                if category == 'cogs':
                    cogs += amount
                elif category == 'opex':
                    opex += amount
                elif category == 'depreciation':
                    depreciation += amount
                elif category == 'other_expense':
                    other_expense += amount
                elif category == 'tax':
                    tax += amount
        
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
    
    is_df = pd.DataFrame(is_data)
    
    # Extract Balance Sheet (similar logic)
    bs_data = []
    bs_accounts = tb_classified[tb_classified['statement'] == 'balance_sheet']
    
    for period in tb_classified[period_column].unique():
        period_data = bs_accounts[bs_accounts[period_column] == period]
        
        current_assets = 0
        fixed_assets = 0
        intangible_assets = 0
        current_liabilities = 0
        long_term_liabilities = 0
        equity = 0
        
        for _, row in period_data.iterrows():
            category = row['category']
            debit = float(row['debit'] or 0)
            credit = float(row['credit'] or 0)
            
            # BALANCE SHEET CALCULATION RULES (based on accounting standards):
            # Assets: ALWAYS debit normal → amount = debit - credit
            #   - If debit > credit: positive asset (normal case)
            #   - If credit > debit: negative asset (contra account, e.g., accumulated depreciation)
            # Liabilities/Equity: ALWAYS credit normal → amount = credit - debit
            #   - If credit > debit: positive liability/equity (normal case)
            #   - If debit > credit: negative liability/equity (unusual but possible)
            # All should result in POSITIVE values in the balance sheet (except contra accounts)
            
            if category in ['current_assets', 'fixed_assets', 'intangible_assets']:
                # Assets: ALWAYS use debit - credit (debit normal, regardless of use_debit flag)
                amount = debit - credit
                
                if category == 'current_assets':
                    current_assets += amount
                elif category == 'fixed_assets':
                    fixed_assets += amount
                elif category == 'intangible_assets':
                    intangible_assets += amount
                    
            elif category in ['current_liabilities', 'long_term_liabilities', 'equity']:
                # Liabilities/Equity: ALWAYS use credit - debit (credit normal, regardless of use_debit flag)
                amount = credit - debit
                
                if category == 'current_liabilities':
                    current_liabilities += amount
                elif category == 'long_term_liabilities':
                    long_term_liabilities += amount
                elif category == 'equity':
                    equity += amount
        
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
    
    bs_df = pd.DataFrame(bs_data)
    
    # Calculate Cash Flow (same as before)
    from components.trial_balance_processor import TrialBalanceProcessor
    processor = TrialBalanceProcessor()
    cf_df = processor._calculate_cash_flow(is_df, bs_df, period_column)
    
    return {
        'income_statement': is_df,
        'balance_sheet': bs_df,
        'cash_flow': cf_df
    }
