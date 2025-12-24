"""
AI Account Classifier
=====================
Uses LLM to intelligently classify trial balance accounts based on accounting standards,
account names, codes, and balance patterns.
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Optional, Any, Tuple
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

ALL_CATEGORIES = INCOME_STATEMENT_CATEGORIES + BALANCE_SHEET_CATEGORIES


class AIAccountClassifier:
    """
    Uses LLM to classify accounts based on accounting standards.
    """
    
    def __init__(self):
        """Initialize LLM clients."""
        self.openai_client = None
        self.anthropic_client = None
        self._init_llm_clients()
    
    def _init_llm_clients(self):
        """Initialize OpenAI and Anthropic clients."""
        try:
            import openai
            if 'openai' in st.secrets and 'api_key' in st.secrets['openai']:
                self.openai_client = openai.OpenAI(
                    api_key=st.secrets['openai']['api_key']
                )
        except (ImportError, KeyError, Exception):
            pass
        
        try:
            import anthropic
            if 'anthropic' in st.secrets and 'api_key' in st.secrets['anthropic']:
                self.anthropic_client = anthropic.Anthropic(
                    api_key=st.secrets['anthropic']['api_key']
                )
        except (ImportError, KeyError, Exception):
            pass
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.openai_client is not None or self.anthropic_client is not None
    
    def classify_accounts_batch(
        self,
        accounts: List[Dict[str, Any]],
        sample_balances: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Classify multiple accounts using AI.
        
        Args:
            accounts: List of account dicts with 'account_name', 'account_code' (optional)
            sample_balances: Optional dict mapping account_key to {'debit': float, 'credit': float}
        
        Returns:
            Dictionary mapping account_key to classification:
            {
                'category': str,
                'statement': 'income_statement' | 'balance_sheet',
                'use_debit': bool,
                'confidence': float,
                'reasoning': str
            }
        """
        if not self.is_available():
            raise ValueError("LLM not available. Please configure API keys in secrets.toml")
        
        # Prepare account data for LLM
        account_data = []
        for acc in accounts:
            account_key = acc.get('account_code') or acc.get('account_name')
            balance_info = sample_balances.get(account_key, {}) if sample_balances else {}
            
            account_data.append({
                'account_code': acc.get('account_code', ''),
                'account_name': acc.get('account_name', ''),
                'sample_debit': balance_info.get('debit', 0),
                'sample_credit': balance_info.get('credit', 0),
                'net_balance': balance_info.get('debit', 0) - balance_info.get('credit', 0)
            })
        
        # Build prompt
        prompt = self._build_classification_prompt(account_data)
        
        # Call LLM
        try:
            if self.openai_client:
                response = self._classify_with_openai(prompt, account_data)
            elif self.anthropic_client:
                response = self._classify_with_anthropic(prompt, account_data)
            else:
                raise ValueError("No LLM client available")
            
            return response
        except Exception as e:
            st.warning(f"AI classification failed: {str(e)}. Falling back to rule-based classification.")
            return self._fallback_classification(accounts, sample_balances)
    
    def _build_classification_prompt(self, account_data: List[Dict]) -> str:
        """Build prompt for LLM classification."""
        accounts_text = "\n".join([
            f"Code: {acc['account_code']}, Name: {acc['account_name']}, "
            f"Debit: {acc['sample_debit']:,.2f}, Credit: {acc['sample_credit']:,.2f}, "
            f"Net: {acc['net_balance']:,.2f}"
            for acc in account_data
        ])
        
        return f"""You are an expert accountant analyzing a trial balance. Classify each account according to standard accounting principles (IFRS/GAAP).

ACCOUNTING STANDARDS - NORMAL BALANCES:
- Revenue accounts: CREDIT normal (increases are credits) → use_debit=False
- Expense accounts: DEBIT normal (increases are debits) → use_debit=True
- Asset accounts: DEBIT normal (increases are debits) → use_debit=True
- Liability accounts: CREDIT normal (increases are credits) → use_debit=False
- Equity accounts: CREDIT normal (increases are credits) → use_debit=False

CATEGORIES:
Income Statement (statement: "income_statement"):
- revenue: Sales revenue, service income, fees, turnover, operating income
- cogs: Cost of goods sold, cost of sales, direct costs, cost of revenue
- opex: Operating expenses, overheads, admin expenses, salaries, wages, rent, utilities, insurance, marketing, general & administrative
- depreciation: Depreciation expense, amortization expense
- other_income: Interest income, dividend income, other operating income
- other_expense: Interest expense, finance costs, other operating expenses
- tax: Income tax expense, corporation tax, tax provision

Balance Sheet (statement: "balance_sheet"):
- current_assets: Cash, bank, cash equivalents, trade debtors, accounts receivable, inventory, stock, prepaid expenses
- fixed_assets: Property plant & equipment, PPE, machinery, vehicles, buildings, accumulated depreciation (negative asset)
- intangible_assets: Goodwill, patents, trademarks, software, intangible assets
- current_liabilities: Trade creditors, accounts payable, short-term debt, overdraft, accruals, current portion of long-term debt
- long_term_liabilities: Long-term debt, term loans, bonds payable, non-current liabilities
- equity: Share capital, retained earnings, reserves, equity, capital

ACCOUNTS TO CLASSIFY:
{accounts_text}

INSTRUCTIONS:
1. Analyze each account's name and code to determine its purpose
2. Check the balance pattern (debit vs credit) to confirm normal balance
3. Classify into the most specific category above
4. Set use_debit based on normal balance rules:
   - Revenue/Income: use_debit=False (credit normal)
   - Expenses: use_debit=True (debit normal)
   - Assets: use_debit=True (debit normal)
   - Liabilities/Equity: use_debit=False (credit normal)
5. Provide confidence (0.0-1.0) based on how certain you are
6. Give brief reasoning (1-2 sentences)

IMPORTANT: The use_debit flag indicates which side (debit or credit) represents the normal/positive balance for that account type.

Return a JSON object where each key is the account_code (or account_name if no code) and value is:
{{
    "category": "exact_category_name_from_list",
    "statement": "income_statement" or "balance_sheet",
    "use_debit": true or false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of classification"
}}

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks, no explanations outside JSON."""
    
    def _classify_with_openai(self, prompt: str, account_data: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """Classify using OpenAI."""
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Use cost-effective model
            messages=[
                {"role": "system", "content": "You are an expert accountant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent classification
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        return self._parse_llm_response(result_text, account_data)
    
    def _classify_with_anthropic(self, prompt: str, account_data: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """Classify using Anthropic."""
        response = self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        result_text = response.content[0].text
        return self._parse_llm_response(result_text, account_data)
    
    def _parse_llm_response(self, response_text: str, account_data: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """Parse LLM response into classifications."""
        # Clean response (remove markdown code blocks if present)
        response_text = response_text.strip()
        if response_text.startswith("```"):
            # Remove code block markers
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
        
        try:
            classifications = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                classifications = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse LLM response as JSON")
        
        # Convert to our format
        result = {}
        
        # If classifications is a single dict with account keys, use it directly
        # If it's a list or has nested structure, handle it
        if isinstance(classifications, list):
            # Convert list to dict
            classifications = {item.get('account_code') or item.get('account_name'): item for item in classifications if isinstance(item, dict)}
        
        for account in account_data:
            account_code = str(account.get('account_code', '')).strip()
            account_name = str(account.get('account_name', '')).strip()
            account_key = account_code if account_code else account_name
            
            # Try multiple keys to find classification
            classification = None
            
            # Strategy 1: Direct key match
            for key in [account_code, account_name, account_key, str(account_code), str(account_name)]:
                if key and key in classifications:
                    classification = classifications[key]
                    break
            
            # Strategy 2: Partial match (account name in key or vice versa)
            if not classification:
                account_name_lower = account_name.lower()
                account_code_lower = account_code.lower()
                for key, value in classifications.items():
                    if isinstance(value, dict):
                        key_lower = str(key).lower()
                        if (account_name_lower and account_name_lower in key_lower) or \
                           (account_code_lower and account_code_lower in key_lower) or \
                           (key_lower in account_name_lower):
                            classification = value
                            break
            
            # Strategy 3: Check if value contains account info
            if not classification:
                for key, value in classifications.items():
                    if isinstance(value, dict):
                        # Check if value has account_name or account_code that matches
                        if value.get('account_name') == account_name or value.get('account_code') == account_code:
                            classification = value
                            break
            
            if classification and isinstance(classification, dict):
                # Validate category
                category = classification.get('category', 'other')
                if category not in ALL_CATEGORIES:
                    # Try to map similar categories
                    category_lower = category.lower()
                    if 'revenue' in category_lower or 'sales' in category_lower or 'income' in category_lower:
                        category = 'revenue'
                    elif 'cost' in category_lower or 'cogs' in category_lower:
                        category = 'cogs'
                    elif 'expense' in category_lower or 'opex' in category_lower or 'overhead' in category_lower:
                        category = 'opex'
                    elif 'depreciation' in category_lower or 'amortization' in category_lower:
                        category = 'depreciation'
                    elif 'asset' in category_lower:
                        if 'current' in category_lower or 'cash' in category_lower or 'receivable' in category_lower:
                            category = 'current_assets'
                        elif 'fixed' in category_lower or 'ppe' in category_lower or 'property' in category_lower:
                            category = 'fixed_assets'
                        elif 'intangible' in category_lower:
                            category = 'intangible_assets'
                    elif 'liability' in category_lower or 'payable' in category_lower:
                        if 'current' in category_lower or 'short' in category_lower:
                            category = 'current_liabilities'
                        else:
                            category = 'long_term_liabilities'
                    elif 'equity' in category_lower or 'capital' in category_lower:
                        category = 'equity'
                    else:
                        category = 'other'
                
                result[account_key] = {
                    'category': category,
                    'statement': classification.get('statement', 'income_statement' if category in INCOME_STATEMENT_CATEGORIES else 'balance_sheet'),
                    'use_debit': bool(classification.get('use_debit', True)),
                    'confidence': float(classification.get('confidence', 0.8)),
                    'reasoning': str(classification.get('reasoning', 'AI classified'))[:200]  # Limit length
                }
            else:
                # Fallback to rule-based
                from components.trial_balance_processor import TrialBalanceProcessor
                processor = TrialBalanceProcessor()
                fallback_class = processor.classify_account(account_name, account_code if account_code else None)
                
                result[account_key] = {
                    'category': fallback_class['category'],
                    'statement': fallback_class['statement'],
                    'use_debit': fallback_class['category'] not in ['revenue', 'other_income', 'current_liabilities', 'long_term_liabilities', 'equity'],
                    'confidence': fallback_class['confidence'] * 0.6,  # Lower confidence for fallback
                    'reasoning': f'Fallback classification (not found in AI response): {fallback_class["category"]}'
                }
        
        return result
    
    def _fallback_classification(
        self,
        accounts: List[Dict[str, Any]],
        sample_balances: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Fallback to rule-based classification if LLM fails."""
        from components.trial_balance_processor import TrialBalanceProcessor
        
        processor = TrialBalanceProcessor()
        result = {}
        
        for acc in accounts:
            account_name = acc.get('account_name', '')
            account_code = acc.get('account_code')
            account_key = account_code or account_name
            
            classification = processor.classify_account(account_name, account_code)
            
            # Determine use_debit based on category and balance
            balance_info = sample_balances.get(account_key, {}) if sample_balances else {}
            debit_bal = balance_info.get('debit', 0)
            credit_bal = balance_info.get('credit', 0)
            
            # Default use_debit based on category
            if classification['category'] in ['revenue', 'other_income']:
                use_debit = False  # Credit normal
            elif classification['category'] in ['cogs', 'opex', 'depreciation', 'other_expense', 'tax']:
                use_debit = True  # Debit normal
            elif classification['category'] in ['current_assets', 'fixed_assets', 'intangible_assets']:
                use_debit = True  # Debit normal
            elif classification['category'] in ['current_liabilities', 'long_term_liabilities', 'equity']:
                use_debit = False  # Credit normal
            else:
                # Infer from balance
                use_debit = debit_bal > credit_bal
            
            result[account_key] = {
                'category': classification['category'],
                'statement': classification['statement'],
                'use_debit': use_debit,
                'confidence': classification['confidence'] * 0.7,  # Lower confidence for fallback
                'reasoning': f'Rule-based: {classification["category"]}'
            }
        
        return result


def classify_accounts_with_ai(
    tb_df: pd.DataFrame,
    account_column: str,
    account_code_column: Optional[str],
    debit_column: str,
    credit_column: str,
    max_accounts: int = 100
) -> Dict[str, Dict[str, Any]]:
    """
    Classify accounts using AI.
    
    Args:
        tb_df: Trial balance DataFrame
        account_column: Name of account name column
        account_code_column: Name of account code column (optional)
        debit_column: Name of debit column
        credit_column: Name of credit column
        max_accounts: Maximum number of accounts to classify in one batch
    
    Returns:
        Dictionary mapping account_key to classification
    """
    classifier = AIAccountClassifier()
    
    if not classifier.is_available():
        st.warning("⚠️ LLM not configured. Please add OpenAI or Anthropic API keys to secrets.toml")
        return {}
    
    # Get unique accounts
    if account_code_column and account_code_column in tb_df.columns:
        accounts_df = tb_df[[account_column, account_code_column]].drop_duplicates()
        key_column = account_code_column
    else:
        accounts_df = tb_df[[account_column]].drop_duplicates()
        key_column = account_column
    
    # Limit accounts if too many (process in batches)
    total_accounts = len(accounts_df)
    if total_accounts > max_accounts:
        st.info(f"📊 Processing {total_accounts} accounts in batches of {max_accounts}...")
        # Process in batches
        all_classifications = {}
        for i in range(0, total_accounts, max_accounts):
            batch_df = accounts_df.iloc[i:i+max_accounts]
            batch_accounts = []
            batch_balances = {}
            
            for idx, row in batch_df.iterrows():
                account_name = row[account_column]
                account_code = row.get(account_code_column) if account_code_column else None
                account_key = account_code or account_name
                
                account_rows = tb_df[tb_df[account_column] == account_name]
                if account_code_column and account_code:
                    account_rows = account_rows[account_rows[account_code_column] == account_code]
                
                total_debit = pd.to_numeric(account_rows[debit_column], errors='coerce').fillna(0).sum()
                total_credit = pd.to_numeric(account_rows[credit_column], errors='coerce').fillna(0).sum()
                
                batch_accounts.append({
                    'account_name': account_name,
                    'account_code': account_code
                })
                
                batch_balances[account_key] = {
                    'debit': float(total_debit),
                    'credit': float(total_credit)
                }
            
            with st.spinner(f"🤖 AI analyzing batch {i//max_accounts + 1} ({len(batch_accounts)} accounts)..."):
                batch_classifications = classifier.classify_accounts_batch(batch_accounts, batch_balances)
                all_classifications.update(batch_classifications)
        
        return all_classifications
    
    # Single batch processing
    
    # Prepare account list
    accounts = []
    sample_balances = {}
    
    for idx, row in accounts_df.iterrows():
        account_name = row[account_column]
        account_code = row.get(account_code_column) if account_code_column else None
        account_key = account_code or account_name
        
        # Calculate sample balances
        account_rows = tb_df[tb_df[account_column] == account_name]
        if account_code_column and account_code:
            account_rows = account_rows[account_rows[account_code_column] == account_code]
        
        total_debit = pd.to_numeric(account_rows[debit_column], errors='coerce').fillna(0).sum()
        total_credit = pd.to_numeric(account_rows[credit_column], errors='coerce').fillna(0).sum()
        
        accounts.append({
            'account_name': account_name,
            'account_code': account_code
        })
        
        sample_balances[account_key] = {
            'debit': float(total_debit),
            'credit': float(total_credit)
        }
    
    # Classify with AI
    with st.spinner(f"🤖 AI is analyzing {len(accounts)} accounts..."):
        classifications = classifier.classify_accounts_batch(accounts, sample_balances)
    
    return classifications
