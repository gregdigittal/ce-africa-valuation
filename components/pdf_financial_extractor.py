"""
PDF Financial Statement Extractor
Extracts Income Statement, Balance Sheet, and Cash Flow from PDF files.
Supports single PDF files containing multiple financial statements.
"""
import streamlit as st
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import io

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    st.warning("pdfplumber not installed. Install with: pip install pdfplumber")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


class PDFFinancialExtractor:
    """Extract financial statements from PDF files."""
    
    def __init__(self):
        self.income_statement_keywords = [
            'revenue', 'sales', 'income', 'turnover', 'cogs', 'cost of goods sold',
            'gross profit', 'operating expenses', 'opex', 'ebitda', 'ebit',
            'depreciation', 'amortization', 'tax', 'net income', 'net profit',
            'other income', 'other expense', 'interest expense', 'interest income'
        ]
        
        self.balance_sheet_keywords = [
            'assets', 'liabilities', 'equity', 'cash', 'accounts receivable',
            'inventory', 'ppe', 'property plant equipment', 'current assets',
            'fixed assets', 'current liabilities', 'long term debt', 'share capital',
            'retained earnings', 'accounts payable'
        ]
        
        self.cash_flow_keywords = [
            'cash flow', 'cash from operations', 'cash from investing',
            'cash from financing', 'operating activities', 'investing activities',
            'financing activities', 'net cash', 'cash and cash equivalents'
        ]
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file."""
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    return text
            except Exception as e:
                st.warning(f"pdfplumber extraction failed: {e}. Trying PyPDF2...")
        
        if PYPDF2_AVAILABLE:
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                return text
            except Exception as e:
                st.error(f"PyPDF2 extraction failed: {e}")
                return ""
        
        st.error("No PDF parsing library available. Please install pdfplumber or PyPDF2.")
        return ""
    
    def extract_tables_from_pdf(self, pdf_file) -> List[pd.DataFrame]:
        """Extract tables from PDF file."""
        tables = []
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        page_tables = page.extract_tables()
                        if page_tables:
                            for table in page_tables:
                                if table and len(table) > 1:
                                    # Convert to DataFrame
                                    df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                                    tables.append(df)
            except Exception as e:
                st.warning(f"Table extraction failed: {e}")
        return tables
    
    def identify_statement_type(self, text: str) -> str:
        """Identify which financial statement this is based on keywords."""
        text_lower = text.lower()
        
        is_score = sum(1 for kw in self.income_statement_keywords if kw in text_lower)
        bs_score = sum(1 for kw in self.balance_sheet_keywords if kw in text_lower)
        cf_score = sum(1 for kw in self.cash_flow_keywords if kw in text_lower)
        
        # Also check for explicit statement names
        if any(term in text_lower for term in ['income statement', 'profit and loss', 'p&l', 'statement of operations']):
            is_score += 5
        if any(term in text_lower for term in ['balance sheet', 'statement of financial position']):
            bs_score += 5
        if any(term in text_lower for term in ['cash flow', 'statement of cash flows']):
            cf_score += 5
        
        scores = {'income_statement': is_score, 'balance_sheet': bs_score, 'cash_flow': cf_score}
        max_type = max(scores, key=scores.get)
        
        return max_type if scores[max_type] > 0 else 'unknown'
    
    def parse_income_statement(self, text: str, tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Parse Income Statement from text and tables."""
        # Try to find a table that looks like an income statement
        for table in tables:
            if table is None or table.empty:
                continue
            
            # Check if table has income statement-like columns
            table_str = table.to_string().lower()
            if any(kw in table_str for kw in ['revenue', 'sales', 'income', 'cogs', 'gross profit']):
                # Clean and process the table
                df = self._clean_financial_table(table, 'income_statement')
                if df is not None and not df.empty:
                    return df
        
        # If no table found, try to extract from text
        return self._extract_from_text(text, 'income_statement')
    
    def parse_balance_sheet(self, text: str, tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Parse Balance Sheet from text and tables."""
        for table in tables:
            if table is None or table.empty:
                continue
            
            table_str = table.to_string().lower()
            if any(kw in table_str for kw in ['assets', 'liabilities', 'equity', 'cash', 'ppe']):
                df = self._clean_financial_table(table, 'balance_sheet')
                if df is not None and not df.empty:
                    return df
        
        return self._extract_from_text(text, 'balance_sheet')
    
    def parse_cash_flow(self, text: str, tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Parse Cash Flow Statement from text and tables."""
        for table in tables:
            if table is None or table.empty:
                continue
            
            table_str = table.to_string().lower()
            if any(kw in table_str for kw in ['cash flow', 'operating', 'investing', 'financing']):
                df = self._clean_financial_table(table, 'cash_flow')
                if df is not None and not df.empty:
                    return df
        
        return self._extract_from_text(text, 'cash_flow')
    
    def _clean_financial_table(self, table: pd.DataFrame, statement_type: str) -> Optional[pd.DataFrame]:
        """Clean and standardize a financial statement table."""
        if table is None or table.empty:
            return None
        
        # Remove empty rows and columns
        table = table.dropna(how='all').dropna(axis=1, how='all')
        
        if table.empty:
            return None
        
        # Try to identify header row
        header_row = 0
        for idx, row in table.iterrows():
            row_str = ' '.join(str(val) for val in row.values if pd.notna(val)).lower()
            if any(kw in row_str for kw in ['account', 'description', 'item', 'line']):
                header_row = idx
                break
        
        # Set header
        if header_row < len(table):
            table.columns = table.iloc[header_row]
            table = table.iloc[header_row + 1:].reset_index(drop=True)
        
        # Clean column names
        table.columns = [str(col).strip() if pd.notna(col) else f'col_{i}' 
                         for i, col in enumerate(table.columns)]
        
        # Remove rows that are mostly empty
        table = table[table.notna().sum(axis=1) > 1]
        
        return table if not table.empty else None
    
    def _extract_from_text(self, text: str, statement_type: str) -> Optional[pd.DataFrame]:
        """Extract financial data from unstructured text (fallback method)."""
        # This is a basic implementation - can be enhanced with regex patterns
        lines = text.split('\n')
        data = []
        
        for line in lines:
            # Look for lines with numbers (potential financial data)
            if re.search(r'[\d,]+\.?\d*', line):
                # Try to extract label and value
                parts = re.split(r'[\s]{2,}|\t', line.strip())
                if len(parts) >= 2:
                    label = parts[0].strip()
                    # Try to extract numeric value
                    numbers = re.findall(r'[\d,]+\.?\d*', ' '.join(parts[1:]))
                    if numbers:
                        try:
                            value = float(numbers[-1].replace(',', ''))
                            data.append({'Line Item': label, 'Amount': value})
                        except ValueError:
                            continue
        
        if data:
            return pd.DataFrame(data)
        return None
    
    def extract_all_statements(self, pdf_file) -> Dict[str, Any]:
        """Extract all financial statements from PDF (handles single PDF with multiple statements)."""
        results = {
            'income_statement': None,
            'balance_sheet': None,
            'cash_flow': None,
            'errors': [],
            'statements_detected': []
        }
        
        try:
            # Extract text and tables
            text = self.extract_text_from_pdf(pdf_file)
            tables = self.extract_tables_from_pdf(pdf_file)
            
            if not text and not tables:
                results['errors'].append("No text or tables found in PDF")
                return results
            
            # Split PDF content by pages or sections to identify different statements
            # For multi-statement PDFs, we need to identify where each statement starts/ends
            pages_text = self._split_by_statements(text)
            
            # Process each identified section
            for section_name, section_text in pages_text.items():
                # Get tables relevant to this section
                section_tables = self._get_tables_for_section(tables, section_text)
                
                # Identify and extract based on section
                if section_name == 'income_statement' or self._is_income_statement(section_text):
                    is_data = self.parse_income_statement(section_text, section_tables)
                    if is_data is not None and not is_data.empty:
                        results['income_statement'] = is_data
                        results['statements_detected'].append('Income Statement')
                
                elif section_name == 'balance_sheet' or self._is_balance_sheet(section_text):
                    bs_data = self.parse_balance_sheet(section_text, section_tables)
                    if bs_data is not None and not bs_data.empty:
                        results['balance_sheet'] = bs_data
                        results['statements_detected'].append('Balance Sheet')
                
                elif section_name == 'cash_flow' or self._is_cash_flow(section_text):
                    cf_data = self.parse_cash_flow(section_text, section_tables)
                    if cf_data is not None and not cf_data.empty:
                        results['cash_flow'] = cf_data
                        results['statements_detected'].append('Cash Flow')
            
            # If no sections identified, try extracting all from full text
            if not results['statements_detected']:
                results['income_statement'] = self.parse_income_statement(text, tables)
                results['balance_sheet'] = self.parse_balance_sheet(text, tables)
                results['cash_flow'] = self.parse_cash_flow(text, tables)
                
                # Update detected statements
                if results['income_statement'] is not None and not results['income_statement'].empty:
                    results['statements_detected'].append('Income Statement')
                if results['balance_sheet'] is not None and not results['balance_sheet'].empty:
                    results['statements_detected'].append('Balance Sheet')
                if results['cash_flow'] is not None and not results['cash_flow'].empty:
                    results['statements_detected'].append('Cash Flow')
            
        except Exception as e:
            results['errors'].append(f"Extraction error: {str(e)}")
            import traceback
            results['errors'].append(traceback.format_exc())
        
        return results
    
    def _split_by_statements(self, text: str) -> Dict[str, str]:
        """Split PDF text into sections by identifying statement boundaries."""
        sections = {}
        text_lower = text.lower()
        
        # Look for statement headers/titles
        # Income Statement patterns
        is_patterns = [
            r'(?:^|\n)\s*(?:income\s+statement|profit\s+and\s+loss|p\s*&\s*l|statement\s+of\s+operations)',
            r'(?:^|\n)\s*(?:revenue|sales|turnover)\s*(?:\n|$)'
        ]
        
        # Balance Sheet patterns
        bs_patterns = [
            r'(?:^|\n)\s*(?:balance\s+sheet|statement\s+of\s+financial\s+position)',
            r'(?:^|\n)\s*(?:assets|liabilities|equity)\s*(?:\n|$)'
        ]
        
        # Cash Flow patterns
        cf_patterns = [
            r'(?:^|\n)\s*(?:cash\s+flow|statement\s+of\s+cash\s+flows)',
            r'(?:^|\n)\s*(?:cash\s+from\s+operations|cash\s+from\s+investing|cash\s+from\s+financing)'
        ]
        
        # Find statement boundaries
        lines = text.split('\n')
        current_section = None
        section_start = 0
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check for Income Statement
            if any(re.search(pattern, line_lower, re.IGNORECASE) for pattern in is_patterns):
                if current_section and section_start < i:
                    sections[current_section] = '\n'.join(lines[section_start:i])
                current_section = 'income_statement'
                section_start = i
            
            # Check for Balance Sheet
            elif any(re.search(pattern, line_lower, re.IGNORECASE) for pattern in bs_patterns):
                if current_section and section_start < i:
                    sections[current_section] = '\n'.join(lines[section_start:i])
                current_section = 'balance_sheet'
                section_start = i
            
            # Check for Cash Flow
            elif any(re.search(pattern, line_lower, re.IGNORECASE) for pattern in cf_patterns):
                if current_section and section_start < i:
                    sections[current_section] = '\n'.join(lines[section_start:i])
                current_section = 'cash_flow'
                section_start = i
        
        # Add final section
        if current_section and section_start < len(lines):
            sections[current_section] = '\n'.join(lines[section_start:])
        
        # If no sections found, return full text as potential multi-statement
        if not sections:
            # Try to identify by content density
            if self._is_income_statement(text):
                sections['income_statement'] = text
            if self._is_balance_sheet(text):
                sections['balance_sheet'] = text
            if self._is_cash_flow(text):
                sections['cash_flow'] = text
        
        return sections
    
    def _is_income_statement(self, text: str) -> bool:
        """Check if text contains income statement content."""
        text_lower = text.lower()
        is_keywords = ['revenue', 'sales', 'cogs', 'gross profit', 'operating expenses', 'ebit', 'net income']
        return sum(1 for kw in is_keywords if kw in text_lower) >= 3
    
    def _is_balance_sheet(self, text: str) -> bool:
        """Check if text contains balance sheet content."""
        text_lower = text.lower()
        bs_keywords = ['assets', 'liabilities', 'equity', 'current assets', 'fixed assets']
        return sum(1 for kw in bs_keywords if kw in text_lower) >= 3
    
    def _is_cash_flow(self, text: str) -> bool:
        """Check if text contains cash flow content."""
        text_lower = text.lower()
        cf_keywords = ['cash flow', 'operating activities', 'investing activities', 'financing activities']
        return sum(1 for kw in cf_keywords if kw in text_lower) >= 2
    
    def _get_tables_for_section(self, all_tables: List[pd.DataFrame], section_text: str) -> List[pd.DataFrame]:
        """Get tables that are relevant to a specific section."""
        relevant_tables = []
        section_lower = section_text.lower()
        
        for table in all_tables:
            if table is None or table.empty:
                continue
            
            # Check if table content matches section
            table_str = table.to_string().lower()
            
            # Simple heuristic: if table has keywords matching the section, include it
            if len(table_str) > 0:
                relevant_tables.append(table)
        
        return relevant_tables


def process_pdf_financial_statements(
    db,
    scenario_id: str,
    user_id: str,
    pdf_file,
    period_date: Optional[str] = None,
    statement_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process PDF file and save extracted financial statements to database.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        user_id: User ID
        pdf_file: Uploaded PDF file
        period_date: Period date (YYYY-MM-DD format) - if None, will try to extract from PDF
        statement_type: Type of statement ('income_statement', 'balance_sheet', 'cash_flow', or None for all)
    
    Returns:
        Dictionary with success status and extracted data
    """
    extractor = PDFFinancialExtractor()
    
    # Extract statements
    results = extractor.extract_all_statements(pdf_file)
    
    if results['errors']:
        return {
            'success': False,
            'error': '; '.join(results['errors']),
            'results': results
        }
    
    # Determine which statements to save
    statements_to_save = []
    if statement_type:
        if statement_type == 'income_statement' and results['income_statement'] is not None:
            statements_to_save.append(('income_statement', results['income_statement']))
        elif statement_type == 'balance_sheet' and results['balance_sheet'] is not None:
            statements_to_save.append(('balance_sheet', results['balance_sheet']))
        elif statement_type == 'cash_flow' and results['cash_flow'] is not None:
            statements_to_save.append(('cash_flow', results['cash_flow']))
    else:
        # Save all found statements
        if results['income_statement'] is not None:
            statements_to_save.append(('income_statement', results['income_statement']))
        if results['balance_sheet'] is not None:
            statements_to_save.append(('balance_sheet', results['balance_sheet']))
        if results['cash_flow'] is not None:
            statements_to_save.append(('cash_flow', results['cash_flow']))
    
    if not statements_to_save:
        return {
            'success': False,
            'error': 'No financial statements found in PDF',
            'results': results
        }
    
    # Save to database using the existing save function
    from components.trial_balance_processor import _save_extracted_statements
    
    # Convert extracted data to the format expected by _save_extracted_statements
    is_df = results.get('income_statement')
    bs_df = results.get('balance_sheet')
    cf_df = results.get('cash_flow')
    
    # If period_date not provided, use current date or extract from PDF
    if period_date is None:
        period_date = datetime.now().strftime('%Y-%m-%d')
    
    # Add period information to DataFrames if missing
    period_col = 'period_date' if 'period_date' in (is_df.columns if is_df is not None else []) else 'month'
    
    # Save statements
    save_result = _save_extracted_statements(
        db, scenario_id, user_id,
        is_df,
        bs_df,
        cf_df,
        period_col
    )
    
    return {
        'success': save_result.get('success', False),
        'error': save_result.get('error'),
        'results': results,
        'saved': save_result.get('saved', {}),
        'statements_found': [stmt[0] for stmt in statements_to_save]
    }
