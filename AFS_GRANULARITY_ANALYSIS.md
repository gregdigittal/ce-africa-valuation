# Financial Statement Import Granularity Analysis

**Date:** 2025-12-19  
**Status:** Current system is SUMMARY-ONLY, needs enhancement for detailed line items

## Current State: SUMMARY-ONLY

### Current Import Format (Summary Level)

**Income Statement (`historic_financials`):**
- Only 6 fields: `month, revenue, cogs, gross_profit, opex, ebit`
- **Missing:** All individual expense line items (Accounting Fees, Advertising, Depreciation, Employee Costs, etc.)
- **Missing:** Individual income line items (Other Operating Income, Foreign Exchange Gains, etc.)
- **Missing:** Tax details (Current Tax, Deferred Tax components)
- **Missing:** Finance cost details (Lease Liabilities, Interest Paid, etc.)

**Balance Sheet (`historical_balance_sheet`):**
- Only summary fields: `total_current_assets, total_assets, total_liabilities, total_equity`
- **Missing:** Individual asset line items (Property Plant and Equipment, Right-of-Use Assets, Intangible Assets - Patterns, Intangible Assets - Computer Software, etc.)
- **Missing:** Individual liability line items (Lease Liabilities - Current, Lease Liabilities - Non-Current, etc.)
- **Missing:** Detailed equity components

**Cash Flow (`historical_cashflow`):**
- Only summary fields: `cash_from_operations, cash_from_investing, cash_from_financing`
- **Missing:** Individual operating activity line items (Depreciation and Amortisation, Interest Income, Finance Costs, etc.)
- **Missing:** Individual working capital changes (Increase in Inventories, Increase in Trade Receivables, etc.)
- **Missing:** Individual investing activity line items (Purchase of Property Plant and Equipment, Purchases of Intangible Assets, etc.)
- **Missing:** Individual financing activity line items (Loans to Shareholders, Cash Repayments on Lease Liabilities, etc.)

## What You Have in AFS Extract Files

### Income Statement (56+ line items per period):
- Revenue
- Cost of Sales
- Gross Profit
- **30+ Operating Expense line items:**
  - Accounting Fees
  - Advertising
  - Amortisation
  - Bad Debts
  - Bank Charges
  - Commission Paid
  - Computer Expenses
  - Consulting and Professional Fees
  - Consumables
  - Credit Cards
  - Depreciation
  - Donations
  - Employee Costs
  - Entertainment
  - Equipment
  - General Expenses
  - Insurance
  - Lease Modification
  - Lease Rentals on Operating Lease
  - Legal Expenses
  - Medical Expenses
  - Motor Vehicle Expenses
  - Municipal Expenses
  - Printing and Stationery
  - Protective Clothing
  - Repairs and Maintenance
  - Security
  - Staff Welfare
  - Telephone and Fax
  - Training
  - Travel - Local
  - Total Operating Expenses
- Operating Profit
- **Finance Income/Expense line items:**
  - Investment Income - Bank Interest
  - Finance Costs - Lease Liabilities
  - Finance Costs - Late Payment Tax
  - Finance Costs - Interest Paid
  - Total Finance Costs
- Profit/(Loss) Before Taxation
- **Tax line items:**
  - Current Tax - Local
  - Deferred Tax - Prior Period Adjustment
  - Deferred Tax - Other
  - Total Taxation
- Net Profit/(Loss) for the Year

### Balance Sheet (30+ line items per period):
- **Non-Current Assets:**
  - Property Plant and Equipment
  - Right-of-Use Assets
  - Intangible Assets - Patterns
  - Intangible Assets - Computer Software
  - Total Intangible Assets
  - Loans to Shareholders
  - Deferred Tax Asset
  - Total Non-Current Assets
- **Current Assets:**
  - Inventories - Work in Progress
  - Inventories - Crushing Equipment
  - Inventories - Goods in Transit
  - Total Inventories
  - Trade Receivables
  - Loss Allowance
  - Trade Receivables Net
  - Staff Loans
  - Purchases Accrual
  - VAT Receivable
  - Prepayments
  - Total Trade and Other Receivables
  - Cash and Cash Equivalents
  - Total Current Assets
- **Equity:**
  - Stated Capital
  - Retained Income/(Accumulated Loss)
  - Total Equity
- **Non-Current Liabilities:**
  - Lease Liabilities - Non-Current
  - Total Non-Current Liabilities
- **Current Liabilities:**
  - Lease Liabilities - Current
  - (Various payables - need to identify)
  - Total Current Liabilities
- Total Assets, Total Liabilities

### Cash Flow (25+ line items per period):
- Loss/Profit Before Taxation
- **Operating Activities - Adjustments:**
  - Depreciation and Amortisation
  - Interest Income
  - Finance Costs
- **Working Capital Changes:**
  - Increase in Inventories
  - Increase in Trade and Other Receivables
  - Increase in Trade and Other Payables
  - Increase in Deferred Income
- Cash (Used in)/Generated From Operations
- Interest Income Received
- Finance Costs Paid
- Tax Paid
- Net Cash From Operating Activities
- **Investing Activities:**
  - Purchase of Property Plant and Equipment
  - Cash Additions to Right-of-Use Assets
  - Purchases of Intangible Assets
  - Net Cash From Investing Activities
- **Financing Activities:**
  - Loans to Shareholders
  - Cash Repayments on Lease Liabilities
  - Other Cash Flows on Lease Liabilities
  - Receipts of Other Financial Liability
  - Net Cash From Financing Activities
- Total Cash Movement for the Year
- Cash and Cash Equivalents at Beginning of Year
- Cash and Cash Equivalents at End of Year

## Solution: Add Detailed Line Item Support

### Option 1: New Tables for Line Items (Recommended)

Create three new tables:
1. `historical_income_statement_line_items`
2. `historical_balance_sheet_line_items`
3. `historical_cashflow_line_items`

**Structure:**
- `scenario_id` (UUID)
- `user_id` (UUID)
- `period_date` (DATE)
- `line_item_name` (TEXT) - e.g., "Accounting Fees", "Depreciation"
- `category` (TEXT) - e.g., "Operating Expenses", "Revenue"
- `sub_category` (TEXT) - Optional, e.g., "PPE", "Leases"
- `amount` (NUMERIC) - The value for this line item
- `statement_type` (TEXT) - 'income_statement', 'balance_sheet', 'cash_flow'

**Benefits:**
- ✅ Preserves ALL line items from AFS files
- ✅ Maintains category/sub-category structure
- ✅ Allows detailed analysis and trend tracking
- ✅ Supports granular forecasting (e.g., forecast individual expense categories)
- ✅ Enables better AI assumptions analysis (can analyze trends in specific expense types)

### Option 2: JSONB Column in Existing Tables

Add a `line_items` JSONB column to existing tables:
- `historic_financials.line_items` (JSONB)
- `historical_balance_sheet.line_items` (JSONB)
- `historical_cashflow.line_items` (JSONB)

**Structure:**
```json
{
  "line_items": [
    {"name": "Accounting Fees", "category": "Operating Expenses", "amount": -375000},
    {"name": "Advertising", "category": "Operating Expenses", "amount": -1445954},
    ...
  ]
}
```

**Benefits:**
- ✅ No new tables needed
- ✅ Flexible structure
- ⚠️ Harder to query individual line items
- ⚠️ Less efficient for filtering/aggregation

### Option 3: Hybrid Approach

Keep summary tables for quick access, add line item tables for detail:
- `historic_financials` - Summary (current)
- `historical_income_statement_line_items` - Detailed (new)
- Both updated together during import

**Benefits:**
- ✅ Best of both worlds
- ✅ Fast summary queries
- ✅ Detailed analysis when needed
- ⚠️ More complex to maintain

## Recommendation: Option 1 (New Tables)

**Why:**
1. **Complete Data Preservation:** All 100+ line items per period preserved
2. **Query Performance:** Can efficiently filter/aggregate by category, line item name
3. **Analytical Power:** Enables trend analysis on individual expense categories
4. **Forecasting Flexibility:** Can forecast individual line items (e.g., "Employee Costs" as % of revenue, "Depreciation" based on asset base)
5. **AI Analysis:** Better assumptions derivation from granular data

## Implementation Plan

1. **Create Migration:** `migrations_add_detailed_line_items.sql` ✅ (Created)
2. **Update Transformation Script:** Modify to output both summary AND detailed line items
3. **Update Import Functions:** Add support for importing detailed line items
4. **Update UI:** Show both summary and detailed views
5. **Update Analysis:** Use detailed line items for AI assumptions

## Next Steps

1. **Run Migration:** Execute `migrations_add_detailed_line_items.sql` in Supabase
2. **Update Transformation Script:** Generate detailed line item files
3. **Update Import Functions:** Add import support for line items
4. **Test:** Import AFS Extract files and verify all line items are preserved
