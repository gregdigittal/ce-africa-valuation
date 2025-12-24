# Financial Statement Import Granularity - Current State & Solution

**Date:** 2025-12-19  
**Status:** ⚠️ **CURRENT SYSTEM IS SUMMARY-ONLY** - Solution provided below

## ⚠️ Current Limitation: SUMMARY-ONLY

The current import system **only supports summary-level data**, not individual line items.

### What Gets Imported (Summary):

**Income Statement:**
- ✅ `revenue` (total)
- ✅ `cogs` (total)
- ✅ `gross_profit` (total)
- ✅ `opex` (total - all expenses summed)
- ✅ `ebit` (calculated)
- ❌ **Missing:** All 30+ individual expense line items (Accounting Fees, Advertising, Depreciation, Employee Costs, etc.)
- ❌ **Missing:** Individual income items (Other Operating Income, Foreign Exchange Gains)
- ❌ **Missing:** Tax details (Current Tax, Deferred Tax components)
- ❌ **Missing:** Finance cost details

**Balance Sheet:**
- ✅ Summary totals only (`total_current_assets`, `total_assets`, etc.)
- ❌ **Missing:** Individual asset line items (Property Plant and Equipment, Right-of-Use Assets, Intangible Assets - Patterns, etc.)
- ❌ **Missing:** Individual liability line items (Lease Liabilities - Current, Lease Liabilities - Non-Current, etc.)

**Cash Flow:**
- ✅ Summary totals only (`cash_from_operations`, `cash_from_investing`, `cash_from_financing`)
- ❌ **Missing:** Individual operating activity line items
- ❌ **Missing:** Individual working capital changes
- ❌ **Missing:** Individual investing/financing activity line items

## ✅ Solution: Detailed Line Item Support

I've created a **complete solution** to preserve ALL line items:

### 1. Database Migration Created

**File:** `migrations_add_detailed_line_items.sql`

Creates three new tables:
- `historical_income_statement_line_items`
- `historical_balance_sheet_line_items`
- `historical_cashflow_line_items`

**Structure:**
- `scenario_id`, `user_id`, `period_date`
- `line_item_name` - e.g., "Accounting Fees", "Depreciation"
- `category` - e.g., "Operating Expenses", "Revenue"
- `sub_category` - Optional, e.g., "PPE", "Leases"
- `amount` - The value for this line item
- `statement_type` - 'income_statement', 'balance_sheet', 'cash_flow'

### 2. Transformation Script Updated

The transformation script now generates **BOTH**:
- **Summary files** (for backward compatibility)
- **Detailed line item files** (for complete data preservation)

**Generated Files:**
```
transformed/
├── historic_financials.csv (Summary - 4 periods)
├── historical_balance_sheet.csv (Summary - 4 periods)
├── historical_cashflow.csv (Summary - 4 periods)
├── historical_income_statement_line_items.csv (Detailed - 210 line items)
├── historical_balance_sheet_line_items.csv (Detailed - 153 line items)
└── historical_cashflow_line_items.csv (Detailed - 116 line items)
```

### 3. Data Preserved

**Income Statement:** 210 line items across 4 periods
- All individual expense categories
- All income categories
- Tax components
- Finance cost details

**Balance Sheet:** 153 line items across 4 periods
- All individual asset line items
- All individual liability line items
- Equity components

**Cash Flow:** 116 line items across 4 periods
- All operating activity line items
- All working capital changes
- All investing/financing activities

## Next Steps to Enable Full Granularity

### Step 1: Run Database Migration
```sql
-- Execute in Supabase SQL Editor:
-- File: migrations_add_detailed_line_items.sql
```

This creates the three new tables for storing detailed line items.

### Step 2: Update Import Functions
The import functions in `column_mapper.py` need to be updated to:
1. Accept the detailed line item CSV files
2. Save data to the new line item tables
3. Optionally also save to summary tables (for backward compatibility)

### Step 3: Update UI
The Setup Wizard needs to:
1. Show both summary and detailed import options
2. Display detailed line items in a viewable format
3. Allow filtering/grouping by category

### Step 4: Update Analysis Functions
AI Assumptions and Trend Analysis should:
1. Use detailed line items when available
2. Provide category-level analysis (e.g., "Employee Costs trend", "Depreciation trend")
3. Enable granular forecasting (forecast individual expense categories)

## Recommendation

**Proceed with the detailed line item approach** because:

1. ✅ **Complete Data Preservation:** All 479 line items (210+153+116) preserved
2. ✅ **Analytical Power:** Can analyze trends in specific expense categories
3. ✅ **Forecasting Flexibility:** Can forecast individual line items (e.g., "Employee Costs" as % of revenue)
4. ✅ **Better AI Analysis:** More granular data = better assumptions derivation
5. ✅ **Backward Compatible:** Summary tables still work for existing functionality

## Current Status

✅ **Migration SQL:** Created  
✅ **Transformation Script:** Updated to generate detailed files  
✅ **Data Extracted:** 479 detailed line items ready to import  
⏳ **Import Functions:** Need to be updated  
⏳ **UI:** Need to be updated  
⏳ **Analysis Functions:** Need to be updated  

## Files Ready for Import

All files are in:
```
/Users/gregmorris/Development Projects/CE Africa/CE Africa Files/AFS Extract/transformed/
```

**Summary files** (current format - ready to import now):
- `historic_financials.csv`
- `historical_balance_sheet.csv`
- `historical_cashflow.csv`

**Detailed line item files** (new format - need import function update):
- `historical_income_statement_line_items.csv`
- `historical_balance_sheet_line_items.csv`
- `historical_cashflow_line_items.csv`

## Decision Required

**Option A:** Proceed with summary-only (current system)
- ✅ Works immediately
- ❌ Loses 479 line items of detail
- ❌ Limited analytical capability

**Option B:** Implement detailed line item support (recommended)
- ✅ Preserves all data
- ✅ Enables granular analysis
- ⏳ Requires migration + import function updates

**Which approach would you prefer?**
