# AFS Extract Transformation - Complete ✅

**Date:** 2025-12-19  
**Status:** Successfully transformed all files

## Summary

All 12 AFS Extract files have been successfully transformed from vertical format (line items as rows) to horizontal format (periods as rows) compatible with the model's import functions.

## Files Processed

### Income Statements (4 periods)
- ✅ FY2022
- ✅ FY2023  
- ✅ FY2024
- ✅ YTD_Oct2025

### Balance Sheets (4 periods)
- ✅ FY2022
- ✅ FY2023
- ✅ FY2024
- ✅ YTD_Oct2025

### Cash Flow Statements (4 periods)
- ✅ FY2022
- ✅ FY2023
- ✅ FY2024
- ✅ YTD_Oct2025

## Output Files

Transformed files are located in:
```
/Users/gregmorris/Development Projects/CE Africa/CE Africa Files/AFS Extract/transformed/
```

### Generated Files:
1. **`historic_financials.csv`** - Income Statement data
   - Columns: `month, revenue, cogs, gross_profit, opex, ebit`
   - 4 periods: 2022-12-01, 2023-12-01, 2024-12-01, 2025-10-01

2. **`historical_balance_sheet.csv`** - Balance Sheet data
   - Columns: `month, cash_and_equivalents, accounts_receivable, inventory, total_current_assets, ppe_net, intangible_assets, total_noncurrent_assets, total_assets, total_current_liabilities, total_noncurrent_liabilities, total_liabilities, share_capital, retained_earnings, total_equity, accounts_payable, short_term_debt, long_term_debt`
   - 4 periods: 2022-12-01, 2023-12-01, 2024-12-01, 2025-10-01

3. **`historical_cashflow.csv`** - Cash Flow data
   - Columns: `month, net_income, depreciation_amortization, change_in_receivables, change_in_inventory, change_in_payables, cash_from_operations, cash_from_investing, cash_from_financing, capital_expenditure, debt_repayment, dividends_paid, net_change_in_cash`
   - 4 periods: 2022-12-01, 2023-12-01, 2024-12-01, 2025-10-01

## Format Consistency

✅ **All files match the expected model format:**
- One row per period (horizontal format)
- Standard column names matching `FIELD_CONFIGS` in `column_mapper.py`
- Period dates in `YYYY-MM-DD` format (first of month)
- Numeric values properly formatted (parentheses converted to negatives)
- COGS values correctly aggregated from multiple line items

## Key Transformations Applied

1. **Structure:** Vertical → Horizontal (transposed)
2. **Field Mapping:** AFS field names → Model field names
3. **Data Aggregation:** 
   - Operating expenses summed into `opex`
   - Multiple COGS line items summed (e.g., "Cost of Goods Sold" + "Cost of Goods Sold - Services")
4. **Calculations:**
   - `ebit` = `gross_profit` - `opex`
5. **Accounting Format Handling:**
   - Parentheses `(152227691)` → negative numbers `-152227691`
   - Commas removed from numeric strings
6. **Period Extraction:**
   - `FY2024` → `2024-12-01`
   - `YTD_Oct2025` → `2025-10-01`

## Next Steps

1. **Import into Model:**
   - Navigate to **Setup → Historics** tab
   - Use the **Income Statement**, **Balance Sheet**, and **Cash Flow** tabs
   - Upload the transformed CSV files from the `transformed/` folder
   - The column mapper will automatically recognize the field names

2. **Verify Data:**
   - Check that all 4 periods imported correctly
   - Verify totals match your source AFS files
   - Review any warnings or mapping suggestions

3. **Use in Forecasts:**
   - Historical data will be available for:
     - AI Assumptions analysis
     - Trend forecast configuration
     - Historical correlation analysis

## Notes

- The transformation script handles various column name formats (case-insensitive)
- Accounting parentheses format is automatically converted
- Empty rows and note rows are filtered out
- Multiple COGS line items are automatically summed when "Total Cost of Sales" is not available

## Script Location

The transformation script is available at:
```
/Users/gregmorris/Development Projects/CE Africa/ce-africa-valuation/scripts/transform_afs_extract.py
```

You can re-run it anytime if you update the source AFS Extract files.
