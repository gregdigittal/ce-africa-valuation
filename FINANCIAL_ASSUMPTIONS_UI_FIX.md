# Financial Assumptions UI Fix

## Issues Fixed

### 1. Removed Duplicates
**Problem:** Duplicate line items were showing:
- "Total Revenue" and "Revenue"
- "Cost of Goods Sold" and "COGS"
- "Operating Expenses" and "OPEX"
- "Gross Profit" appeared twice

**Solution:**
- Implemented canonical name mapping (prefer `total_revenue` over `revenue`)
- Added duplicate detection by display name
- Only show one canonical version of each metric

### 2. Excluded Calculated Elements
**Problem:** Calculated figures (Gross Profit, EBIT, Tax, Net Income) were showing as configurable.

**Solution:**
- Import `FORECAST_ELEMENTS` from `forecast_correlation_engine`
- Filter out any assumption where `is_calculated == True`
- Calculated elements are now excluded from the configuration UI

**Calculated Elements Excluded:**
- `gross_profit` (calculated: revenue - cogs)
- `ebit` (calculated: gross_profit - opex - depreciation)
- `interest_expense` (calculated from balance sheet)
- `tax` (calculated: max(ebt * tax_rate, 0))
- `net_profit` (calculated: ebit - interest_expense - tax)

### 3. Percentage-Based for COGS and OPEX
**Problem:** COGS and OPEX were shown as absolute numbers instead of percentages of revenue.

**Solution:**
- Modified `analyze_all_financials()` to calculate COGS and OPEX as percentages of revenue
- New assumption keys: `cogs_pct_of_revenue` and `opex_pct_of_revenue`
- Display shows percentages (e.g., "COGS % of Revenue: 65.2%")
- Falls back to absolute values only if revenue data is unavailable

**Best Practice:** Financial modeling typically uses:
- COGS as % of Revenue (e.g., 65% of revenue)
- OPEX as % of Revenue (e.g., 20% of revenue)
- This allows the model to scale with revenue growth

### 4. Trend-Based Distribution Clarification
**Problem:** UI didn't clarify that distributions are applied around trend forecasts, not fixed historical values.

**Solution:**
- Added prominent caption at top of "Configure All Financial Assumptions" section
- Added detailed explanation in "Advanced: Adjust Distribution Parameters" expander
- Added note in individual assumption editors
- Updated help text for "Use Distribution" toggle

**Key Message:**
```
⚠️ Important: Distributions are applied around trend forecasts at each period, 
not fixed historical values. Configure trends in 'Trend Forecast' tab first.

Distribution Parameters Applied to Trend Forecasts:
- Mean = Trend forecast value at each future period
- Standard Deviation = Proportionally scaled from historical 
  (scaled by forecast/historical ratio)
- This ensures MC simulation confidence intervals follow the trend trajectory

Example: If historical mean=100m, stdev=15m, and Year 3 forecast=200m:
- MC uses: mean=200m, stdev=30m (15m × 200m/100m)
```

## Code Changes

### `components/ai_assumptions_engine.py`

1. **`analyze_all_financials()` method:**
   - Removed duplicate metric analysis (only analyze canonical names)
   - Changed COGS and OPEX to percentage-based analysis
   - Removed analysis of calculated elements

2. **`render_configure_all_financial_assumptions()` function:**
   - Added canonical name mapping and duplicate filtering
   - Added filter to exclude calculated elements using `FORECAST_ELEMENTS`
   - Added prominent caption about trend-based distributions
   - Updated display formatting for percentage-based metrics

3. **`render_assumption_editor()` function:**
   - Added note about trend-based distributions in historical stats section
   - Updated help text for "Use Distribution" toggle

## Benefits

1. **Cleaner UI:** No duplicates, only relevant configurable elements
2. **Best Practice Modeling:** COGS/OPEX as % of revenue aligns with financial modeling standards
3. **Correct Logic:** Calculated elements can't be configured (they're derived)
4. **Clear Communication:** Users understand distributions are applied around trends, not fixed values

## Testing

To verify:
1. Run AI Assumptions analysis
2. Go to "Configure All Financial Assumptions" tab
3. Verify:
   - ✅ No duplicates (only "Total Revenue", not "Revenue" and "Total Revenue")
   - ✅ No calculated elements (no Gross Profit, EBIT, Tax, Net Income)
   - ✅ COGS and OPEX shown as percentages (e.g., "COGS % of Revenue")
   - ✅ Clear messaging about trend-based distributions

---

**Date:** December 20, 2025  
**Status:** ✅ Implemented
