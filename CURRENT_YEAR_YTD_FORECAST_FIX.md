# Current Year YTD + Forecast Fix

## Problem
For the current year-to-date (YTD), the model should:
1. Include actual data for months that have passed (YTD actuals)
2. Forecast the remaining months of the current year
3. Show full year figures as the sum of actual YTD + forecast remaining months

## Requirements
- Current year annual totals = Sum of YTD actuals + Sum of forecast for remaining months
- Annual view should show combined actual + forecast for current year
- Monthly view should show actual months with "A" marker and forecast months without marker

## Solution

### 1. Enhanced Annual Aggregation

**File:** `components/forecast_section.py` - `aggregate_to_annual()`

**Changes:**
- Identifies current year
- Checks if current year has both actuals (YTD) and forecast (remaining months)
- Marks current year as "Actual + Forecast Annual" if it has both
- Properly sums both actual and forecast data for current year

**Logic:**
```python
current_year = datetime.now().year
current_year_data = monthly_df[monthly_df['period_year'] == current_year]

if has_actuals and has_forecast:
    # Current year has both actuals (YTD) and forecast (remaining months)
    annual_df.loc[annual_df['period_year'] == current_year, 'data_type'] = 'Actual + Forecast Annual'
    annual_df.loc[annual_df['period_year'] == current_year, 'period_label'] = f"FY{current_year}"
```

### 2. Forecast Start Date Logic

**File:** `components/forecast_section.py` - `build_monthly_financials()`

**Changes:**
- Checks if historical data includes current month
- If current month has actual data, forecast starts from next month
- If current month has no actual data, forecast starts from current month
- Ensures all remaining months of current year are forecasted

**Logic:**
```python
# Check if we have historical data for current month
has_current_month_actual = False
if historical_data is not None and not historical_data.empty:
    for _, hist_row in historical_data.iterrows():
        hist_date = hist_row.get('period_date')
        if hist_date.year == current_year and hist_date.month == current_month:
            has_current_month_actual = True
            break

# If we have actual for current month, start forecast from next month
# Otherwise, start from current month
if has_current_month_actual:
    period_date = datetime.now().replace(day=1) + relativedelta(months=i+1)
else:
    period_date = datetime.now().replace(day=1) + relativedelta(months=i)
```

## How It Works

1. **Historical Data Loading:**
   - Loads actual financial data (including YTD for current year)
   - Marks all historical periods with `is_actual = True`

2. **Forecast Generation:**
   - Starts from current month (or next month if current month has actuals)
   - Generates forecast for remaining months
   - Marks all forecast periods with `is_actual = False`

3. **Monthly View:**
   - Shows actual months with "A" marker (e.g., "Jan 2025 A", "Feb 2025 A")
   - Shows forecast months without marker (e.g., "Mar 2025", "Apr 2025")
   - Both appear in the same table

4. **Annual Aggregation:**
   - Groups by year
   - For current year:
     - Sums actual YTD months
     - Sums forecast remaining months
     - Combines both for full year total
     - Marks as "Actual + Forecast Annual"
   - For past years: Sums actuals only
   - For future years: Sums forecast only

## Example

**Scenario:** Today is March 15, 2025

**Current Year (2025):**
- **Actual YTD:** Jan 2025, Feb 2025, Mar 2025 (marked with "A")
- **Forecast Remaining:** Apr 2025, May 2025, ..., Dec 2025 (no "A" marker)
- **Annual Total:** Sum of all 12 months (3 actual + 9 forecast)

**Annual View:**
- **FY2025:** Shows combined total (Actual + Forecast Annual)
- **FY2024:** Shows actual total only (FY2024 A)
- **FY2026:** Shows forecast total only (FY2026)

## Benefits

1. **Accurate Current Year:** Full year includes both actuals and forecast
2. **Clear Distinction:** Users can see which months are actual vs forecast
3. **Proper Aggregation:** Annual totals correctly combine actuals and forecast
4. **Flexible:** Handles cases where current month has or doesn't have actuals

## Testing

To verify the fix:

1. **Import historical data:**
   - Include data up to current month (YTD)
   - Example: If today is March 15, import Jan-Mar 2025

2. **Run forecast:**
   - Forecast should start from current month (or next if current has actuals)
   - Should forecast remaining months of current year

3. **Check monthly view:**
   - Actual months should show with "A" marker
   - Forecast months should show without marker
   - Both should appear for current year

4. **Check annual view:**
   - Current year should show combined total
   - Should be labeled as "Actual + Forecast Annual"
   - Total should equal sum of YTD actuals + forecast remaining months

5. **Verify calculation:**
   - Sum actual YTD months manually
   - Sum forecast remaining months manually
   - Annual total should equal sum of both

## Expected Behavior

- ✅ Current year includes YTD actuals
- ✅ Current year includes forecast for remaining months
- ✅ Annual aggregation sums both for current year
- ✅ Current year labeled as "Actual + Forecast Annual"
- ✅ Monthly view shows actual months with "A" marker
- ✅ Monthly view shows forecast months without marker
- ✅ Full year total = YTD actuals + forecast remaining months

---

**Date:** December 20, 2025  
**Status:** ✅ Fixed
