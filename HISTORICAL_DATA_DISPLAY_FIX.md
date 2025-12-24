# Historical Data Display in Financial Statements - Fix

## Problem
Historical numbers were not showing in the financial statement results, and when they did show, they weren't clearly identified as actual/historical data.

## Requirements
1. Show historical data alongside forecast data
2. Mark historical periods with "A" next to the FY label (e.g., "FY2022 A")
3. Provide different background color for historical columns for easy identification

## Fixes Applied

### 1. Enhanced Historical Data Loading
**File:** `components/forecast_section.py` - `render_results_tab()`

**Changes:**
- Now uses `load_historical_data()` from AI Assumptions engine (same logic)
- Includes aggregation from detailed line items if summary data isn't available
- Ensures `is_actual` flag is set to `True` for all historical data
- Sets `data_type` to 'Actual' for historical records

### 2. Updated Annual Aggregation
**File:** `components/forecast_section.py` - `aggregate_to_annual()`

**Before:**
```python
annual_df['period_label'] = 'FY' + annual_df['period_year'].astype(str)
```

**After:**
- Preserves `is_actual` flag when aggregating
- Creates period_label with "A" marker: `"FY2022 A"` for historical years
- Sets `data_type` to 'Actual Annual' for historical years

### 3. Enhanced Income Statement Styling
**File:** `components/forecast_section.py` - `render_income_statement_table()`

**CSS Changes:**
- Added `.actual-header` class for historical column headers
  - Background: `rgba(100,116,139,0.3)` (slate gray)
  - Border: Left and right borders in slate gray
  - Color: White text
- Enhanced `.actual` class for historical cells
  - Background: `rgba(100,116,139,0.15)` (lighter slate gray)
  - Borders: Left and right borders for visual separation

**Header Changes:**
- Annual view: Shows "FY2022 A" (A already in label)
- Monthly view: Shows "Dec 2022 (A)" (adds (A) marker)
- Applies `actual-header` class to historical column headers

### 4. Updated Balance Sheet Styling
**File:** `components/forecast_section.py` - `render_balance_sheet_table()`

**Changes:**
- Annual aggregation preserves `is_actual` flag
- Period labels include "A" for historical years
- Headers use distinct background color for historical columns
- Cells use slate gray background with borders

### 5. Updated Cash Flow Styling
**File:** `components/forecast_section.py` - `render_cash_flow_table()`

**Changes:**
- Annual aggregation preserves `is_actual` flag
- Period labels include "A" for historical years
- Headers use distinct background color for historical columns
- Cells use slate gray background with borders

## Visual Design

### Historical Columns (Actual Data)
- **Header Background:** Slate gray (`rgba(100,116,139,0.3)`)
- **Header Text:** White
- **Header Borders:** Left and right borders in slate gray
- **Cell Background:** Light slate gray (`rgba(100,116,139,0.15)`)
- **Cell Borders:** Left and right borders for visual separation
- **Label Format:**
  - Annual: `FY2022 A`
  - Monthly: `Dec 2022 (A)`

### Forecast Columns (Projected Data)
- **Header Background:** Dark (`#1E1E1E`)
- **Header Text:** Gold (`#D4A537`)
- **Cell Background:** Default (transparent)
- **Label Format:**
  - Annual: `FY2025`
  - Monthly: `Jan 2025`

## How It Works

1. **Historical data is loaded** using the same logic as AI Assumptions
2. **Data is merged** with forecast data in `build_monthly_financials()`
3. **Annual aggregation** preserves `is_actual` flag and adds "A" to labels
4. **Table rendering** applies distinct styling to historical columns
5. **Headers and cells** use slate gray background for easy identification

## Testing

To verify the fix:

1. **Import historical data** in Setup → Historics
2. **Run a forecast**
3. **Go to Forecast Results → Financial Statements**
4. **Check:**
   - ✅ Historical periods show with "A" marker
   - ✅ Historical columns have slate gray background
   - ✅ Historical columns are visually distinct from forecast columns
   - ✅ Works in both monthly and annual view

## Expected Behavior

- **Historical data visible** alongside forecast data
- **Clear visual distinction** between actual and forecast columns
- **"A" marker** on all historical period labels
- **Consistent styling** across Income Statement, Balance Sheet, and Cash Flow

---

**Date:** December 20, 2025  
**Status:** ✅ Fixed
