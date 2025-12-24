# Historical Validation Check - Fix

## Problem
Users need a validation check at the bottom of the financial statements that compares the imported net income from financial statements to the calculated net income from the actuals section. If they balance, it should state "Historics agree with actual upload".

## Requirements
1. Show Net Income per Financial Statements (imported value)
2. Show Calculated Net Income (Revenue - COGS - OPEX - Interest - Tax)
3. Compare the two values
4. Display validation message:
   - ✅ "Historics agree with actual upload" if they match (within tolerance)
   - ⚠️ "Discrepancy detected - please review" if they don't match

## Solution

### 1. Added Validation Function

**File:** `components/forecast_section.py` - `_render_historical_validation_check()`

**Functionality:**
- Filters data to actuals periods only (`is_actual == True`)
- Gets imported net income from `net_income` column (from imported financial statements)
- Calculates net income from components:
  ```
  Calculated Net Income = Revenue - COGS - OPEX - Interest - Tax
  ```
- Compares values with tolerance:
  - 1% of imported value, or
  - Minimum 100 currency units
  - Whichever is larger
- Displays validation message at bottom of income statement

### 2. Integrated into Income Statement Renderer

**File:** `components/forecast_section.py` - `render_income_statement_table()`

**Changes:**
- Added call to `_render_historical_validation_check(display_data)` after table is rendered
- Validation appears at bottom of income statement
- Only shows for actuals periods (historical data)

## Implementation Details

### Validation Logic

```python
# Get imported net income (from financial statements)
imported_net_income = actuals_data['net_income']

# Calculate from components
calculated_net_income = revenue - cogs - opex - interest - tax

# Compare with tolerance
tolerance = max(abs(imported_total) * 0.01, 100)  # 1% or 100 units
difference = abs(imported_total - calculated_total)
agrees = difference <= tolerance
```

### Display Format

**If Agrees:**
```
✅ Historical Validation: Net Income per Financial Statements: R 1,000,000 | Calculated Net Income: R 1,000,000 | Historics agree with actual upload
```

**If Disagrees:**
```
⚠️ Historical Validation: Net Income per Financial Statements: R 1,000,000 | Calculated Net Income: R 950,000 | Difference: R 50,000 | Discrepancy detected - please review
```

## How It Works

1. **User imports financial statements:**
   - Income Statement, Balance Sheet, Cash Flow imported
   - Net income stored in `net_income` column

2. **Historical data is displayed:**
   - Actuals shown in income statement with "A" marker
   - Data includes all components: revenue, COGS, OPEX, interest, tax

3. **Validation runs automatically:**
   - After income statement table is rendered
   - Compares imported vs calculated net income
   - Shows result at bottom of statement

4. **User sees validation:**
   - Green success message if values agree
   - Yellow warning if discrepancy detected

## Benefits

1. **Data Quality Assurance:** Validates that imported data is internally consistent
2. **Error Detection:** Catches discrepancies in data import or calculation
3. **User Confidence:** Confirms that historical data is correctly imported
4. **Transparency:** Shows both imported and calculated values for comparison

## Tolerance Settings

- **1% tolerance:** Accounts for rounding differences in financial statements
- **Minimum 100 units:** Prevents false positives for very small values
- **Flexible:** Adapts to different currency scales

## Testing

To verify the fix:

1. **Import historical financial statements:**
   - Go to Setup → Historics
   - Import Income Statement with net income

2. **View financial statements:**
   - Go to Forecast Results → Financial Statements → Income Statement
   - Scroll to bottom

3. **Check validation:**
   - ✅ Should show "Historics agree with actual upload" if values match
   - ⚠️ Should show discrepancy warning if values don't match

4. **Test with different scenarios:**
   - Test with matching values (should show success)
   - Test with mismatched values (should show warning)
   - Test with annual vs monthly view

## Expected Behavior

- ✅ Validation appears at bottom of income statement
- ✅ Only shows for actuals periods (historical data)
- ✅ Shows imported net income from financial statements
- ✅ Shows calculated net income from components
- ✅ Compares with reasonable tolerance
- ✅ Displays clear success or warning message
- ✅ Works in both monthly and annual view

## Edge Cases Handled

1. **No actuals data:** Validation doesn't show
2. **Missing net_income column:** Validation doesn't show
3. **Missing components:** Uses 0 for missing values
4. **Multiple periods:** Aggregates totals for comparison
5. **Very small values:** Uses minimum tolerance to avoid false positives

---

**Date:** December 20, 2025  
**Status:** ✅ Fixed
