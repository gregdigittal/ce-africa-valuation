# Fix Summary - December 20, 2025

## Issues Addressed

### 1. Trend Forecast Fallback Behavior ✅ FIXED

**User Requirement:** When a trend is selected but fails, throw an error showing where the problem is (don't fall back to pipeline silently).

**Changes Made:**

**File: `forecast_engine.py`**

1. **Main trend path (lines 166-214):**
   - Removed silent fallback to pipeline
   - Added detailed error display with:
     - Element name
     - Method configured  
     - Trend type
     - Parameters
     - Specific error (None result, length mismatch)
   - Added "How to Fix" guidance
   - Returns `success: False` with error message

2. **Legacy trend path (lines 265-294):**
   - Removed silent fallback
   - Added error for insufficient historical data
   - Added error for missing historical data

3. **COGS trend failure (line 971-984):**
   - Added warning message when COGS trend fails
   - Still uses margin-based calculation but user is informed

4. **OPEX trend failure (line 1057-1068):**
   - Added warning message when OPEX trend fails
   - Still uses expense-based calculation but user is informed

---

### 2. Calculated Elements Still Appearing in UI ✅ FIXED

**User Requirement:** Calculated elements (Gross Profit, EBIT, Tax, Net Income) should NOT appear in the Financial Assumptions configuration UI.

**Root Cause:** Key mismatch - `FORECAST_ELEMENTS` has keys like `gross_profit` but assumptions might have:
- Different keys: `total_gross_profit`
- Display names: "Gross Profit"

**Changes Made:**

**File: `components/ai_assumptions_engine.py`**

1. **Created robust `is_calculated_element()` function** with:

```python
CALCULATED_KEY_PATTERNS = {
    'gross_profit', 'total_gross_profit', 'gp',
    'ebit', 'operating_income', 'operating_profit',
    'ebitda',
    'interest_expense', 'interest', 'finance_cost', 'total_interest_expense',
    'tax', 'income_tax', 'tax_expense', 'taxation', 'total_tax',
    'net_profit', 'net_income', 'profit_after_tax', 'pat',
    'ebt', 'profit_before_tax', 'pbt',
}

CALCULATED_DISPLAY_PATTERNS = [
    'gross profit', 'gross margin', 
    'ebit', 'operating income', 'operating profit',
    'ebitda',
    'interest expense', 'finance cost',
    'tax', 'taxation', 'income tax',
    'net profit', 'net income', 'profit after tax',
    'profit before tax', 'ebt',
]
```

2. **Filter checks BOTH key AND display_name:**
   - Direct key match
   - Normalized key (removes `total_` prefix)
   - Display name pattern matching
   - FORECAST_ELEMENTS lookup

3. **Updated UI header** to clarify purpose:
   - Clear info box explaining Monte Carlo distribution configuration
   - Explains that calculated elements are excluded
   - Explains distributions are applied around trend forecasts

---

## Files Changed

| File | Changes |
|------|---------|
| `forecast_engine.py` | Removed silent fallback, added error messages |
| `components/ai_assumptions_engine.py` | Robust calculated element filtering |
| `REQUIREMENTS_DOCUMENT.md` | Updated T5, T7 requirements |
| `CODE_REVIEW_SUMMARY.md` | Updated fix descriptions |

---

## Verification Steps

### Test Trend Forecast Error Handling:
1. Enable "Use Trend Forecast" checkbox
2. Configure an invalid or incomplete trend for revenue
3. Run the forecast
4. **Expected:** Error message with diagnostic details, NOT silent fallback to pipeline

### Test Calculated Elements Excluded:
1. Go to AI Assumptions → Financial Assumptions tab
2. **Expected:** Should see ONLY:
   - Total Revenue
   - COGS % of Revenue  
   - OPEX % of Revenue
   - Depreciation (if in data)
   - Derived percentages (Gross Margin %, Revenue Growth %)
3. **Should NOT see:**
   - Gross Profit
   - EBIT
   - Interest Expense
   - Tax
   - Net Profit / Net Income

---

## Logic Flow

```
Financial Assumptions UI
         │
         ▼
for each assumption in assumptions_set:
         │
         ▼
is_calculated_element(key, display_name)?
         │
    ┌────┴────┐
   Yes        No
    │          │
    ▼          ▼
  SKIP      Display in UI
```

```
Trend Forecast Enabled?
         │
    ┌────┴────┐
   Yes        No
    │          │
    ▼          ▼
Generate    Use Pipeline
 Trend         (OK)
    │
    ▼
Success?
    │
┌───┴───┐
Yes     No
 │       │
 ▼       ▼
Use    ERROR
Trend  (show details,
       stop forecast)
```
