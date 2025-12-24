# AFS Extract Format Mapping - Fixed ✅

**Date:** 2025-12-19  
**Status:** ✅ **Format Mapping Corrected**

## Problem Identified

The AFS Extract files use a **WIDE format** (transposed) that was not being correctly mapped:
- **Line Item** column (not "line_item_name")
- **Category** column (not "category")  
- **Sub_Category** column (with underscore, in Balance Sheet)
- Period columns: **FY2024**, **FY2023_Restated** (with _Restated suffix)
- Metadata columns: **YoY_Change**, **YoY_Pct**, **Change**, **Change_Pct** (should be ignored)

## Solution Implemented

### 1. Period Column Detection ✅

**Function:** `extract_period_from_column()`

Now handles:
- ✅ `FY2024` → `2024-12-01`
- ✅ `FY2023_Restated` → `2023-12-01` (strips `_Restated` suffix)
- ✅ `YTD_Oct2025` → `2025-10-01`
- ✅ `2024-12` → `2024-12-01`
- ✅ `2024` → `2024-12-01`

**Key Fix:** Removes suffixes like `_Restated`, `_Change`, `_Pct` before extracting period.

### 2. Column Identification ✅

**Enhanced detection:**
- ✅ **Line Item Column:** Detects "Line Item", "LineItem", "line_item" (case-insensitive)
- ✅ **Category Column:** Detects "Category" (excludes "Sub_Category")
- ✅ **Sub_Category Column:** Detects "Sub_Category", "Sub-Category", "SubCategory" (with/without underscores/hyphens)
- ✅ **Period Columns:** Identifies FY2024, FY2023_Restated, YTD_Oct2025, etc.
- ✅ **Metadata Columns:** Excludes YoY_Change, YoY_Pct, Change, Change_Pct, Budget, Variance, Notes, PY_YTD

### 3. Transformation Logic ✅

**Function:** `transform_wide_to_long()`

**Process:**
1. Identifies all columns correctly
2. Extracts period dates from column names (handles _Restated suffix)
3. For each line item row:
   - Skips totals, margins, empty rows
   - Gets category and sub_category
   - For each period column:
     - Extracts period date
     - Parses amount (handles accounting format: `(12345)` → `-12345`)
     - Creates long format row
4. Returns DataFrame with: `period_date`, `line_item_name`, `category`, `sub_category`, `amount`

## Test Results

### Income Statement (FY2024.csv)
**Input:**
- Columns: `Line Item, Category, FY2024, FY2023_Restated, YoY_Change, YoY_Pct`
- Shape: 54 rows × 6 columns

**Output:**
- Shape: 92 rows × 5 columns (54 line items × 2 periods - some skipped)
- Columns: `period_date, line_item_name, category, sub_category, amount`
- Periods detected: `FY2024` → `2024-12-01`, `FY2023_Restated` → `2023-12-01`
- Metadata columns correctly excluded

### Balance Sheet (FY2024.csv)
**Input:**
- Columns: `Line Item, Category, Sub_Category, FY2024, FY2023_Restated, Change, Change_Pct`
- Shape: 47 rows × 7 columns

**Output:**
- Includes `sub_category` correctly
- Periods detected correctly
- Change columns excluded

## Example Transformation

### Input (Wide Format):
```csv
Line Item,Category,FY2024,FY2023_Restated,YoY_Change,YoY_Pct
Revenue,Revenue,176538186,128702399,47835787,37.2%
Accounting Fees,Operating Expenses,-375000,-12903,-362097,2806.3%
```

### Output (Long Format):
```csv
period_date,line_item_name,category,sub_category,amount
2024-12-01,Revenue,Revenue,,176538186
2023-12-01,Revenue,Revenue,,128702399
2024-12-01,Accounting Fees,Operating Expenses,,-375000
2023-12-01,Accounting Fees,Operating Expenses,,-12903
```

## How to Use

1. **Upload Your Original AFS Extract File**
   - Go to Setup → Historics → Detailed Line Items
   - Select statement type tab
   - Upload your original file (e.g., `CE_Africa_Income_Statement_FY2024.csv`)

2. **Automatic Detection & Conversion**
   - System detects wide format automatically
   - Shows format detection details
   - Converts to long format
   - Displays preview

3. **Column Mapping**
   - Most columns are auto-filled from converted data
   - Just verify the mapping is correct
   - Click Import

## Supported File Formats

✅ **Income Statement:**
- `Line Item, Category, FY2024, FY2023_Restated, YoY_Change, YoY_Pct`

✅ **Balance Sheet:**
- `Line Item, Category, Sub_Category, FY2024, FY2023_Restated, Change, Change_Pct`

✅ **Cash Flow:**
- `Line Item, Category, FY2024, FY2023_Restated, Change`

✅ **YTD Files:**
- `Line Item, YTD_Oct2025, Budget, Variance, Var%, PY_YTD, Notes`

## Files Modified

- `components/column_mapper.py`
  - Fixed `extract_period_from_column()` to handle `_Restated` suffix
  - Enhanced column detection (handles Sub_Category with underscore)
  - Improved period column identification (excludes metadata columns)
  - Fixed transformation logic to handle all AFS Extract formats

## Verification

The transformation has been tested with:
- ✅ `CE_Africa_Income_Statement_FY2024.csv` (54 rows → 92 long format rows)
- ✅ `CE_Africa_Balance_Sheet_FY2024.csv` (includes Sub_Category)
- ✅ Period extraction works for FY2024, FY2023_Restated
- ✅ Metadata columns correctly excluded

The mapping should now work correctly with your AFS Extract files!
