# Wide Format Import Guide - Automatic Conversion ✅

**Date:** 2025-12-19  
**Status:** ✅ **Automatic Conversion Implemented**

## Problem

Your AFS Extract files are in **WIDE format** (transposed):
- Each **row** = one line item
- Each **column** = one period (FY2024, FY2023, etc.)

But the database expects **LONG format**:
- Each **row** = one line item for one period
- Columns: period_date, line_item_name, category, amount

## Solution: Automatic Detection & Conversion

The import system now **automatically detects** wide format and converts it to long format during import.

### How It Works

1. **Upload your CSV file** (wide format)
2. **System detects** wide format automatically
3. **Converts** to long format
4. **Shows preview** of transformed data
5. **Maps columns** and imports

## Example

### Your File (Wide Format):
```csv
Line Item,Category,FY2024,FY2023_Restated
Revenue,Revenue,176538186,128702399
Cost of Sales,Cost of Sales,-131633550,-88208361
Accounting Fees,Operating Expenses,-375000,-12903
Advertising,Operating Expenses,-1445954,-1581995
Depreciation,Operating Expenses,-742779,-1050310
```

### Automatically Converted To (Long Format):
```csv
period_date,line_item_name,category,amount
2024-12-01,Revenue,Revenue,176538186
2023-12-01,Revenue,Revenue,128702399
2024-12-01,Cost of Sales,Cost of Sales,-131633550
2023-12-01,Cost of Sales,Cost of Sales,-88208361
2024-12-01,Accounting Fees,Operating Expenses,-375000
2023-12-01,Accounting Fees,Operating Expenses,-12903
2024-12-01,Advertising,Operating Expenses,-1445954
2023-12-01,Advertising,Operating Expenses,-1581995
2024-12-01,Depreciation,Operating Expenses,-742779
2023-12-01,Depreciation,Operating Expenses,-1050310
```

## Supported Period Formats

The system automatically recognizes period columns in these formats:

1. **FY2024** → Converts to `2024-12-01` (year-end)
2. **YTD_Oct2025** → Converts to `2025-10-01` (year-to-date)
3. **2024-12** → Converts to `2024-12-01` (month-year)
4. **2024** → Converts to `2024-12-01` (year only)

## How to Use

### Step 1: Upload Your File
1. Go to **Setup → Historics → Detailed Line Items**
2. Select the statement type tab (Income Statement, Balance Sheet, or Cash Flow)
3. Click **"Upload CSV file"**
4. Select your wide format file

### Step 2: Automatic Detection
The system will:
- ✅ Detect wide format automatically
- ✅ Show info message: "📊 Wide Format Detected"
- ✅ Display format detection details
- ✅ Convert to long format
- ✅ Show preview of transformed data

### Step 3: Column Mapping
Map the columns:
- **Period Date** → Auto-filled from converted data
- **Line Item Name** → Map to "Line Item" column
- **Category** → Map to "Category" column
- **Sub-Category** → Map to "Sub-Category" (if available)
- **Amount** → Auto-filled from converted data

### Step 4: Import
Click **"🚀 Import"** - the system handles everything!

## What Gets Converted

**Input (Wide):**
- 50 line items × 4 periods = 50 rows

**Output (Long):**
- 50 line items × 4 periods = **200 rows** (one per line item per period)

## Format Detection Logic

The system detects wide format if:
1. ✅ Has a column with "line" and "item" in the name
2. ✅ Has columns that look like periods (FY2024, YTD_Oct2025, 2024, etc.)
3. ✅ Optional: Has a "Category" column

## Benefits

1. ✅ **No Manual Transformation** - Upload your original files directly
2. ✅ **Automatic Detection** - System recognizes wide format automatically
3. ✅ **Preview Before Import** - See transformed data before importing
4. ✅ **Handles Multiple Periods** - Converts all period columns at once
5. ✅ **Preserves All Data** - Every line item × every period is preserved

## Example Workflow

**Before (Manual):**
1. Download transformation script
2. Run Python script
3. Get transformed CSV
4. Import transformed CSV

**Now (Automatic):**
1. Upload original wide format CSV
2. System converts automatically
3. Import!

## Troubleshooting

### If Wide Format Not Detected

If your file isn't automatically detected as wide format:
1. Check column names - ensure you have a "Line Item" column
2. Check period columns - ensure they follow formats: FY2024, YTD_Oct2025, 2024-12, or 2024
3. You can still import manually by using the long format template

### If Period Dates Are Wrong

The system extracts periods from column names:
- `FY2024` → Assumes year-end: `2024-12-01`
- `YTD_Oct2025` → Uses month from name: `2025-10-01`
- `2024-12` → Uses exact month: `2024-12-01`
- `2024` → Assumes year-end: `2024-12-01`

If you need different dates, you can:
1. Rename columns before import (e.g., `FY2024` → `2024-06-01`)
2. Or import as long format and set dates manually

## Technical Details

**Function:** `transform_wide_to_long()`

**Process:**
1. Identifies line item column (e.g., "Line Item")
2. Identifies category column (e.g., "Category")
3. Identifies period columns (FY2024, YTD_Oct2025, etc.)
4. For each line item row:
   - For each period column:
     - Extracts period date from column name
     - Creates a new row: period_date, line_item_name, category, amount
5. Returns long format DataFrame

**Handles:**
- Accounting number format: `(12345)` → `-12345`
- Empty values: Skips empty cells
- Totals/Margins: Skips rows with "Total" or "%" in name
- Multiple periods: Converts all period columns

## Files Modified

- `components/column_mapper.py`
  - Added `extract_period_from_column()` function
  - Added `transform_wide_to_long()` function
  - Updated `render_import_with_mapping()` to detect and convert wide format
  - Updated field descriptions to mention wide format support

## Next Steps

1. ✅ **Test with Your Files** - Upload your original AFS Extract files
2. ✅ **Verify Conversion** - Check the preview shows correct long format
3. ✅ **Import** - Complete the import process
4. ✅ **Run AI Analysis** - Analyze the imported detailed line items
