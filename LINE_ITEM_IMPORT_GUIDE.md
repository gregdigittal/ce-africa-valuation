# Line Item Import Guide

## ✅ All Three Statement Types Supported

The import system automatically handles **transposed (wide format)** files for all three statement types:

1. **Income Statement Line Items** ✅
2. **Balance Sheet Line Items** ✅  
3. **Cash Flow Line Items** ✅

## Supported File Formats

All three types support the same wide format structure:

### Income Statement Format
```csv
Line Item,Category,FY2024,FY2023_Restated,YoY_Change,YoY_Pct
Revenue,Revenue,176538186,128702399,47835787,37.2%
Accounting Fees,Operating Expenses,-375000,-12903,-362097,2806.3%
```

### Balance Sheet Format
```csv
Line Item,Category,Sub_Category,FY2024,FY2023_Restated,Change,Change_Pct
Property Plant and Equipment,Non-Current Assets,PPE,1321393,1344906,-23513,-1.7%
Right-of-Use Assets,Non-Current Assets,Leases,3878831,6021132,-2142301,-35.6%
```

### Cash Flow Format
```csv
Line Item,Category,FY2024,FY2023_Restated,Change
Loss/Profit Before Taxation,Operating Activities,-962197,14608748,-15570945
Depreciation and Amortisation,Operating Activities - Adjustments,3703735,5454327,-1750592
```

## How to Import

### Step 1: Navigate to Setup
1. Go to **Setup** → **Historics** → **Detailed Line Items** tab
2. You'll see three sub-tabs:
   - **Income Statement**
   - **Balance Sheet**
   - **Cash Flow**

### Step 2: Upload Your File
1. Select the appropriate sub-tab (e.g., "Balance Sheet")
2. Click **Upload CSV file**
3. Select your AFS Extract file (e.g., `CE_Africa_Balance_Sheet_FY2024.csv`)

### Step 3: Automatic Transformation
The system will:
- ✅ **Detect** wide format automatically
- ✅ **Show** format detection details
- ✅ **Convert** to long format automatically
- ✅ **Display** preview of transformed data

### Step 4: Verify Mapping
1. Review the column mapping (usually auto-filled correctly)
2. Verify the mapped fields:
   - `period_date` → Period columns (FY2024, FY2023_Restated, etc.)
   - `line_item_name` → Line Item column
   - `category` → Category column
   - `sub_category` → Sub_Category column (Balance Sheet only)
   - `amount` → Period value columns

### Step 5: Import
1. Click **🚀 Import [Statement Type] Line Items**
2. Wait for processing
3. Check success message

## What Gets Imported

For each file, the system:
- Extracts all period columns (FY2024, FY2023_Restated, YTD_Oct2025, etc.)
- Converts each line item × period combination to a separate row
- Handles accounting format: `(12345)` → `-12345`
- Skips totals, margins, and empty rows
- Preserves categories and sub-categories

## Example Transformation

### Input (Wide Format - Balance Sheet):
```csv
Line Item,Category,Sub_Category,FY2024,FY2023_Restated
Property Plant and Equipment,Non-Current Assets,PPE,1321393,1344906
Right-of-Use Assets,Non-Current Assets,Leases,3878831,6021132
```

### Output (Long Format - Saved to Database):
```csv
period_date,line_item_name,category,sub_category,amount
2024-12-01,Property Plant and Equipment,Non-Current Assets,PPE,1321393
2023-12-01,Property Plant and Equipment,Non-Current Assets,PPE,1344906
2024-12-01,Right-of-Use Assets,Non-Current Assets,Leases,3878831
2023-12-01,Right-of-Use Assets,Non-Current Assets,Leases,6021132
```

## Supported Period Formats

The system recognizes these period column formats:
- ✅ `FY2024` → `2024-12-01`
- ✅ `FY2023_Restated` → `2023-12-01` (handles `_Restated` suffix)
- ✅ `YTD_Oct2025` → `2025-10-01`
- ✅ `2024-12` → `2024-12-01`
- ✅ `2024` → `2024-12-01`

## Files Tested

✅ **Income Statement:**
- `CE_Africa_Income_Statement_FY2024.csv` (54 rows → 92 long format rows)
- `CE_Africa_Income_Statement_FY2022.csv`
- `CE_Africa_Income_Statement_FY2023.csv`
- `CE_Africa_Income_Statement_YTD_Oct2025.csv`

✅ **Balance Sheet:**
- `CE_Africa_Balance_Sheet_FY2024.csv` (45 rows → 72 long format rows)
- `CE_Africa_Balance_Sheet_FY2022.csv`
- `CE_Africa_Balance_Sheet_FY2023.csv`
- `CE_Africa_Balance_Sheet_YTD_Oct2025.csv`

✅ **Cash Flow:**
- `CE_Africa_Cash_Flow_FY2024.csv` (25 rows → 48 long format rows)
- `CE_Africa_Cash_Flow_FY2022.csv`
- `CE_Africa_Cash_Flow_FY2023.csv`
- `CE_Africa_Cash_Flow_YTD_Oct2025.csv`

## Troubleshooting

### Error: "Table Not Found"
- Run the migration: `python3 run_migration.py migrations_add_detailed_line_items.sql`
- Then run the fix: `python3 run_migration.py migrations_fix_line_items_user_fkey.sql`

### Error: "Foreign Key Constraint"
- The fix migration removes the constraint that was causing issues
- Run: `python3 run_migration.py migrations_fix_line_items_user_fkey.sql`

### Import Shows "All Failed"
- Check the error messages (they now persist and are visible)
- Common issues:
  - Missing `scenario_id` (make sure a scenario is selected)
  - Invalid date formats
  - Missing required fields

## Next Steps After Import

Once imported, the detailed line items can be:
1. **Used for AI Assumptions** - The AI will analyze each line item individually
2. **Forecasted** - Generate forecasts for each detailed line item
3. **Aggregated** - Automatically aggregated back to summary totals for reporting

## Summary

✅ **All three statement types work the same way**
✅ **Automatic wide-to-long format conversion**
✅ **No manual transformation needed**
✅ **Just upload your original AFS Extract files**

The system handles everything automatically! 🎉
