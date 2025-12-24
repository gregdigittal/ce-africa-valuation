# AFS Extract Format Review

**Date:** 2025-12-19  
**Location:** `/Users/gregmorris/Development Projects/CE Africa/CE Africa Files/AFS Extract`

## Summary

The AFS Extract files use a **vertical format** (line items as rows, periods as columns), while the model expects a **horizontal format** (one row per period, line items as columns).

## Format Comparison

### Current AFS Extract Format (Vertical)

**Income Statement:**
```
Line Item,Category,FY2024,FY2023_Restated,YoY_Change,YoY_Pct
Revenue,Revenue,176538186,128702399,47835787,37.2%
Cost of Sales,Cost of Sales,-131633550,-88208361,-43425189,49.2%
Gross Profit,Gross Profit,44904636,40494038,4410598,10.9%
...
```

**Balance Sheet:**
```
Line Item,Category,Sub_Category,FY2024,FY2023_Restated,Change,Change_Pct
Property Plant and Equipment,Non-Current Assets,PPE,1321393,1344906,-23513,-1.7%
Right-of-Use Assets,Non-Current Assets,Leases,3878831,6021132,-2142301,-35.6%
...
```

**Cash Flow:**
```
Line Item,Category,FY2024,FY2023_Restated,Change
Loss/Profit Before Taxation,Operating Activities,-962197,14608748,-15570945
Depreciation and Amortisation,Operating Activities - Adjustments,3703735,5454327,-1750592
...
```

### Expected Model Format (Horizontal)

**Income Statement (`historic_financials`):**
```csv
month,revenue,cogs,gross_profit,opex,ebit
2024-12-01,176538186,131633550,44904636,<opex_total>,<ebit>
2023-12-01,128702399,88208361,40494038,<opex_total>,<ebit>
```

**Balance Sheet (`historical_balance_sheet`):**
```csv
month,cash_and_equivalents,accounts_receivable,inventory,total_current_assets,ppe_net,total_assets,accounts_payable,short_term_debt,total_current_liabilities,long_term_debt,total_liabilities,share_capital,retained_earnings,total_equity
2024-12-01,0,44141309,50602109,105493771,1321393,115801076,<ap>,<std>,<tcl>,<ltd>,<tl>,24500,1588311,1612811
```

**Cash Flow (`historical_cashflow`):**
```csv
month,net_income,depreciation_amortization,change_in_receivables,change_in_inventory,change_in_payables,cash_from_operations,capital_expenditure,cash_from_investing,debt_repayment,dividends_paid,cash_from_financing,net_change_in_cash
2024-12-01,<ni>,3703735,-38568235,-29356513,25143229,-17356152,-1509677,16058075,-2807754
```

## Key Issues

### 1. **Structure Mismatch**
- **AFS Format:** One row per line item, multiple period columns (FY2024, FY2023, etc.)
- **Model Format:** One row per period, one column per line item

### 2. **Missing Required Fields**
- **Income Statement:** Missing `month` column (period identifier)
- **Balance Sheet:** Missing `month` column and many expected field names
- **Cash Flow:** Missing `month` column and some expected field names

### 3. **Field Name Mismatches**

#### Income Statement:
| AFS Extract | Model Expected | Notes |
|------------|----------------|-------|
| `Revenue` | `revenue` | ✓ Matches (case-insensitive) |
| `Cost of Sales` | `cogs` | Needs mapping |
| `Gross Profit` | `gross_profit` | Needs mapping |
| Various expense line items | `opex` | Need to sum all operating expenses |
| No EBIT column | `ebit` | Need to calculate: Gross Profit - OPEX |

#### Balance Sheet:
| AFS Extract | Model Expected | Notes |
|------------|----------------|-------|
| `Cash and Cash Equivalents` | `cash_and_equivalents` | Needs mapping |
| `Trade Receivables Net` | `accounts_receivable` | Needs mapping |
| `Total Inventories` | `inventory` | Needs mapping |
| `Total Current Assets` | `total_current_assets` | ✓ Matches |
| `Property Plant and Equipment` | `ppe_net` | Needs mapping |
| `Total Intangible Assets` | `intangible_assets` | Needs mapping |
| `Total Non-Current Assets` | `total_noncurrent_assets` | Needs mapping |
| `Total Assets` | `total_assets` | ✓ Matches |
| Various payables | `accounts_payable` | Need to identify and sum |
| `Lease Liabilities - Current` | `short_term_debt` | May need mapping |
| `Total Current Liabilities` | `total_current_liabilities` | ✓ Matches |
| `Lease Liabilities - Non-Current` | `long_term_debt` | May need mapping |
| `Total Non-Current Liabilities` | `total_noncurrent_liabilities` | ✓ Matches |
| `Total Liabilities` | `total_liabilities` | ✓ Matches |
| `Stated Capital` | `share_capital` | Needs mapping |
| `Retained Income/(Accumulated Loss)` | `retained_earnings` | Needs mapping |
| `Total Equity` | `total_equity` | ✓ Matches |

#### Cash Flow:
| AFS Extract | Model Expected | Notes |
|------------|----------------|-------|
| `Loss/Profit Before Taxation` | `net_income` | May need adjustment |
| `Depreciation and Amortisation` | `depreciation_amortization` | Needs mapping |
| `Increase in Trade and Other Receivables` | `change_in_receivables` | Sign may need adjustment |
| `Increase in Inventories` | `change_in_inventory` | Sign may need adjustment |
| `Increase in Trade and Other Payables` | `change_in_payables` | Sign may need adjustment |
| `Net Cash From Operating Activities` | `cash_from_operations` | Needs mapping |
| `Net Cash From Investing Activities` | `cash_from_investing` | Needs mapping |
| `Net Cash From Financing Activities` | `cash_from_financing` | Needs mapping |
| `Total Cash Movement for the Year` | `net_change_in_cash` | Needs mapping |

### 4. **Data Transformation Required**

1. **Transpose:** Convert from vertical (line items as rows) to horizontal (periods as rows)
2. **Aggregate:** Sum operating expenses into `opex` column
3. **Calculate:** Compute `ebit` = `gross_profit` - `opex`
4. **Map Fields:** Rename columns to match expected field names
5. **Add Period Column:** Create `month` column from filename (FY2024 → 2024-12-01)
6. **Handle Signs:** Ensure correct signs for cash flow changes (increases are negative for assets, positive for liabilities)

## Recommendations

### Option 1: Transform Files (Recommended)
Create a transformation script to convert AFS Extract format to model format:

1. **Read AFS files** (one per period)
2. **Transpose** to horizontal format
3. **Map field names** using lookup table
4. **Aggregate/calculate** derived fields
5. **Add period column** from filename
6. **Output** in model format

### Option 2: Enhance Import Function
Modify the import function to:
1. **Detect format** (vertical vs horizontal)
2. **Auto-transpose** if vertical format detected
3. **Auto-map fields** using enhanced field hints
4. **Handle aggregation** of line items into summary fields

### Option 3: Manual Transformation
Manually transform files in Excel/Google Sheets:
1. Transpose each file
2. Rename columns
3. Add period column
4. Aggregate expenses
5. Calculate derived fields

## Field Mapping Reference

### Income Statement Mapping
```python
{
    'Revenue': 'revenue',
    'Cost of Sales': 'cogs',
    'Gross Profit': 'gross_profit',
    # Sum all Operating Expenses rows → 'opex'
    # Calculate: 'ebit' = 'gross_profit' - 'opex'
}
```

### Balance Sheet Mapping
```python
{
    'Cash and Cash Equivalents': 'cash_and_equivalents',
    'Trade Receivables Net': 'accounts_receivable',
    'Total Inventories': 'inventory',
    'Total Current Assets': 'total_current_assets',
    'Property Plant and Equipment': 'ppe_net',
    'Total Intangible Assets': 'intangible_assets',
    'Total Non-Current Assets': 'total_noncurrent_assets',
    'Total Assets': 'total_assets',
    # Sum payables → 'accounts_payable'
    'Lease Liabilities - Current': 'short_term_debt',  # Or separate field
    'Total Current Liabilities': 'total_current_liabilities',
    'Lease Liabilities - Non-Current': 'long_term_debt',  # Or separate field
    'Total Non-Current Liabilities': 'total_noncurrent_liabilities',
    'Total Liabilities': 'total_liabilities',
    'Stated Capital': 'share_capital',
    'Retained Income/(Accumulated Loss)': 'retained_earnings',
    'Total Equity': 'total_equity'
}
```

### Cash Flow Mapping
```python
{
    'Loss/Profit Before Taxation': 'net_income',  # May need tax adjustment
    'Depreciation and Amortisation': 'depreciation_amortization',
    'Increase in Trade and Other Receivables': 'change_in_receivables',  # Note: negative for increase
    'Increase in Inventories': 'change_in_inventory',  # Note: negative for increase
    'Increase in Trade and Other Payables': 'change_in_payables',  # Note: positive for increase
    'Net Cash From Operating Activities': 'cash_from_operations',
    'Net Cash From Investing Activities': 'cash_from_investing',
    'Net Cash From Financing Activities': 'cash_from_financing',
    'Total Cash Movement for the Year': 'net_change_in_cash'
}
```

## Next Steps

1. **Create transformation script** to convert AFS format to model format
2. **Test with one file** to verify mapping accuracy
3. **Batch process** all files in AFS Extract folder
4. **Validate** transformed data against expected format
5. **Import** transformed files into model
