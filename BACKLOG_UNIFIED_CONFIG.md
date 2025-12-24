# Unified Configuration Backlog

## Current Implementation (December 20, 2025)

### Delivered Features ✅

1. **Unified Line-Item Configuration UI** ✅
   - Single table showing all configurable line items
   - Columns: Line Item, Category, Trend Type, Growth %, Use MC, Distribution, CV%
   - Batch editing with st.data_editor
   - Single save button for all changes
   - Filter by category

2. **Forecast Method Toggle** ✅
   - Clear radio selection: Pipeline-Based vs Trend-Based
   - Persists to assumptions database
   - Displayed in Forecast → Run tab
   - Respected by forecast engine

3. **Aggregates Calculated from Line Items** ✅
   - Aggregates preview shows how line items sum
   - No more conflicting configuration paths

4. **Legacy Compatibility** ✅
   - Old tabs (Trend Forecast, Distributions) hidden by default
   - Toggle to show legacy tabs if needed
   - Marked as "Legacy" with warnings

5. **Line-Item Level Forecast Engine** ✅ (Phase 2)
   - Each line item forecast individually with its own trend
   - Aggregates calculated from line item sums
   - MC simulation at line-item level
   - Falls back to legacy if unified config empty

6. **Correlation Configuration** ✅ (Phase 3)
   - Configure correlations between line items
   - Presets: High correlation, Low correlation, Independent
   - UI to add/remove individual correlation pairs
   - Integrates with MC sampling

7. **Scenario Comparison** ✅ (Phase 4)
   - Run both Pipeline and Trend methods
   - Side-by-side comparison table
   - Variance analysis
   - Chart overlays
   - Method recommendations

8. **Storage Migration Utility** ✅ (Tech Debt)
   - Analyze current storage state
   - Dry run migration
   - Migrate legacy data to unified config
   - Archives legacy data

---

## Backlog Enhancements

### Phase 1: Monthly Sales-Driven Upsampling
**Priority:** High  
**Trigger:** When monthly sales data is available
**Status:** PENDING

**Description:**
When the user has some monthly sales data (e.g., from sales orders or invoices), use the monthly sales pattern to upsample revenue line items rather than pro-rata. Overheads should still be pro-rated evenly as they are typically static month-on-month.

**Implementation:**
1. Detect if monthly sales data exists (e.g., from `sales_orders` or similar table)
2. Extract monthly seasonality pattern from sales data
3. Apply pattern to revenue line items during upsampling
4. Keep OPEX line items pro-rata (12ths for annual, Nths for YTD)

---

## Technical Debt (Remaining)

1. **Test Coverage**
   - Add unit tests for LineItemConfig
   - Add integration tests for unified config save/load
   - Add tests for forecast method toggle
   - Add tests for correlation sampling

2. **Performance Optimization**
   - Cache line item forecasts
   - Lazy load historical data
   - Optimize MC sampling for large line item counts
