# Monte Carlo Period-Specific Distributions

## Problem Statement

Previously, the Monte Carlo simulation used historical mean and standard deviation for all periods, which caused the MC ranges to be wildly different from the fitted forecast values. For example:
- Historical mean: 100m, stdev: 15m
- Year 3 forecast: 200m
- MC was using: mean=100m, stdev=15m (wrong!)
- Should use: mean=200m, stdev=30m (proportionally scaled)

Also, calculated elements (EBIT, gross_profit, etc.) should not have MC parameters since they're derived from other simulated values.

## Solution

### 1. Period-Specific Distributions

**Principle:** For each period, use the fitted forecast value as the new mean, and proportionally scale the standard deviation.

**Formula:**
```
new_mean = forecast_value_for_period
new_stdev = historical_stdev * (forecast_value / historical_mean)
```

**Example:**
- Historical: mean=100m, stdev=15m
- Period 3 forecast: 200m
- MC uses: mean=200m, stdev=30m (15m * 200m/100m)

### 2. Exclude Calculated Elements

Calculated elements (EBIT, gross_profit, interest_expense, tax, net_profit) are **not** sampled from distributions. They are calculated from other simulated values:
- `gross_profit = revenue - cogs`
- `ebit = gross_profit - opex - depreciation`
- `interest_expense = calculated_from_balance_sheet`
- `tax = max(ebt * tax_rate, 0)`
- `net_profit = ebit - interest_expense - tax`

## Implementation

### New Functions

#### `get_historical_distribution_params(distribution_params)`
Extracts historical mean and standard deviation from distribution parameters for different distribution types:
- Normal: Direct mean/std
- Lognormal: Converts to normal space
- Triangular: Calculates mean and approximate std
- Beta: Calculates mean and variance
- Uniform: Calculates mean and std

#### `sample_from_period_specific_distribution(...)`
Samples from a distribution centered on the forecast value with proportionally scaled standard deviation:
- Uses forecast value as mean
- Scales stdev: `scaled_std = historical_std * (forecast_value / historical_mean)`
- Handles different distribution types (normal, lognormal, triangular, beta, uniform)
- Ensures reproducibility with seed offsets

### Updated Monte Carlo Logic

**File:** `components/forecast_section.py` - `run_monte_carlo_enhanced()`

**Key Changes:**

1. **Extract Historical Parameters:**
   ```python
   historical_params[element_name] = {
       'mean': hist_mean,
       'std': hist_std,
       'distribution_type': dist.distribution_type,
       'distribution_params': dist
   }
   ```

2. **Skip Calculated Elements:**
   ```python
   from components.forecast_correlation_engine import FORECAST_ELEMENTS
   element_def = FORECAST_ELEMENTS.get(element_name, {})
   if element_def.get('is_calculated', False):
       continue  # Skip calculated elements
   ```

3. **Period-by-Period Sampling:**
   ```python
   for month_idx in range(n_months):
       forecast_val = base_cogs[month_idx]
       seed_offset = i * n_months + month_idx
       sampled_val = sample_from_period_specific_distribution(
           forecast_val, hist_mean, hist_std, dist_type, dist_params, seed_offset
       )
       cogs_sim[month_idx] = sampled_val
   ```

4. **Calculated Elements:**
   ```python
   # No MC parameters - calculated from simulated values
   gp_sim = revenue_sim - cogs_sim
   ebit_sim = gp_sim - opex_sim
   ```

## Benefits

1. **Accurate MC Ranges:** MC simulation now centers on forecast values, not historical values
2. **Proportional Scaling:** Standard deviation scales with forecast growth
3. **Correct Dependencies:** Calculated elements are derived from simulated values, not sampled independently
4. **Reproducibility:** Seed offsets ensure consistent results across runs

## Example

**Scenario:**
- Historical revenue: mean=100m, stdev=15m
- Forecast Year 1: 120m
- Forecast Year 3: 200m

**Old Approach:**
- All periods: mean=100m, stdev=15m
- Year 3 MC: ~85m-115m (centered on 100m, wrong!)

**New Approach:**
- Year 1: mean=120m, stdev=18m (15m * 120m/100m)
- Year 3: mean=200m, stdev=30m (15m * 200m/100m)
- Year 3 MC: ~170m-230m (centered on 200m, correct!)

## Testing

To verify:
1. Run forecast with trend-based assumptions
2. Check MC results - should center on forecast values
3. Verify calculated elements (EBIT, GP) are not sampled independently
4. Confirm MC ranges scale proportionally with forecast growth

---

**Date:** December 20, 2025  
**Status:** ✅ Implemented
