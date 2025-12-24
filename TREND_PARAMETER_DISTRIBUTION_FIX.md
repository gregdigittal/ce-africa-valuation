# Trend Parameter Distribution Application - Fix

## Problem
When setting financial assumptions with distributions, the distributions were not being applied to trend parameters in Monte Carlo simulations. The simulation should calculate figures in relation to the trend data points by varying the trend parameters according to the fitted distributions.

## Requirements
1. When a distribution is set for a financial assumption (e.g., revenue, COGS, OPEX)
2. And that assumption uses a trend-based forecast (trend_fit method)
3. The Monte Carlo simulation should sample from the distribution and apply it to the trend parameters
4. The simulation should calculate figures that vary around the trend according to the distribution

## Solution

### 1. Added Helper Functions

**File:** `components/forecast_section.py`

#### `get_distribution_for_element()`
- Maps forecast element names to AI assumption keys
- Retrieves distribution parameters for a given element
- Returns `DistributionParams` if available, `None` otherwise

#### `sample_trend_multiplier_from_distribution()`
- Samples a single value from a distribution
- Calculates the expected value based on distribution type
- Returns a multiplier (sample / expected) that can be applied to trend forecasts
- Clamps multiplier to reasonable range (0.5x to 2.0x)

### 2. Enhanced Monte Carlo Function

**File:** `components/forecast_section.py` - `run_monte_carlo_enhanced()`

**New Parameters:**
- `forecast_configs: Optional[Dict]` - Trend forecast configurations
- `historical_data: Optional[pd.DataFrame]` - Historical data for trend generation

**New Logic:**
1. **Check for trend-based forecasts with distributions:**
   ```python
   use_trend_distributions = (
       forecast_configs is not None 
       and ai_assumptions is not None
       and hasattr(ai_assumptions, 'assumptions_saved') 
       and ai_assumptions.assumptions_saved
   )
   ```

2. **Get distributions for trend-based elements:**
   - Iterate through `forecast_configs`
   - For elements with `method == 'trend_fit'`
   - Get distribution from AI assumptions
   - Store in `trend_distributions` dict

3. **In each Monte Carlo iteration:**
   - Sample multipliers from distributions for revenue, COGS, OPEX
   - Apply multipliers to base forecasts
   - This varies the trend according to the distribution

**Example:**
```python
# If revenue uses trend_fit and has a distribution
if 'revenue' in trend_distributions:
    dist = trend_distributions['revenue']
    revenue_multiplier = sample_trend_multiplier_from_distribution(dist, i)
    fleet_factor = np.ones(n_months) * revenue_multiplier
```

### 3. Updated Function Calls

**File:** `components/forecast_section.py` - Forecast tab

**Changes:**
- Load `forecast_configs` from assumptions
- Load `historical_data` if `forecast_configs` exist
- Pass both to `run_monte_carlo_enhanced()`

## How It Works

1. **User sets up trend forecasts:**
   - Configures trend functions (linear, exponential, etc.) in Forecast Configuration
   - Adjusts trend parameters (slope, intercept, growth_rate, etc.)

2. **User sets distributions:**
   - AI Assumptions fits distributions to historical data
   - Distributions are stored in `AssumptionsSet`

3. **Monte Carlo simulation:**
   - Detects elements using `trend_fit` method
   - Finds corresponding distributions in AI assumptions
   - For each iteration:
     - Samples a multiplier from the distribution
     - Applies multiplier to the base trend forecast
     - This creates variation around the trend according to the distribution

4. **Result:**
   - Forecasts vary according to both:
     - The trend function (deterministic growth pattern)
     - The distribution (stochastic variation around the trend)

## Example

**Scenario:**
- Revenue uses linear trend: `y = 1000 + 50x`
- Revenue has normal distribution: `N(mean=1000, std=100)`

**Monte Carlo Iteration 1:**
- Sample from distribution: `sample = 1050`
- Expected value: `expected = 1000`
- Multiplier: `multiplier = 1050 / 1000 = 1.05`
- Applied forecast: `y = (1000 + 50x) * 1.05`

**Monte Carlo Iteration 2:**
- Sample from distribution: `sample = 950`
- Multiplier: `multiplier = 950 / 1000 = 0.95`
- Applied forecast: `y = (1000 + 50x) * 0.95`

**Result:**
- All forecasts follow the trend pattern (linear growth)
- But vary up/down according to the distribution
- Creates realistic uncertainty around the trend

## Benefits

1. **Realistic Uncertainty:** Forecasts vary around trends in a statistically sound way
2. **Distribution-Driven:** Uses fitted distributions from historical data
3. **Trend-Preserving:** Maintains the trend pattern while adding variation
4. **Flexible:** Works with any trend function type (linear, exponential, logarithmic, etc.)
5. **Efficient:** Doesn't require regenerating entire forecasts for each iteration

## Testing

To verify the fix:

1. **Set up trend forecasts:**
   - Go to Forecast Configuration
   - Configure revenue, COGS, OPEX with trend_fit method
   - Adjust trend parameters

2. **Set up distributions:**
   - Run AI Assumptions
   - Ensure distributions are fitted for revenue, COGS, OPEX

3. **Run Monte Carlo:**
   - Enable Monte Carlo simulation
   - Check that results vary around the trend
   - Verify that variation matches the distribution

4. **Check results:**
   - Percentiles should show variation around the trend
   - Distribution should match the fitted distribution
   - Trend pattern should be preserved

## Expected Behavior

- ✅ Distributions are applied to trend-based forecasts
- ✅ Monte Carlo samples from distributions for each iteration
- ✅ Forecasts vary around the trend according to the distribution
- ✅ Trend pattern is preserved (linear growth, exponential, etc.)
- ✅ Works with all trend function types
- ✅ Efficient (doesn't regenerate forecasts)

---

**Date:** December 20, 2025  
**Status:** ✅ Fixed
