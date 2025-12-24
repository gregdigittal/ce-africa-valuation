# What-If Agent Snapshot Support Fix
**Date:** December 16, 2025  
**Status:** ✅ **FIXED**

---

## Problem

The What-If Agent showed the message:
> "💡 No Baseline Forecast Found - Run a forecast first to create a baseline, then use the What-If Agent to explore different scenarios."

This occurred even when snapshot valuations existed in the database. The What-If Agent was only checking for forecast results in session state (`st.session_state.get('forecast_results')`), but not checking for saved snapshots.

---

## Root Cause

The What-If Agent section in `app_refactored.py` was only checking for baseline forecast in one location:

```python
# Get baseline forecast
forecast_key = f"forecast_results_{scenario_id}"
baseline_forecast = st.session_state.get(forecast_key)

render_whatif_agent(
    db=db,
    scenario_id=scenario_id,
    user_id=user_id,
    baseline_forecast=baseline_forecast  # ❌ Only checks session state
)
```

**Issues:**
1. Only checked `forecast_results_{scenario_id}` in session state
2. Didn't check `forecast_results` (alternative key)
3. Didn't check for saved snapshots in the database
4. Session state clears on page reload, so saved snapshots weren't accessible

---

## Solution

Updated the What-If Agent section to check multiple sources for baseline forecast:

**File:** `app_refactored.py` (lines ~1998-2025)

**Changes:**
1. **Check session state with scenario-specific key:**
   ```python
   forecast_key = f"forecast_results_{scenario_id}"
   baseline_forecast = st.session_state.get(forecast_key)
   ```

2. **Fallback to generic session state key:**
   ```python
   if not baseline_forecast:
       baseline_forecast = st.session_state.get('forecast_results')
   ```

3. **Load from latest snapshot if not in session state:**
   ```python
   if not baseline_forecast:
       try:
           from components.forecast_section import load_snapshots
           import json
           
           snapshots = load_snapshots(db, scenario_id, limit=1)
           if snapshots:
               snapshot = snapshots[0]
               forecast_data = json.loads(snapshot.get('forecast_data', '{}'))
               baseline_forecast = {
                   'success': True,
                   'timeline': forecast_data.get('timeline', []),
                   'revenue': forecast_data.get('revenue', {}),
                   'costs': forecast_data.get('costs', {}),
                   'profit': forecast_data.get('profit', {}),
                   'summary': json.loads(snapshot.get('summary_stats', '{}'))
               }
       except Exception:
           baseline_forecast = None
   ```

**Complete Updated Code:**
```python
if WHATIF_AGENT_AVAILABLE:
    # Get baseline forecast from session state first
    forecast_key = f"forecast_results_{scenario_id}"
    baseline_forecast = st.session_state.get(forecast_key)
    
    # If not in session state, try loading from latest snapshot
    if not baseline_forecast:
        baseline_forecast = st.session_state.get('forecast_results')
    
    # If still not found, try loading from database snapshots
    if not baseline_forecast:
        try:
            from components.forecast_section import load_snapshots
            import json
            
            snapshots = load_snapshots(db, scenario_id, limit=1)
            if snapshots:
                snapshot = snapshots[0]
                try:
                    forecast_data = json.loads(snapshot.get('forecast_data', '{}'))
                    baseline_forecast = {
                        'success': True,
                        'timeline': forecast_data.get('timeline', []),
                        'revenue': forecast_data.get('revenue', {}),
                        'costs': forecast_data.get('costs', {}),
                        'profit': forecast_data.get('profit', {}),
                        'summary': json.loads(snapshot.get('summary_stats', '{}'))
                    }
                except Exception as e:
                    baseline_forecast = None
        except Exception:
            baseline_forecast = None
    
    render_whatif_agent(
        db=db,
        scenario_id=scenario_id,
        user_id=user_id,
        baseline_forecast=baseline_forecast
    )
```

---

## How It Works

### Baseline Forecast Lookup Priority:

1. **Session State (Scenario-Specific):** `forecast_results_{scenario_id}`
   - Used when forecast was just run in current session
   - Fastest access

2. **Session State (Generic):** `forecast_results`
   - Fallback for older code that uses generic key
   - Still fast, no database access

3. **Latest Snapshot (Database):**
   - Loads most recent snapshot for the scenario
   - Converts snapshot format to forecast results format
   - Works even after page reload

4. **None:**
   - If no baseline found, What-If Agent shows the "No Baseline Forecast Found" message

### Snapshot Format Conversion:

Snapshots are stored in the database with:
- `forecast_data` (JSON string): Contains timeline, revenue, costs, profit
- `summary_stats` (JSON string): Contains summary statistics

These are converted to the format expected by the What-If Agent:
```python
{
    'success': True,
    'timeline': [...],
    'revenue': {...},
    'costs': {...},
    'profit': {...},
    'summary': {...}
}
```

---

## Benefits

### ✅ Improvements:

1. **Works After Page Reload** - Snapshots persist, so What-If Agent can use them even after session state clears
2. **Multiple Fallback Options** - Checks multiple sources before giving up
3. **Consistent with Other Components** - Uses same snapshot loading pattern as forecast results tab
4. **No Breaking Changes** - Still works with session state when available

### 📊 Use Cases Now Supported:

1. **Run Forecast → Use What-If Agent** (session state)
   - Fast, no database access needed

2. **Reload Page → Use What-If Agent** (snapshot)
   - Works even after page reload
   - Uses latest saved snapshot

3. **Load Old Snapshot → Use What-If Agent** (future enhancement)
   - Framework ready for loading specific snapshots

---

## Testing

### Test Cases:

1. ✅ **Run forecast, then use What-If Agent**
   - Should work immediately (session state)

2. ✅ **Save snapshot, reload page, use What-If Agent**
   - Should load from snapshot
   - Should show "✅ Baseline forecast loaded"

3. ✅ **No forecast or snapshot**
   - Should show "No Baseline Forecast Found" message
   - Should provide button to go to Forecast section

---

## Files Modified

1. `app_refactored.py`
   - Updated `render_whatif_section()` function
   - Added snapshot loading logic
   - Added fallback chain for baseline forecast lookup

---

## Status

✅ **FIXED** - What-If Agent now correctly detects and uses snapshot valuations as baseline forecasts.

The What-If Agent will now:
- ✅ Check session state first (fastest)
- ✅ Fall back to latest snapshot (persistent)
- ✅ Show appropriate message if no baseline found
- ✅ Work after page reloads
