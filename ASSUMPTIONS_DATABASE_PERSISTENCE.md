# Assumptions Database Persistence - Confirmation & Enhancement

## Current Status

**Assumptions ARE saved to the database** (Supabase `assumptions` table), not just in browser session state.

### How It Works

1. **Database Storage:**
   - All assumptions are saved to Supabase `assumptions` table
   - Stored as JSONB in the `data` column
   - Includes: `forecast_configs`, `ai_assumptions`, manual assumptions, etc.

2. **Session State Caching:**
   - Used for performance (avoids redundant DB queries)
   - Cache is invalidated when assumptions are saved
   - On app restart, cache is empty, so loads from database

3. **Persistence:**
   - ✅ Assumptions persist across app restarts
   - ✅ Saved to database immediately when user saves
   - ✅ Loaded from database on app startup

## Enhancements Made

### 1. Improved Cache Invalidation

**File:** `components/forecast_config_ui.py` - Save button

**Changes:**
- Invalidates cache after saving to database
- Ensures fresh load on next access
- Clear success message indicates database save

**Before:**
```python
if db.update_assumptions(...):
    st.success("✅ Forecast configuration saved!")
    st.rerun()
```

**After:**
```python
if db.update_assumptions(...):
    st.success("✅ Forecast configuration saved to database!")
    
    # Invalidate cache to ensure fresh load
    cache_key = f'assumptions_{scenario_id}'
    if cache_key in st.session_state:
        del st.session_state[cache_key]
    
    st.rerun()
```

### 2. Enhanced Load Function

**File:** `components/forecast_section.py` - `load_assumptions()`

**Changes:**
- Always loads from database first (not just from cache)
- Cache is only used if database load fails
- Added `force_refresh` parameter for explicit refresh
- Clear documentation that database is source of truth

**Key Points:**
- On app restart: Cache is empty → Loads from database ✅
- During session: Uses cache for performance (but DB is source of truth)
- After save: Cache invalidated → Next load gets fresh data from DB ✅

### 3. AI Assumptions Cache Invalidation

**File:** `components/ai_assumptions_engine.py` - `save_assumptions_to_db()`

**Changes:**
- Invalidates both assumptions cache and AI assumptions cache
- Ensures all caches are cleared after save
- Guarantees fresh data on next load

## Database Schema

**Table:** `assumptions`
- `scenario_id`: UUID (foreign key to scenarios)
- `user_id`: UUID (for RLS)
- `data`: JSONB (contains all assumptions)

**Data Structure:**
```json
{
  "forecast_configs": {
    "revenue": {
      "method": "trend_fit",
      "trend_function_type": "linear",
      "trend_parameters": {"slope": 100, "intercept": 1000}
    },
    ...
  },
  "ai_assumptions": {
    "assumptions": {...},
    "analysis_complete": true,
    "assumptions_saved": true
  },
  "manual_assumptions": {...}
}
```

## How It Works

### Saving Assumptions

1. **User saves configuration:**
   - Clicks "Save Configuration" button
   - `update_assumptions()` called
   - Data saved to Supabase `assumptions` table
   - Cache invalidated
   - Success message shown

2. **Database Operation:**
   ```python
   # Check if record exists
   existing = db.client.table("assumptions").select("id").eq("scenario_id", scenario_id).execute()
   
   if existing.data:
       # Update existing
       db.client.table("assumptions").update({"data": assumptions}).eq("id", assump_id).execute()
   else:
       # Insert new
       db.client.table("assumptions").insert(payload).execute()
   ```

### Loading Assumptions

1. **On App Startup:**
   - Cache is empty (new session)
   - `load_assumptions()` called
   - Loads from database
   - Caches in session_state for performance

2. **During Session:**
   - Uses cache if available (performance)
   - But database is always source of truth
   - Cache invalidated after saves

3. **After Save:**
   - Cache cleared
   - Next load gets fresh data from database

## Verification

To verify assumptions are in database:

1. **Save assumptions:**
   - Configure and save any assumptions
   - Check success message: "✅ ... saved to database!"

2. **Restart app:**
   - Close and reopen the app
   - Select the same scenario
   - Assumptions should load automatically

3. **Check database:**
   - Query Supabase `assumptions` table
   - Should see `data` JSONB with your assumptions

## Benefits

1. **Persistence:** Assumptions survive app restarts ✅
2. **Performance:** Caching reduces DB queries during session
3. **Reliability:** Database is source of truth, cache is optimization
4. **Consistency:** Cache invalidated after saves ensures fresh data

## Summary

- ✅ **Assumptions ARE saved to database** (not just browser)
- ✅ **Persist across app restarts**
- ✅ **Cache is performance optimization only**
- ✅ **Database is source of truth**
- ✅ **Cache invalidated after saves**

The system was already saving to database, but now:
- Cache invalidation is explicit
- Load function prioritizes database
- Clear messaging that data is saved to database
- Ensures fresh data after saves

---

**Date:** December 20, 2025  
**Status:** ✅ Confirmed & Enhanced
