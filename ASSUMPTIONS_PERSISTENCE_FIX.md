# Assumptions Persistence Fix
**Date:** December 16, 2025  
**Status:** ✅ **FIXED**

---

## Problem

When users saved assumptions and then exited the application, the save did not persist. Assumptions were lost after application restart, even though the save operation appeared to succeed.

---

## Root Cause

The `save_assumptions_to_db` function was **not including `user_id`** in the database save operation. This caused two critical issues:

1. **RLS (Row Level Security) Violation:** Supabase RLS policies require `user_id` to match `auth.uid()` for data access. Without `user_id`:
   - Data might not be saved properly
   - Data might not be accessible when loading
   - RLS policies would block access to the saved data

2. **Missing User Context:** The database table `ai_assumptions` requires `user_id` for proper data isolation and security.

**Original Code:**
```python
def save_assumptions_to_db(db, assumptions_set: AssumptionsSet) -> bool:
    data = {
        'scenario_id': assumptions_set.scenario_id,
        'data': assumptions_set.to_dict(),
        'updated_at': assumptions_set.updated_at
        # ❌ Missing 'user_id' - Required for RLS!
    }
    
    db.client.table('ai_assumptions').upsert(
        data, on_conflict='scenario_id'  # ❌ Wrong conflict key
    ).execute()
```

---

## Solution

### 1. Updated `save_assumptions_to_db` Function

**File:** `components/ai_assumptions_engine.py` (lines ~768-804)

**Changes:**
- Added `user_id` parameter
- Included `user_id` in data payload
- Updated conflict key to include `user_id`
- Added fallback to `update_assumptions` method
- Improved error handling

**Updated Code:**
```python
def save_assumptions_to_db(db, assumptions_set: AssumptionsSet, user_id: str = None) -> bool:
    """Save assumptions to database.
    
    Args:
        db: Database handler
        assumptions_set: AssumptionsSet to save
        user_id: User ID for RLS compliance (required for persistence)
    """
    # Get user_id if not provided
    if not user_id:
        try:
            from supabase_utils import get_user_id
            user_id = get_user_id()
        except:
            st.error("User ID required for saving assumptions")
            return False
    
    data = {
        'scenario_id': assumptions_set.scenario_id,
        'user_id': user_id,  # ✅ CRITICAL: Required for RLS
        'data': assumptions_set.to_dict(),
        'updated_at': assumptions_set.updated_at
    }
    
    # Try upsert with user_id for RLS
    db.client.table('ai_assumptions').upsert(
        data, on_conflict='scenario_id,user_id'  # ✅ Correct conflict key
    ).execute()
```

### 2. Updated `load_assumptions_from_db` Function

**File:** `components/ai_assumptions_engine.py` (lines ~747-765)

**Changes:**
- Added `user_id` parameter
- Added `user_id` filter to database query for RLS compliance
- Added fallback to `get_scenario_assumptions` method

**Updated Code:**
```python
def load_assumptions_from_db(db, scenario_id: str, user_id: str = None) -> Optional[AssumptionsSet]:
    """Load saved assumptions from database.
    
    Args:
        db: Database handler
        scenario_id: Scenario ID
        user_id: User ID for RLS compliance (required for loading)
    """
    # Get user_id if not provided
    if not user_id:
        try:
            from supabase_utils import get_user_id
            user_id = get_user_id()
        except:
            pass
    
    # Add user_id filter for RLS compliance
    query = db.client.table('ai_assumptions').select('*').eq(
        'scenario_id', scenario_id
    )
    if user_id:
        query = query.eq('user_id', user_id)  # ✅ RLS compliance
    
    response = query.execute()
    # ... rest of loading logic
```

### 3. Updated `render_save_apply_tab` Function

**File:** `components/ai_assumptions_engine.py` (lines ~1651-1710)

**Changes:**
- Added `user_id` parameter
- Pass `user_id` to `save_assumptions_to_db` call

**Updated Code:**
```python
def render_save_apply_tab(db, scenario_id: str, assumptions_set: AssumptionsSet, user_id: str = None):
    # Get user_id if not provided
    if not user_id:
        from supabase_utils import get_user_id
        user_id = get_user_id()
    
    # ... UI code ...
    
    if st.button("💾 Save Assumptions", ...):
        success = save_assumptions_to_db(db, assumptions_set, user_id)  # ✅ Pass user_id
```

### 4. Updated `get_saved_assumptions` Function

**File:** `components/ai_assumptions_engine.py` (lines ~1838-1850)

**Changes:**
- Added `user_id` parameter
- Pass `user_id` to `load_assumptions_from_db` call

**Updated Code:**
```python
def get_saved_assumptions(db, scenario_id: str, user_id: str = None) -> Optional[AssumptionsSet]:
    """Get saved assumptions for use by other modules."""
    # Get user_id if not provided
    if not user_id:
        from supabase_utils import get_user_id
        user_id = get_user_id()
    
    # Then try database with user_id
    return load_assumptions_from_db(db, scenario_id, user_id)  # ✅ Pass user_id
```

### 5. Updated Function Calls

**File:** `components/ai_assumptions_engine.py`

**Changes:**
- Updated `render_save_apply_tab` call to pass `user_id` (line ~1232)
- Updated `load_assumptions_from_db` call to pass `user_id` (line ~1232)

---

## How It Works Now

### Save Flow:

1. **User clicks "Save Assumptions"**
   - `render_save_apply_tab` gets `user_id` (from parameter or `get_user_id()`)
   - Calls `save_assumptions_to_db(db, assumptions_set, user_id)`

2. **Save Function:**
   - Validates `user_id` is present
   - Creates data payload with `scenario_id`, `user_id`, `data`, `updated_at`
   - Upserts to `ai_assumptions` table with conflict key `scenario_id,user_id`
   - Returns success/failure

3. **Database:**
   - RLS policies allow save because `user_id` matches `auth.uid()`
   - Data is properly isolated per user
   - Data persists across application restarts

### Load Flow:

1. **Application starts or scenario selected**
   - `get_saved_assumptions` or `load_assumptions_from_db` is called
   - Gets `user_id` (from parameter or `get_user_id()`)

2. **Load Function:**
   - Queries `ai_assumptions` table with `scenario_id` AND `user_id` filters
   - RLS policies allow access because `user_id` matches `auth.uid()`
   - Returns loaded `AssumptionsSet` or `None`

3. **Result:**
   - Assumptions are loaded from database
   - Available in session state for use
   - Persists across page reloads

---

## Benefits

### ✅ Improvements:

1. **Proper RLS Compliance** - Data is saved and loaded with correct `user_id` context
2. **Data Persistence** - Assumptions now persist across application restarts
3. **Security** - Data is properly isolated per user
4. **Reliability** - Save operations are more reliable with proper error handling
5. **Backward Compatible** - Functions auto-detect `user_id` if not provided

### 📊 Use Cases Now Supported:

1. **Save → Exit → Restart** ✅
   - Assumptions are saved with `user_id`
   - Assumptions are loaded on restart
   - Data persists correctly

2. **Save → Reload Page** ✅
   - Assumptions are saved to database
   - Assumptions are loaded from database
   - Session state is restored

3. **Multi-User Scenarios** ✅
   - Each user's assumptions are isolated
   - RLS policies enforce proper access
   - No data leakage between users

---

## Testing

### Test Cases:

1. ✅ **Save assumptions, exit app, restart**
   - Save should succeed
   - Assumptions should be loaded on restart
   - All assumption values should match

2. ✅ **Save assumptions, reload page**
   - Assumptions should be loaded from database
   - Session state should be restored
   - UI should show saved status

3. ✅ **Save assumptions, switch scenarios, return**
   - Assumptions should be saved per scenario
   - Correct assumptions should load for each scenario
   - No cross-contamination

---

## Files Modified

1. `components/ai_assumptions_engine.py`
   - `save_assumptions_to_db()` - Added `user_id` parameter and inclusion in save
   - `load_assumptions_from_db()` - Added `user_id` parameter and filter
   - `render_save_apply_tab()` - Added `user_id` parameter and passing to save
   - `get_saved_assumptions()` - Added `user_id` parameter and passing to load
   - Updated all function calls to pass `user_id`

---

## Database Schema Requirements

The `ai_assumptions` table should have:
- `scenario_id` (UUID, foreign key to scenarios)
- `user_id` (UUID, foreign key to users, required for RLS)
- `data` (JSONB, contains assumptions data)
- `updated_at` (timestamp)
- Unique constraint on `(scenario_id, user_id)` for upsert conflict resolution

---

## Status

✅ **FIXED** - Assumptions now persist correctly across application restarts.

The save operation now:
- ✅ Includes `user_id` for RLS compliance
- ✅ Uses correct conflict key for upsert
- ✅ Properly saves to database
- ✅ Can be loaded after application restart
- ✅ Maintains data isolation per user
