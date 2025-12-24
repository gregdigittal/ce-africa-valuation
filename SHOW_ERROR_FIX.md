# show_error Import Fix
**Date:** December 17, 2025  
**Status:** ✅ **FIXED**

---

## Problem

When trying to save assumptions, users got:
```
NameError: name 'show_error' is not defined
```

This occurred in `components/ai_assumptions_engine.py` at line 920 in the `save_assumptions_to_db` function.

---

## Root Cause

The import of `show_error` from `components.ui_components` was inside a try/except block, but if the import failed silently or there was a circular import issue during Streamlit's module loading, the fallback functions might not have been properly assigned to the module namespace.

---

## Solution

**File:** `components/ai_assumptions_engine.py` (lines ~42-124)

**Changes:**
1. **Defined fallback functions first** - Before attempting import
2. **Assigned fallback functions to module namespace** - Ensures they're always available
3. **Enhanced exception handling** - Catches `ImportError`, `AttributeError`, `ModuleNotFoundError`, `NameError`
4. **Verified functions are callable** - Added check to ensure imported functions work

**Updated Code:**
```python
# Define fallback functions first to ensure they're always available
def _fallback_show_success(msg, details=None, icon="✅"):
    """Fallback success message function."""
    full_msg = f"{icon} **{msg}**"
    if details:
        full_msg += f"\n\n{details}"
    st.success(full_msg)

def _fallback_show_error(msg, details=None, icon="❌"):
    """Fallback error message function."""
    full_msg = f"{icon} **{msg}**"
    if details:
        full_msg += f"\n\n{details}"
    st.error(full_msg)

# ... (other fallback functions)

# Try to import from ui_components, use fallbacks if import fails
try:
    from components.ui_components import (
        show_success, show_error, show_warning, show_info, show_loading, show_progress
    )
    # Verify functions are actually callable
    if not all(callable(f) for f in [show_success, show_error, show_warning, show_info, show_loading, show_progress]):
        raise AttributeError("Imported functions are not callable")
except (ImportError, AttributeError, ModuleNotFoundError, NameError) as e:
    # Use fallback functions if import fails
    show_success = _fallback_show_success
    show_error = _fallback_show_error
    show_warning = _fallback_show_warning
    show_info = _fallback_show_info
    show_loading = _fallback_show_loading
    show_progress = _fallback_show_progress
except Exception:
    # Ultimate fallback - use fallback functions
    show_success = _fallback_show_success
    show_error = _fallback_show_error
    show_warning = _fallback_show_warning
    show_info = _fallback_show_info
    show_loading = _fallback_show_loading
    show_progress = _fallback_show_progress
```

---

## How It Works Now

1. **Fallback functions defined first** - Always available regardless of import status
2. **Import attempt** - Tries to import from `ui_components`
3. **Verification** - Checks that imported functions are callable
4. **Fallback assignment** - If import fails, assigns fallback functions to module namespace
5. **Functions always available** - `show_error`, `show_success`, etc. are always defined

---

## Verification

✅ All functions verified as available and callable:
- `show_success`: available and callable
- `show_error`: available and callable
- `show_warning`: available and callable
- `show_info`: available and callable
- `show_loading`: available and callable
- `show_progress`: available and callable

---

## Status

✅ **FIXED** - The `show_error` function (and all other standardized UI functions) are now guaranteed to be available in the module namespace, whether imported from `ui_components` or using fallback functions.

The error should no longer occur when saving assumptions.
