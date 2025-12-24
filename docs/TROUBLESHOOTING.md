# Troubleshooting Guide

**Version:** 2.0  
**Last Updated:** December 17, 2025

---

## Common Issues and Solutions

### Application Won't Start

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
1. Activate virtual environment: `source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. If `requirements.txt` missing, install manually:
   ```bash
   pip install streamlit pandas numpy plotly scipy supabase python-dateutil
   ```

---

### Database Connection Issues

**Error**: `Error connecting to Supabase`

**Solutions**:
1. Check `secrets.toml` exists in `.streamlit/` directory
2. Verify Supabase URL and key are correct
3. Check Supabase project is active
4. Verify network connectivity
5. Check RLS policies allow your user

**Error**: `Table 'X' does not exist`

**Solution**:
1. Run database migrations
2. Check table names in Supabase dashboard
3. Verify schema matches expected structure

---

### Forecast Issues

**Error**: `No assumptions configured`

**Solution**:
1. Complete Setup step first
2. Ensure assumptions are saved
3. Check `assumptions` table has data for scenario

**Error**: `No machines found in fleet`

**Solution**:
1. Import machine data in Setup
2. Check machines are marked as "Active"
3. Verify `machine_instances` or `installed_base` table has data
4. Check `scenario_id` matches

**Error**: `Forecast results not persisting`

**Solution**:
1. Results auto-save to snapshots
2. Check Forecast Snapshots tab
3. Ensure you're using the same scenario
4. Verify database write permissions

---

### AI Assumptions Issues

**Error**: `Analysis results not saving`

**Solution**:
1. Analysis auto-saves when complete
2. Explicit save is for final assumptions
3. Check database connection
4. Verify `user_id` is set correctly

**Error**: `JSON serialization error`

**Solution**:
- Fixed in Sprint 19
- Ensure you're using latest code
- Check for numpy types in data

---

### What-If Agent Issues

**Error**: `No baseline forecast found`

**Solution**:
1. Run a forecast first
2. Check forecast results are in session state
3. Verify snapshots exist in database
4. Try reloading the page

**Error**: `TypeError in calculations`

**Solution**:
1. Check for None values in adjustments
2. Verify baseline forecast structure
3. Ensure all required fields present

---

### Manufacturing Strategy Issues

**Error**: `Manufacturing strategy not set`

**Solution**:
1. Save manufacturing strategy explicitly
2. Check database for saved strategy
3. Verify `manufacturing_strategy_saved` flag
4. Try refreshing the page

**Error**: `Circular dependency`

**Solution**:
- Fixed in Sprint 19
- Manufacturing is now optional
- Can be set before or after forecast

---

### Workflow Issues

**Error**: `Prerequisites not met` (but they are)

**Solution**:
1. Check workflow progress in database
2. Verify stage completion flags
3. Try marking stage complete manually
4. Clear session state cache

**Error**: `Workflow stage not accessible`

**Solution**:
1. Complete prerequisite stages first
2. Check workflow definition in code
3. Verify database workflow_progress table

---

### Session State Issues

**Error**: `Session state lost on reload`

**Solution**:
1. This is expected - use database persistence
2. Forecast results load from snapshots
3. AI assumptions auto-save to database
4. Use SessionManager for state management

**Error**: `Duplicate element keys`

**Solution**:
1. Ensure unique keys for Streamlit elements
2. Use scenario_id in keys where needed
3. Check for duplicate button/form keys

---

### Performance Issues

**Slow forecast calculation**:
1. Reduce forecast duration
2. Reduce number of machines
3. Check for inefficient loops
4. Use vectorized operations

**Memory issues**:
1. Clear session state cache
2. Reduce data size
3. Process in batches

---

## Debugging Steps

### 1. Check Logs

Look for error messages in:
- Streamlit console output
- Browser developer console
- Database logs (Supabase dashboard)

### 2. Verify State

Use SessionManager to inspect state:

```python
from services.session_manager import SessionManager
summary = SessionManager.get_state_summary()
print(summary)
```

### 3. Test Components

Run component verification:

```bash
python scripts/verify_components.py
```

### 4. Check Database

Verify data in Supabase:
- Check tables have data
- Verify RLS policies
- Check user_id matches

### 5. Clear Cache

Clear session state cache:

```python
from services.session_manager import SessionManager
SessionManager.clear_cache()
```

---

## Getting Help

1. Check this troubleshooting guide
2. Review component-specific help (ℹ️ icons)
3. Check [Developer Guide](DEVELOPER_GUIDE.md)
4. Review error messages carefully
5. Check database state in Supabase dashboard

---

## Known Issues

### NumPy 2.0 Compatibility

**Issue**: `np.float_` removed in NumPy 2.0

**Status**: Fixed in Sprint 19
- Use `np.floating` instead
- Code updated for compatibility

### JSON Serialization

**Issue**: Numpy types not JSON serializable

**Status**: Fixed in Sprint 19
- `convert_to_serializable()` function added
- All to_dict() methods updated

---

## Reporting Issues

When reporting issues, include:
1. Error message (full traceback)
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details (Python version, OS)
5. Relevant code/logs
