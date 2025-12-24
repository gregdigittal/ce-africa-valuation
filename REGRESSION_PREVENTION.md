# Regression Prevention Guide
**Last Updated:** December 15, 2025

## Overview

This document describes the regression prevention system implemented to ensure functionality doesn't regress when making changes.

---

## 🛡️ Protection Mechanisms

### 1. Pre-commit Hooks (`.pre-commit-config.yaml`)

**Purpose:** Automatically verify components before each commit.

**Installation:**
```bash
pip install pre-commit
pre-commit install
```

**What it checks:**
- ✅ All required component files exist
- ✅ All components are importable
- ✅ All required functions are available
- ✅ Python syntax is valid
- ✅ Code formatting (Black, isort)
- ✅ No merge conflicts
- ✅ No debug statements

**Usage:**
- Runs automatically on `git commit`
- Run manually: `pre-commit run --all-files`

### 2. Component Verification Script (`scripts/verify_components.py`)

**Purpose:** Standalone script to verify all components.

**Usage:**
```bash
python3 scripts/verify_components.py
```

**What it checks:**
- All required root files exist
- All required components exist
- All components are importable
- All required functions are available

**Exit codes:**
- `0` = All checks passed
- `1` = Issues found (prevents commit)

### 3. Component Inventory Tracker (`scripts/component_inventory.py`)

**Purpose:** Track component availability over time.

**Usage:**
```bash
python3 scripts/component_inventory.py
```

**Output:**
- Human-readable report
- JSON inventory (`.component_inventory.json`)
- Component sizes and modification dates

**Use cases:**
- Before/after comparison
- Tracking component changes
- Identifying missing components

### 4. Backup Before Changes (`scripts/backup_before_changes.sh`)

**Purpose:** Create automatic backups before making changes.

**Usage:**
```bash
./scripts/backup_before_changes.sh
```

**What it backs up:**
- All root Python files
- All component files
- Creates timestamped backup in `.backups/`

**Restore:**
```bash
cp .backups/pre_change_TIMESTAMP/*.py .
cp .backups/pre_change_TIMESTAMP/components/*.py components/
```

---

## 📋 Required Components Checklist

### Root Files (Must Exist)
- ✅ `app_refactored.py` - Main application
- ✅ `db_connector.py` - Database handler
- ✅ `supabase_utils.py` - Supabase utilities
- ✅ `funding_engine.py` - Funding calculations
- ✅ `linear_theme.py` - Theme system

### Components (Must Exist)
- ✅ `components/command_center.py`
- ✅ `components/setup_wizard.py`
- ✅ `components/forecast_section.py`
- ✅ `components/scenario_comparison.py`
- ✅ `components/ai_assumptions_engine.py`
- ✅ `components/vertical_integration.py`
- ✅ `components/ai_trend_analysis.py`
- ✅ `components/funding_ui.py`
- ✅ `components/user_management.py`
- ✅ `components/ui_components.py`
- ✅ `components/financial_statements.py`
- ✅ `components/enhanced_navigation.py`
- ✅ `components/ai_assumptions_integration.py`

### Optional Components (Graceful Fallbacks)
- ⚠️ `components/whatif_agent.py` - Referenced but optional
- ⚠️ `components/workflow_navigator.py` - UI enhancement
- ⚠️ `components/workflow_guidance.py` - UI enhancement

---

## 🔄 Workflow for Making Changes

### Before Making Changes

1. **Create Backup:**
   ```bash
   ./scripts/backup_before_changes.sh
   ```

2. **Verify Current State:**
   ```bash
   python3 scripts/verify_components.py
   python3 scripts/component_inventory.py
   ```

3. **Note Current Inventory:**
   - Check `.component_inventory.json`
   - Document what you're changing

### During Development

1. **Test Frequently:**
   ```bash
   python3 scripts/verify_components.py
   ```

2. **Check Imports:**
   ```bash
   python3 -c "from components.YOUR_COMPONENT import YOUR_FUNCTION"
   ```

3. **Run Syntax Check:**
   ```bash
   python3 -m py_compile your_file.py
   ```

### Before Committing

1. **Run Pre-commit Hooks:**
   ```bash
   pre-commit run --all-files
   ```

2. **Verify Components:**
   ```bash
   python3 scripts/verify_components.py
   ```

3. **Update Inventory:**
   ```bash
   python3 scripts/component_inventory.py
   ```

4. **Commit:**
   ```bash
   git add .
   git commit -m "Your message"
   # Pre-commit hooks run automatically
   ```

---

## 🚨 If Regression Detected

### Immediate Actions

1. **Stop and Assess:**
   - Don't commit broken code
   - Identify what broke

2. **Check Verification:**
   ```bash
   python3 scripts/verify_components.py
   ```

3. **Restore from Backup:**
   ```bash
   # Find latest backup
   ls -lt .backups/
   
   # Restore
   cp .backups/pre_change_TIMESTAMP/*.py .
   cp .backups/pre_change_TIMESTAMP/components/*.py components/
   ```

4. **Verify Restoration:**
   ```bash
   python3 scripts/verify_components.py
   ```

### Investigation

1. **Compare Inventories:**
   ```bash
   diff .component_inventory.json .backups/pre_change_TIMESTAMP/.component_inventory.json
   ```

2. **Check Git History:**
   ```bash
   git log --oneline -10
   git diff HEAD~1
   ```

3. **Identify Missing Files:**
   ```bash
   python3 scripts/component_inventory.py
   # Compare with previous inventory
   ```

---

## 📊 Monitoring Component Health

### Daily Checks

Run component inventory to track changes:
```bash
python3 scripts/component_inventory.py
```

### Weekly Reviews

1. Review `.component_inventory.json` history
2. Check for missing components
3. Verify all imports still work
4. Update this document if needed

### Before Major Changes

1. Full backup
2. Component inventory snapshot
3. Verification script run
4. Document expected changes

---

## 🔧 Maintenance

### Updating Required Components List

Edit `scripts/verify_components.py`:
```python
REQUIRED_COMPONENTS = {
    'new_component': ['required_function'],
    # ... existing components
}
```

### Adding New Checks

1. Add to `verify_components.py`
2. Update `.pre-commit-config.yaml` if needed
3. Test the new check
4. Update this document

---

## ✅ Best Practices

1. **Always Backup Before Changes**
   - Use `backup_before_changes.sh`
   - Or create manual backup

2. **Verify Before Committing**
   - Run verification script
   - Fix issues before committing

3. **Test Imports**
   - Test component imports after changes
   - Verify functions are accessible

4. **Track Changes**
   - Update component inventory
   - Document what changed

5. **Use Pre-commit Hooks**
   - Install and use pre-commit
   - Let it catch issues early

6. **Regular Inventory Checks**
   - Run inventory script regularly
   - Compare over time

---

## 📝 Quick Reference

```bash
# Verify components
python3 scripts/verify_components.py

# Generate inventory
python3 scripts/component_inventory.py

# Create backup
./scripts/backup_before_changes.sh

# Run pre-commit hooks
pre-commit run --all-files

# Test component import
python3 -c "from components.COMPONENT import FUNCTION"

# Check syntax
python3 -m py_compile file.py
```

---

## 🎯 Success Criteria

A successful change should:
- ✅ Pass all verification checks
- ✅ All components importable
- ✅ All functions available
- ✅ No syntax errors
- ✅ Backup created
- ✅ Inventory updated

If any check fails, **DO NOT COMMIT** until fixed.
