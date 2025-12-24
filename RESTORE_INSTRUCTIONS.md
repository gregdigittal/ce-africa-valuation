# Quick Restore Instructions

## Restore to Last Known Working Model

To restore the most recent backup, simply run:

```bash
bash scripts/restore_last_working.sh
```

Or from the project root:

```bash
./scripts/restore_last_working.sh
```

This will:
1. ✅ Automatically find the most recent backup
2. ✅ Ask for confirmation before restoring
3. ✅ Restore all code files
4. ✅ Restore components, services, migrations, tests
5. ✅ Preserve your current secrets.toml (not overwritten)

## What Gets Restored

- ✅ All Python source files
- ✅ Components directory
- ✅ Services directory
- ✅ Migration SQL files
- ✅ Tests directory
- ✅ Scripts (except restore script itself)

## What Doesn't Get Restored

- ❌ `secrets.toml` (for security - you keep your current secrets)
- ❌ `.git/` directory (Git history preserved)
- ❌ Any new files you created after backup

## Example Usage

```bash
# Navigate to project root
cd "/Users/gregmorris/Development Projects/CE Africa/ce-africa-valuation"

# Restore to last working model
bash scripts/restore_last_working.sh

# The script will:
# 1. Find the most recent backup
# 2. Show you which backup it found
# 3. Ask for confirmation
# 4. Restore all files
```

## After Restore

1. **Test the application:**
   ```bash
   streamlit run app_refactored.py
   ```

2. **Check for errors** in the console

3. **Verify features** work as expected

## Finding Available Backups

To see all available backups:

```bash
ls -la .backups/
```

The most recent backup will be restored automatically.

---

**Simple Command:** `bash scripts/restore_last_working.sh`
