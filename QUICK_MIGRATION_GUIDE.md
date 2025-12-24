# Quick Migration Guide

## Problem
`psql` command not found - but that's okay! Use the Python migration runner instead.

## Solution (3 Options)

### ✅ Option 1: Python Script (Easiest - Recommended)

```bash
cd "/Users/gregmorris/Development Projects/CE Africa/ce-africa-valuation"

# Set your database password (get it from Supabase Dashboard → Settings → Database)
export SUPABASE_DB_PASSWORD='your-password-here'

# Run the migration
python3 run_migration.py migrations_add_detailed_line_items.sql
```

**To find your database password:**
1. Go to https://supabase.com/dashboard
2. Select your project
3. Go to **Settings** → **Database**
4. Look for **Database Password** or **Connection String**
5. Copy the password (or extract it from the connection string)

### ✅ Option 2: Supabase Dashboard (No Command Line)

1. Go to https://supabase.com/dashboard
2. Select your project: **qxbngbmpstwebjkbpdcj**
3. Click **SQL Editor** in the left sidebar
4. Click **New Query**
5. Open `migrations_add_detailed_line_items.sql` in your editor
6. Copy the entire file contents
7. Paste into the SQL Editor
8. Click **Run** (or press Cmd+Enter)
9. You should see "Success. No rows returned"

### Option 3: Install psql (If you prefer)

```bash
# macOS
brew install postgresql

# Then use the original script
./run_migration_psql.sh migrations_add_detailed_line_items.sql
```

## Verification

After running the migration, verify the tables exist:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name LIKE 'historical_%_line_items';
```

You should see:
- `historical_income_statement_line_items`
- `historical_balance_sheet_line_items`
- `historical_cashflow_line_items`

## Next Steps

1. Refresh your Streamlit app
2. Try importing your income statement line items again
3. The import should now work! ✅
