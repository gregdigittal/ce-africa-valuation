# Migration Instructions: Detailed Line Items Tables

## Problem
When trying to import detailed line items, you may see this error:
```
Could not find the table 'public.historical_income_statement_line_items' in the schema cache
```

This means the database tables for detailed line items haven't been created yet.

## Solution: Run the Migration

### Option 1: Python Migration Runner (Recommended - No psql Required)

**Step 1:** Install psycopg2 if not already installed:
```bash
pip install psycopg2-binary
```

**Step 2:** Set your database password:
```bash
export SUPABASE_DB_PASSWORD='your-database-password'
```

**Step 3:** Run the migration:
```bash
cd "/Users/gregmorris/Development Projects/CE Africa/ce-africa-valuation"
python3 run_migration.py migrations_add_detailed_line_items.sql
```

The script will:
- Automatically read your Supabase URL from `.streamlit/secrets.toml`
- Prompt for password if not set in environment
- Execute the migration using Python (no psql needed)

### Option 2: Using psql (If Installed)

**Step 1:** Set Database URL:
```bash
export DATABASE_URL='postgresql://postgres:YOUR_PASSWORD@db.qxbngbmpstwebjkbpdcj.supabase.co:5432/postgres'
```

**Step 2:** Run the migration:
```bash
cd "/Users/gregmorris/Development Projects/CE Africa/ce-africa-valuation"
./run_migration_psql.sh migrations_add_detailed_line_items.sql
```

### Option 3: Supabase Dashboard (Easiest - No Command Line)

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your project
3. Navigate to **SQL Editor**
4. Click **New Query**
5. Copy and paste the entire contents of `migrations_add_detailed_line_items.sql`
6. Click **Run** (or press Cmd/Ctrl + Enter)
7. Verify success message

## What This Migration Creates

The migration creates three new tables:

1. **`historical_income_statement_line_items`**
   - Stores individual income statement line items (e.g., Accounting Fees, Advertising, Depreciation)
   - Columns: `period_date`, `line_item_name`, `category`, `sub_category`, `amount`

2. **`historical_balance_sheet_line_items`**
   - Stores individual balance sheet line items (e.g., Property Plant and Equipment, Trade Receivables)
   - Columns: `period_date`, `line_item_name`, `category`, `sub_category`, `amount`

3. **`historical_cashflow_line_items`**
   - Stores individual cash flow line items (e.g., Depreciation and Amortisation, Increase in Inventories)
   - Columns: `period_date`, `line_item_name`, `category`, `sub_category`, `amount`

All tables include:
- Row Level Security (RLS) policies
- Indexes for performance
- Foreign key constraints to `scenarios` and `auth.users`

## Verification

After running the migration, you can verify the tables exist:

```sql
-- Check if tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name LIKE 'historical_%_line_items';
```

You should see:
- `historical_income_statement_line_items`
- `historical_balance_sheet_line_items`
- `historical_cashflow_line_items`

## Troubleshooting

### Error: "DATABASE_URL environment variable not set"
- Make sure you've exported the DATABASE_URL before running the script
- Or use the direct psql command with the full connection string

### Error: "permission denied"
- Make sure the script is executable: `chmod +x run_migration_psql.sh`
- Or run with: `bash run_migration_psql.sh migrations_add_detailed_line_items.sql`

### Error: "connection refused" or "authentication failed"
- Verify your database password is correct
- Check that the database host and port are accessible
- Ensure your IP is whitelisted in Supabase (if applicable)

### Error: "relation already exists"
- The tables already exist, which is fine
- The migration uses `CREATE TABLE IF NOT EXISTS`, so it's safe to run multiple times

## Next Steps

After running the migration:
1. Refresh your Streamlit app
2. Navigate to **Setup → Historics → Detailed Line Items**
3. Upload your AFS Extract files
4. The import should now work correctly!
