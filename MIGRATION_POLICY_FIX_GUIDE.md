# Fix for "Policy Already Exists" Errors

## Problem
When running migrations, you may encounter:
```
ERROR: 42710: policy "Anyone can read jurisdictions" for table "gf_jurisdictions" already exists
```

## Solution
PostgreSQL doesn't support `CREATE POLICY IF NOT EXISTS`, so you need to drop the policy first if it exists.

## Pattern to Use

In any migration file that creates policies, use this pattern:

```sql
-- Drop policy if it exists
DROP POLICY IF EXISTS "policy_name" ON table_name;

-- Then create the policy
CREATE POLICY "policy_name"
ON table_name
FOR SELECT  -- or INSERT, UPDATE, DELETE
USING (condition);
```

## Example Fix

For the `gf_jurisdictions` table:

```sql
-- Drop the policy if it exists
DROP POLICY IF EXISTS "Anyone can read jurisdictions" ON gf_jurisdictions;

-- Recreate the policy
CREATE POLICY "Anyone can read jurisdictions"
ON gf_jurisdictions
FOR SELECT
USING (true);
```

## Running the Fix

1. **Using Python script:**
   ```bash
   python run_migration.py migrations_fix_jurisdictions_policy.sql
   ```

2. **Using psql:**
   ```bash
   psql $DATABASE_URL -f migrations_fix_jurisdictions_policy.sql
   ```

3. **Via Supabase Dashboard:**
   - Go to SQL Editor
   - Paste the SQL from the migration file
   - Run it

## For Other Policies

If you encounter similar errors for other policies, apply the same pattern:
1. Find the migration file creating the policy
2. Add `DROP POLICY IF EXISTS` before `CREATE POLICY`
3. Re-run the migration
