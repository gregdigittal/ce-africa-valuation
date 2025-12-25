-- Migration: Align Online RLS + assumptions.user_id (Recommended)
-- Date: 2025-12-25
-- Purpose:
--   Bring the database closer to the app's intended online/RLS-ready schema, notably:
--   - add assumptions.user_id (backfilled from scenarios.user_id)
--   - enforce one assumptions row per scenario (dedupe + unique index)
--   - enable RLS + policies for key app tables
--
-- IMPORTANT:
--   - Run this in Supabase SQL Editor (or via psql as an admin/service role).
--   - Enabling RLS will restrict anon/auth access unless policies exist (this script creates them).
--   - Service Role bypasses RLS; anon/auth users will be constrained by these policies.

-- =============================================================================
-- 0) ASSUMPTIONS: add user_id + backfill + dedupe + indexes
-- =============================================================================

ALTER TABLE IF EXISTS public.assumptions
  ADD COLUMN IF NOT EXISTS user_id uuid;

-- Backfill assumptions.user_id from scenarios.user_id
UPDATE public.assumptions a
SET user_id = s.user_id
FROM public.scenarios s
WHERE a.scenario_id = s.id
  AND a.user_id IS NULL;

-- Optional safety trigger: keep assumptions.user_id consistent with scenarios.user_id
-- (helps if a client forgets to send user_id on insert).
CREATE OR REPLACE FUNCTION public._set_assumptions_user_id_from_scenario()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  IF NEW.user_id IS NULL THEN
    SELECT s.user_id INTO NEW.user_id
    FROM public.scenarios s
    WHERE s.id = NEW.scenario_id;
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_set_assumptions_user_id ON public.assumptions;
CREATE TRIGGER trg_set_assumptions_user_id
BEFORE INSERT OR UPDATE OF scenario_id, user_id
ON public.assumptions
FOR EACH ROW
EXECUTE FUNCTION public._set_assumptions_user_id_from_scenario();

-- Dedupe assumptions: keep the newest row per scenario_id (by created_at, then id)
WITH ranked AS (
  SELECT
    id,
    scenario_id,
    ROW_NUMBER() OVER (
      PARTITION BY scenario_id
      ORDER BY created_at DESC NULLS LAST, id DESC
    ) AS rn
  FROM public.assumptions
)
DELETE FROM public.assumptions a
USING ranked r
WHERE a.id = r.id
  AND r.rn > 1;

-- Enforce one assumptions row per scenario_id
CREATE UNIQUE INDEX IF NOT EXISTS ux_assumptions_scenario_id
  ON public.assumptions (scenario_id);

-- Helpful index for RLS + joins
CREATE INDEX IF NOT EXISTS idx_assumptions_user_id
  ON public.assumptions (user_id);

-- Only set NOT NULL once backfilled
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM public.assumptions WHERE user_id IS NULL) THEN
    RAISE NOTICE 'assumptions.user_id still has NULLs after backfill; leaving column nullable. Fix rows then rerun SET NOT NULL.';
  ELSE
    ALTER TABLE public.assumptions ALTER COLUMN user_id SET NOT NULL;
  END IF;
END $$;

-- =============================================================================
-- 1) SCENARIOS + ASSUMPTIONS: enable RLS + policies
-- =============================================================================

ALTER TABLE public.scenarios ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.assumptions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can manage their own scenarios" ON public.scenarios;
CREATE POLICY "Users can manage their own scenarios"
  ON public.scenarios
  FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can manage their own assumptions" ON public.assumptions;
CREATE POLICY "Users can manage their own assumptions"
  ON public.assumptions
  FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- =============================================================================
-- 2) APP TABLES WITH user_id: enable RLS + policies
-- =============================================================================

-- These tables store user_id directly (uuid). Apply standard "manage own rows" policy.
DO $$
DECLARE
  t text;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'customers',
    'sites',
    'machine_instances',
    'wear_profiles',
    'installed_base',
    'prospects',
    'forecast_snapshots',
    'workflow_progress',
    'aged_debtors',
    'aged_creditors',
    'sales_orders',
    'granular_sales_history',
    'historical_income_statement_line_items',
    'historical_balance_sheet_line_items',
    'historical_cashflow_line_items'
  ]
  LOOP
    IF to_regclass('public.' || t) IS NOT NULL THEN
      EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY;', t);
      EXECUTE format('DROP POLICY IF EXISTS %I ON public.%I;', 'Users can manage their own ' || t, t);
      EXECUTE format(
        'CREATE POLICY %I ON public.%I FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);',
        'Users can manage their own ' || t, t
      );
    END IF;
  END LOOP;
END $$;

-- =============================================================================
-- 3) APP TABLES WITHOUT user_id but with scenario_id: enable RLS via scenarios join
-- =============================================================================

-- These tables are scenario-scoped. Enforce access via scenarios.user_id.
DO $$
DECLARE
  t text;
  pol text;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'historic_financials',
    'historic_customer_revenue',
    'historic_expense_categories',
    'historical_balance_sheet',
    'historical_cashflow',
    'historical_trial_balance',
    'expense_assumptions'
  ]
  LOOP
    IF to_regclass('public.' || t) IS NOT NULL THEN
      EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY;', t);
      pol := 'Users can manage their scenario ' || t;
      -- Policy names are identifiers (not string literals). Use %I to quote safely (supports spaces).
      EXECUTE format('DROP POLICY IF EXISTS %I ON public.%I;', pol, t);
      EXECUTE format(
        'CREATE POLICY %I ON public.%I FOR ALL USING (EXISTS (SELECT 1 FROM public.scenarios s WHERE s.id = %I.scenario_id AND s.user_id = auth.uid())) WITH CHECK (EXISTS (SELECT 1 FROM public.scenarios s WHERE s.id = %I.scenario_id AND s.user_id = auth.uid()));',
        pol, t, t, t
      );
    END IF;
  END LOOP;
END $$;

-- =============================================================================
-- 4) CUSTOMER CODE UNIQUENESS (Recommended for multi-tenant online)
-- =============================================================================
-- Your current schema shows customers.customer_code as globally UNIQUE.
-- For online multi-user use, it should be unique per user: UNIQUE(user_id, customer_code).
--
-- This block is safe if constraint exists; if not, it will be skipped.
DO $$
BEGIN
  IF to_regclass('public.customers') IS NOT NULL THEN
    -- Drop the old global unique constraint if it exists (Supabase default name: customers_customer_code_key)
    IF EXISTS (
      SELECT 1
      FROM information_schema.table_constraints
      WHERE table_schema = 'public'
        AND table_name = 'customers'
        AND constraint_type = 'UNIQUE'
        AND constraint_name = 'customers_customer_code_key'
    ) THEN
      ALTER TABLE public.customers DROP CONSTRAINT customers_customer_code_key;
    END IF;

    -- Create per-user unique index
    CREATE UNIQUE INDEX IF NOT EXISTS ux_customers_user_customer_code
      ON public.customers (user_id, customer_code);
  END IF;
END $$;

-- =============================================================================
-- 5) NOTES
-- =============================================================================
-- If you have additional tables with user_id stored as TEXT (not uuid),
-- prefer migrating those columns to uuid for consistency, or use policies like:
--   USING (auth.uid()::text = user_id)
-- for those specific tables.

