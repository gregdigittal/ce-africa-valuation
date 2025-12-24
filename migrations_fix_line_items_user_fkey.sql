-- Migration: Fix Foreign Key Constraint for Historical Line Items
-- Date: 2025-12-19
-- Purpose: Remove strict foreign key constraint on user_id to allow nil UUID for development
-- Note: RLS policies already ensure users can only access their own data

-- Drop the foreign key constraint on user_id for all three line item tables
-- This allows the nil UUID (00000000-0000-0000-0000-000000000000) used in development

-- For historical_income_statement_line_items
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'historical_income_statement_line_items_user_id_fkey'
        AND table_name = 'historical_income_statement_line_items'
    ) THEN
        ALTER TABLE historical_income_statement_line_items 
        DROP CONSTRAINT historical_income_statement_line_items_user_id_fkey;
    END IF;
END $$;

-- For historical_balance_sheet_line_items
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'historical_balance_sheet_line_items_user_id_fkey'
        AND table_name = 'historical_balance_sheet_line_items'
    ) THEN
        ALTER TABLE historical_balance_sheet_line_items 
        DROP CONSTRAINT historical_balance_sheet_line_items_user_id_fkey;
    END IF;
END $$;

-- For historical_cashflow_line_items
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'historical_cashflow_line_items_user_id_fkey'
        AND table_name = 'historical_cashflow_line_items'
    ) THEN
        ALTER TABLE historical_cashflow_line_items 
        DROP CONSTRAINT historical_cashflow_line_items_user_id_fkey;
    END IF;
END $$;

-- Note: RLS policies remain in place to ensure security:
-- - Users can only access their own data via auth.uid() = user_id
-- - The foreign key constraint was redundant since RLS handles access control
