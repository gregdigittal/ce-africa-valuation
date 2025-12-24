-- Migration: Add Support for Detailed Financial Statement Line Items
-- Date: 2025-12-19
-- Purpose: Store all individual line items from financial statements (not just summaries)

-- Create table for detailed income statement line items
-- Note: user_id does NOT have a foreign key constraint to allow nil UUID in development
-- RLS policies ensure users can only access their own data
CREATE TABLE IF NOT EXISTS historical_income_statement_line_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scenario_id UUID NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    period_date DATE NOT NULL,
    line_item_name TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_category TEXT,
    amount NUMERIC(15, 2) NOT NULL,
    statement_type TEXT NOT NULL DEFAULT 'income_statement', -- 'income_statement', 'balance_sheet', 'cash_flow'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(scenario_id, period_date, line_item_name, statement_type)
);

-- Create table for detailed balance sheet line items
-- Note: user_id does NOT have a foreign key constraint to allow nil UUID in development
-- RLS policies ensure users can only access their own data
CREATE TABLE IF NOT EXISTS historical_balance_sheet_line_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scenario_id UUID NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    period_date DATE NOT NULL,
    line_item_name TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_category TEXT,
    amount NUMERIC(15, 2) NOT NULL,
    statement_type TEXT NOT NULL DEFAULT 'balance_sheet',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(scenario_id, period_date, line_item_name, statement_type)
);

-- Create table for detailed cash flow line items
-- Note: user_id does NOT have a foreign key constraint to allow nil UUID in development
-- RLS policies ensure users can only access their own data
CREATE TABLE IF NOT EXISTS historical_cashflow_line_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scenario_id UUID NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    period_date DATE NOT NULL,
    line_item_name TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_category TEXT,
    amount NUMERIC(15, 2) NOT NULL,
    statement_type TEXT NOT NULL DEFAULT 'cash_flow',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(scenario_id, period_date, line_item_name, statement_type)
);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_hist_is_line_items_scenario_period ON historical_income_statement_line_items(scenario_id, period_date);
CREATE INDEX IF NOT EXISTS idx_hist_is_line_items_category ON historical_income_statement_line_items(scenario_id, category);
CREATE INDEX IF NOT EXISTS idx_hist_bs_line_items_scenario_period ON historical_balance_sheet_line_items(scenario_id, period_date);
CREATE INDEX IF NOT EXISTS idx_hist_bs_line_items_category ON historical_balance_sheet_line_items(scenario_id, category);
CREATE INDEX IF NOT EXISTS idx_hist_cf_line_items_scenario_period ON historical_cashflow_line_items(scenario_id, period_date);
CREATE INDEX IF NOT EXISTS idx_hist_cf_line_items_category ON historical_cashflow_line_items(scenario_id, category);

-- Enable RLS
ALTER TABLE historical_income_statement_line_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE historical_balance_sheet_line_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE historical_cashflow_line_items ENABLE ROW LEVEL SECURITY;

-- RLS Policies for historical_income_statement_line_items
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'historical_income_statement_line_items' AND policyname = 'Users can manage their own income statement line items') THEN
        CREATE POLICY "Users can manage their own income statement line items"
            ON historical_income_statement_line_items
            FOR ALL
            USING (auth.uid() = user_id)
            WITH CHECK (auth.uid() = user_id);
    END IF;
END $$;

-- RLS Policies for historical_balance_sheet_line_items
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'historical_balance_sheet_line_items' AND policyname = 'Users can manage their own balance sheet line items') THEN
        CREATE POLICY "Users can manage their own balance sheet line items"
            ON historical_balance_sheet_line_items
            FOR ALL
            USING (auth.uid() = user_id)
            WITH CHECK (auth.uid() = user_id);
    END IF;
END $$;

-- RLS Policies for historical_cashflow_line_items
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'historical_cashflow_line_items' AND policyname = 'Users can manage their own cashflow line items') THEN
        CREATE POLICY "Users can manage their own cashflow line items"
            ON historical_cashflow_line_items
            FOR ALL
            USING (auth.uid() = user_id)
            WITH CHECK (auth.uid() = user_id);
    END IF;
END $$;

-- Add comments
COMMENT ON TABLE historical_income_statement_line_items IS 'Stores individual line items from income statements (e.g., Accounting Fees, Advertising, Depreciation)';
COMMENT ON TABLE historical_balance_sheet_line_items IS 'Stores individual line items from balance sheets (e.g., Property Plant and Equipment, Right-of-Use Assets, Trade Receivables)';
COMMENT ON TABLE historical_cashflow_line_items IS 'Stores individual line items from cash flow statements (e.g., Depreciation and Amortisation, Increase in Inventories, Net Cash From Operating Activities)';

COMMENT ON COLUMN historical_income_statement_line_items.line_item_name IS 'Name of the line item (e.g., "Accounting Fees", "Depreciation", "Revenue")';
COMMENT ON COLUMN historical_income_statement_line_items.category IS 'Category from source file (e.g., "Operating Expenses", "Revenue", "Other Income")';
COMMENT ON COLUMN historical_income_statement_line_items.sub_category IS 'Sub-category if available (e.g., "PPE", "Leases", "Intangibles" for balance sheet)';
COMMENT ON COLUMN historical_income_statement_line_items.amount IS 'Amount for this line item (negative for expenses, positive for income)';
