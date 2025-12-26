-- Migration: Add Support for Detailed Forecast Line Items
-- Date: 2025-12-19
-- Purpose: Store detailed line item forecasts matching historical granularity

-- Create table for detailed income statement forecast line items
CREATE TABLE IF NOT EXISTS forecast_income_statement_line_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scenario_id UUID NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    -- NOTE: No FK to auth.users to allow dev mode (nil UUID) and service-role inserts.
    user_id UUID NOT NULL,
    snapshot_id UUID REFERENCES forecast_snapshots(id) ON DELETE CASCADE,
    period_date DATE NOT NULL,
    line_item_name TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_category TEXT,
    amount NUMERIC(15, 2) NOT NULL,
    forecast_method TEXT, -- 'trend_fit', 'correlation', 'fixed', 'percentage'
    forecast_source TEXT, -- Source element if correlated (e.g., 'revenue')
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(scenario_id, snapshot_id, period_date, line_item_name)
);

-- Create table for detailed balance sheet forecast line items
CREATE TABLE IF NOT EXISTS forecast_balance_sheet_line_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scenario_id UUID NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    -- NOTE: No FK to auth.users to allow dev mode (nil UUID) and service-role inserts.
    user_id UUID NOT NULL,
    snapshot_id UUID REFERENCES forecast_snapshots(id) ON DELETE CASCADE,
    period_date DATE NOT NULL,
    line_item_name TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_category TEXT,
    amount NUMERIC(15, 2) NOT NULL,
    forecast_method TEXT,
    forecast_source TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(scenario_id, snapshot_id, period_date, line_item_name)
);

-- Create table for detailed cash flow forecast line items
CREATE TABLE IF NOT EXISTS forecast_cashflow_line_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scenario_id UUID NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    -- NOTE: No FK to auth.users to allow dev mode (nil UUID) and service-role inserts.
    user_id UUID NOT NULL,
    snapshot_id UUID REFERENCES forecast_snapshots(id) ON DELETE CASCADE,
    period_date DATE NOT NULL,
    line_item_name TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_category TEXT,
    amount NUMERIC(15, 2) NOT NULL,
    forecast_method TEXT,
    forecast_source TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(scenario_id, snapshot_id, period_date, line_item_name)
);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_forecast_is_line_items_scenario_snapshot ON forecast_income_statement_line_items(scenario_id, snapshot_id, period_date);
CREATE INDEX IF NOT EXISTS idx_forecast_is_line_items_category ON forecast_income_statement_line_items(scenario_id, category);
CREATE INDEX IF NOT EXISTS idx_forecast_bs_line_items_scenario_snapshot ON forecast_balance_sheet_line_items(scenario_id, snapshot_id, period_date);
CREATE INDEX IF NOT EXISTS idx_forecast_bs_line_items_category ON forecast_balance_sheet_line_items(scenario_id, category);
CREATE INDEX IF NOT EXISTS idx_forecast_cf_line_items_scenario_snapshot ON forecast_cashflow_line_items(scenario_id, snapshot_id, period_date);
CREATE INDEX IF NOT EXISTS idx_forecast_cf_line_items_category ON forecast_cashflow_line_items(scenario_id, category);

-- Enable RLS
ALTER TABLE forecast_income_statement_line_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecast_balance_sheet_line_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecast_cashflow_line_items ENABLE ROW LEVEL SECURITY;

-- RLS Policies for forecast_income_statement_line_items
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'forecast_income_statement_line_items' AND policyname = 'Users can manage their own income statement forecast line items') THEN
        CREATE POLICY "Users can manage their own income statement forecast line items"
            ON forecast_income_statement_line_items
            FOR ALL
            USING (auth.uid() = user_id)
            WITH CHECK (auth.uid() = user_id);
    END IF;
END $$;

-- RLS Policies for forecast_balance_sheet_line_items
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'forecast_balance_sheet_line_items' AND policyname = 'Users can manage their own balance sheet forecast line items') THEN
        CREATE POLICY "Users can manage their own balance sheet forecast line items"
            ON forecast_balance_sheet_line_items
            FOR ALL
            USING (auth.uid() = user_id)
            WITH CHECK (auth.uid() = user_id);
    END IF;
END $$;

-- RLS Policies for forecast_cashflow_line_items
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'forecast_cashflow_line_items' AND policyname = 'Users can manage their own cashflow forecast line items') THEN
        CREATE POLICY "Users can manage their own cashflow forecast line items"
            ON forecast_cashflow_line_items
            FOR ALL
            USING (auth.uid() = user_id)
            WITH CHECK (auth.uid() = user_id);
    END IF;
END $$;

-- Add comments
COMMENT ON TABLE forecast_income_statement_line_items IS 'Stores detailed line item forecasts for income statements (e.g., Accounting Fees, Advertising, Depreciation)';
COMMENT ON TABLE forecast_balance_sheet_line_items IS 'Stores detailed line item forecasts for balance sheets (e.g., Property Plant and Equipment, Right-of-Use Assets, Trade Receivables)';
COMMENT ON TABLE forecast_cashflow_line_items IS 'Stores detailed line item forecasts for cash flow statements (e.g., Depreciation and Amortisation, Increase in Inventories, Net Cash From Operating Activities)';

COMMENT ON COLUMN forecast_income_statement_line_items.line_item_name IS 'Name of the line item (e.g., "Accounting Fees", "Depreciation", "Revenue")';
COMMENT ON COLUMN forecast_income_statement_line_items.category IS 'Category from historical data (e.g., "Operating Expenses", "Revenue", "Other Income")';
COMMENT ON COLUMN forecast_income_statement_line_items.forecast_method IS 'Method used to forecast: trend_fit, correlation, fixed, percentage';
COMMENT ON COLUMN forecast_income_statement_line_items.forecast_source IS 'Source element if correlated (e.g., "revenue" if forecasted as % of revenue)';
