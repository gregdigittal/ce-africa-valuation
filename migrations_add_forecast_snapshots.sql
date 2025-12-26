-- Migration: Add Support for Forecast Snapshots
-- Date: 2025-12-26
-- Purpose: Persist forecast results (JSONB) per scenario/user.

CREATE TABLE IF NOT EXISTS forecast_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    scenario_id UUID NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    snapshot_name TEXT NOT NULL,
    snapshot_date DATE DEFAULT CURRENT_DATE,
    snapshot_type TEXT DEFAULT 'base',
    assumptions_data JSONB,
    forecast_data JSONB,
    prospects_data JSONB,
    valuation_data JSONB,
    monte_carlo_data JSONB,
    total_revenue_forecast NUMERIC,
    total_gross_profit_forecast NUMERIC,
    enterprise_value NUMERIC,
    summary_stats JSONB,
    notes TEXT,
    is_locked BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for common access patterns
CREATE INDEX IF NOT EXISTS idx_forecast_snapshots_scenario_user ON forecast_snapshots(scenario_id, user_id);
CREATE INDEX IF NOT EXISTS idx_forecast_snapshots_created ON forecast_snapshots(created_at DESC);

-- Enable RLS
ALTER TABLE forecast_snapshots ENABLE ROW LEVEL SECURITY;

-- RLS policy
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE tablename = 'forecast_snapshots'
          AND policyname = 'Users can manage their own forecast snapshots'
    ) THEN
        CREATE POLICY "Users can manage their own forecast snapshots"
            ON forecast_snapshots
            FOR ALL
            USING (auth.uid() = user_id)
            WITH CHECK (auth.uid() = user_id);
    END IF;
END $$;

COMMENT ON TABLE forecast_snapshots IS 'Stores saved forecast results (JSONB) per scenario/user.';
