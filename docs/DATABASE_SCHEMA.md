# Database Schema Documentation

**Version:** 2.0  
**Last Updated:** December 17, 2025

---

## Overview

The platform uses Supabase (PostgreSQL) with Row Level Security (RLS) enabled.

**Key Principle**: All tables require `user_id` matching `auth.uid()` for RLS compliance.

---

## Core Tables

### `scenarios`

Stores scenario metadata.

**Columns**:
- `id` (UUID, PK)
- `user_id` (UUID, FK to auth.users)
- `name` (TEXT)
- `status` (TEXT) - 'draft', 'active', 'archived'
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

**RLS**: Users can only access their own scenarios

---

### `assumptions`

Stores scenario assumptions as JSONB.

**Columns**:
- `id` (UUID, PK)
- `scenario_id` (UUID, FK to scenarios)
- `user_id` (UUID, FK to auth.users)
- `data` (JSONB) - Contains:
  - Manual assumptions
  - `ai_assumptions` (nested) - AI-derived assumptions
  - `manufacturing_strategy` (nested) - Manufacturing config
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

**RLS**: Users can only access their own assumptions

**JSONB Structure**:
```json
{
  "wacc": 0.12,
  "inflation_rate": 5,
  "gross_margin_liner": 38,
  "gross_margin_refurb": 32,
  "forecast_duration_months": 60,
  "ai_assumptions": {
    // AI assumptions data
  },
  "manufacturing_strategy": {
    // Manufacturing config
  }
}
```

---

### `machine_instances`

Stores machine fleet data (Sprint 2+ schema).

**Columns**:
- `id` (UUID, PK)
- `scenario_id` (UUID, FK to scenarios)
- `user_id` (UUID, FK to auth.users)
- `site_id` (UUID, FK to sites)
- `machine_id` (TEXT)
- `status` (TEXT) - 'Active', 'Inactive'
- `commission_date` (DATE)
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

**Relationships**:
- `sites` → `customers` (via foreign keys)

**RLS**: Users can only access their own machines

---

### `installed_base`

Legacy table for machine data (fallback).

**Columns**:
- `id` (UUID, PK)
- `scenario_id` (UUID)
- `user_id` (UUID)
- `machine_id` (TEXT)
- `customer_name` (TEXT)
- `site_name` (TEXT)
- `machine_model` (TEXT)
- `commission_date` (DATE)
- `status` (TEXT)

**Note**: Still used as fallback for older scenarios

---

### `historic_financials`

Stores monthly P&L historical data.

**Columns**:
- `id` (UUID, PK)
- `scenario_id` (UUID, FK to scenarios)
- `user_id` (UUID, FK to auth.users)
- `month` (DATE)
- `revenue` (NUMERIC)
- `cogs` (NUMERIC)
- `gross_profit` (NUMERIC)
- `opex` (NUMERIC)
- `ebit` (NUMERIC)
- `created_at` (TIMESTAMP)

**RLS**: Users can only access their own data

---

### `forecast_snapshots`

Stores saved forecast results.

**Columns**:
- `id` (UUID, PK)
- `scenario_id` (UUID, FK to scenarios)
- `user_id` (UUID, FK to auth.users)
- `snapshot_name` (TEXT)
- `forecast_data` (JSONB) - Full forecast results
- `summary_stats` (JSONB) - Summary statistics
- `is_locked` (BOOLEAN)
- `created_at` (TIMESTAMP)

**RLS**: Users can only access their own snapshots

---

### `workflow_progress`

Tracks workflow stage completion.

**Columns**:
- `id` (UUID, PK)
- `scenario_id` (UUID, FK to scenarios)
- `user_id` (UUID, FK to auth.users)
- `stage` (TEXT) - 'setup', 'ai_analysis', 'forecast', etc.
- `completed` (BOOLEAN)
- `completed_at` (TIMESTAMP)

**Unique Constraint**: `(scenario_id, stage)`

**RLS**: Users can only access their own progress

---

### `prospects`

Stores pipeline/prospect data.

**Columns**:
- `id` (UUID, PK)
- `scenario_id` (UUID, FK to scenarios)
- `user_id` (UUID, FK to auth.users)
- `expected_close_date` (DATE)
- `confidence_pct` (NUMERIC) - 0-100
- `annual_liner_value` (NUMERIC)
- `refurb_value` (NUMERIC)
- `created_at` (TIMESTAMP)

**RLS**: Users can only access their own prospects

---

### `expense_assumptions`

Stores operating expense assumptions.

**Columns**:
- `id` (UUID, PK)
- `scenario_id` (UUID, FK to scenarios)
- `user_id` (UUID, FK to auth.users)
- `category` (TEXT)
- `function_type` (TEXT) - 'fixed', 'variable', 'power', 'step', 'linked', 'budget'
- `fixed_monthly` (NUMERIC)
- `variable_rate` (NUMERIC)
- `escalation_rate` (NUMERIC)
- `is_active` (BOOLEAN)
- `created_at` (TIMESTAMP)

**RLS**: Users can only access their own expenses

---

## Supporting Tables

### `customers`

Customer master data.

**Columns**:
- `id` (UUID, PK)
- `user_id` (UUID, FK to auth.users)
- `customer_name` (TEXT)
- `created_at` (TIMESTAMP)

### `sites`

Site master data.

**Columns**:
- `id` (UUID, PK)
- `user_id` (UUID, FK to auth.users)
- `customer_id` (UUID, FK to customers)
- `site_name` (TEXT)
- `ore_type_id` (UUID, FK to ore_types)
- `created_at` (TIMESTAMP)

### `wear_profiles`

Wear profile definitions.

**Columns**:
- `id` (UUID, PK)
- `user_id` (UUID, FK to auth.users)
- `machine_model` (TEXT)
- `liner_life_months` (INTEGER)
- `refurb_interval_months` (INTEGER)
- `avg_consumable_revenue` (NUMERIC)
- `avg_refurb_revenue` (NUMERIC)
- `gross_margin_liner` (NUMERIC)
- `gross_margin_refurb` (NUMERIC)

---

## Row Level Security (RLS)

**Critical**: All tables have RLS enabled.

**Policy Pattern**:
```sql
CREATE POLICY "Users can only access their own data"
ON table_name
FOR ALL
USING (user_id = auth.uid());
```

**Important**: Always include `user_id` in queries and inserts.

---

## Indexes

Recommended indexes:

```sql
-- Scenarios
CREATE INDEX idx_scenarios_user_id ON scenarios(user_id);

-- Assumptions
CREATE INDEX idx_assumptions_scenario_user ON assumptions(scenario_id, user_id);

-- Machine Instances
CREATE INDEX idx_machines_scenario ON machine_instances(scenario_id);
CREATE INDEX idx_machines_user ON machine_instances(user_id);

-- Forecast Snapshots
CREATE INDEX idx_snapshots_scenario_user ON forecast_snapshots(scenario_id, user_id);
CREATE INDEX idx_snapshots_created ON forecast_snapshots(created_at DESC);

-- Workflow Progress
CREATE INDEX idx_workflow_scenario_stage ON workflow_progress(scenario_id, stage);
```

---

## Data Relationships

```
scenarios (1) ──→ (many) assumptions
scenarios (1) ──→ (many) machine_instances
scenarios (1) ──→ (many) historic_financials
scenarios (1) ──→ (many) forecast_snapshots
scenarios (1) ──→ (many) prospects
scenarios (1) ──→ (many) expense_assumptions
scenarios (1) ──→ (many) workflow_progress

customers (1) ──→ (many) sites
sites (1) ──→ (many) machine_instances
```

---

## Migration Notes

### From Legacy Schema

The platform supports both:
- **New Schema**: `machine_instances` with `sites` → `customers` hierarchy
- **Legacy Schema**: `installed_base` table

Code automatically falls back to legacy schema if new schema is empty.

---

## Best Practices

1. **Always include user_id**: Required for RLS
2. **Use JSONB for flexible data**: Assumptions, forecast data
3. **Index foreign keys**: Improve query performance
4. **Validate data types**: Ensure numeric fields are numeric
5. **Use transactions**: For multi-table operations
6. **Handle NULLs**: Check for None/null values

---

## Common Queries

### Get Scenario with Assumptions

```sql
SELECT s.*, a.data as assumptions
FROM scenarios s
LEFT JOIN assumptions a ON s.id = a.scenario_id
WHERE s.user_id = $1 AND s.id = $2;
```

### Get Workflow Progress

```sql
SELECT stage, completed, completed_at
FROM workflow_progress
WHERE scenario_id = $1 AND user_id = $2;
```

### Get Latest Snapshot

```sql
SELECT *
FROM forecast_snapshots
WHERE scenario_id = $1 AND user_id = $2
ORDER BY created_at DESC
LIMIT 1;
```
