-- Migration: Add depreciation, interest_expense, tax, and related columns to historic_financials
-- Date: 2025-12-19
-- Purpose: Support trial balance extraction of depreciation, interest, and tax data

-- Add missing columns to historic_financials table
ALTER TABLE historic_financials
ADD COLUMN IF NOT EXISTS depreciation NUMERIC DEFAULT 0,
ADD COLUMN IF NOT EXISTS interest_expense NUMERIC DEFAULT 0,
ADD COLUMN IF NOT EXISTS tax NUMERIC DEFAULT 0,
ADD COLUMN IF NOT EXISTS other_income NUMERIC DEFAULT 0,
ADD COLUMN IF NOT EXISTS other_expense NUMERIC DEFAULT 0,
ADD COLUMN IF NOT EXISTS net_profit NUMERIC DEFAULT 0;

-- Add comments for documentation
COMMENT ON COLUMN historic_financials.depreciation IS 'Depreciation and amortization expense';
COMMENT ON COLUMN historic_financials.interest_expense IS 'Interest expense (may be extracted from other_expense if not separately available)';
COMMENT ON COLUMN historic_financials.tax IS 'Income tax expense';
COMMENT ON COLUMN historic_financials.other_income IS 'Other operating income';
COMMENT ON COLUMN historic_financials.other_expense IS 'Other operating expenses (may include interest if not separately tracked)';
COMMENT ON COLUMN historic_financials.net_profit IS 'Net profit after tax';

-- Update existing records to have default values (if any exist)
UPDATE historic_financials
SET 
    depreciation = COALESCE(depreciation, 0),
    interest_expense = COALESCE(interest_expense, 0),
    tax = COALESCE(tax, 0),
    other_income = COALESCE(other_income, 0),
    other_expense = COALESCE(other_expense, 0),
    net_profit = COALESCE(net_profit, 0)
WHERE 
    depreciation IS NULL 
    OR interest_expense IS NULL 
    OR tax IS NULL 
    OR other_income IS NULL 
    OR other_expense IS NULL 
    OR net_profit IS NULL;
