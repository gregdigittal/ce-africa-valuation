-- Migration: Add Sales Orders (Historical) Import Support
-- Date: 2025-12-25
-- Purpose:
--   Store granular sales order lines to:
--   1) derive monthly seasonality patterns (used for upsampling annual/YTD statements)
--   2) infer Wear Parts vs Refurbishment/Service revenue split for historical Income Statement
--
-- Notes:
-- - user_id does NOT have a foreign key constraint to allow nil UUID in development
-- - RLS policies ensure users can only access their own data
--
CREATE TABLE IF NOT EXISTS sales_orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scenario_id UUID NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    order_number TEXT,
    order_date DATE NOT NULL,
    due_date DATE,
    customer_code TEXT,
    customer_name TEXT,
    item_code TEXT,
    description TEXT,
    quantity NUMERIC(15, 4),
    unit_price NUMERIC(15, 4),
    discount_pct NUMERIC(10, 4),
    total_amount NUMERIC(15, 2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sales_orders_scenario_date ON sales_orders(scenario_id, order_date);
CREATE INDEX IF NOT EXISTS idx_sales_orders_user_date ON sales_orders(user_id, order_date);

-- Enable RLS
ALTER TABLE sales_orders ENABLE ROW LEVEL SECURITY;

-- RLS Policy
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE tablename = 'sales_orders'
          AND policyname = 'Users can manage their own sales orders'
    ) THEN
        CREATE POLICY "Users can manage their own sales orders"
            ON sales_orders
            FOR ALL
            USING (auth.uid() = user_id)
            WITH CHECK (auth.uid() = user_id);
    END IF;
END $$;

COMMENT ON TABLE sales_orders IS 'Stores granular sales order lines (historical) used for seasonality + revenue split (wear vs refurb/service).';
COMMENT ON COLUMN sales_orders.total_amount IS 'Line total amount for the order line (positive for sales).';

