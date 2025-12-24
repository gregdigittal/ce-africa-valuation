-- Fix for "policy already exists" error on gf_jurisdictions table
-- This migration safely handles the policy creation by dropping it first if it exists

-- Drop the policy if it exists, then recreate it
DROP POLICY IF EXISTS "Anyone can read jurisdictions" ON gf_jurisdictions;

-- Recreate the policy
CREATE POLICY "Anyone can read jurisdictions"
ON gf_jurisdictions
FOR SELECT
USING (true);
