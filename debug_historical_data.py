#!/usr/bin/env python3
"""
Debug script to check historical line items data for duplicate issues.
Run this to diagnose the "cannot assemble with duplicate keys" error.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from db_connector import SupabaseHandler
import streamlit as st

# Initialize
st.set_page_config(page_title="Debug Historical Data", layout="wide")
st.title("🔍 Debug Historical Data")

# Get scenario ID from user
scenario_id = st.text_input("Scenario ID", value="67ce009a-97fd-4404-bb3e-8d61d0d7c5fb")
user_id = "00000000-0000-0000-0000-000000000000"  # Nil UUID

if st.button("Check Data"):
    db = SupabaseHandler()
    
    # Load detailed line items
    st.header("📊 Detailed Line Items Data")
    
    try:
        response = db.client.table('historical_income_statement_line_items').select('*').eq(
            'scenario_id', scenario_id
        ).eq('user_id', user_id).order('period_date').execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            st.success(f"✅ Loaded {len(df)} rows")
            
            # Check for duplicates
            st.subheader("🔍 Duplicate Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                duplicate_periods = df[df['period_date'].duplicated(keep=False)]
                st.metric("Duplicate Period Dates", len(duplicate_periods))
                if len(duplicate_periods) > 0:
                    st.dataframe(duplicate_periods[['period_date', 'line_item_name', 'amount']].head(10))
            
            with col2:
                duplicate_line_items = df[df.duplicated(subset=['period_date', 'line_item_name'], keep=False)]
                st.metric("Duplicate Period + Line Item", len(duplicate_line_items))
                if len(duplicate_line_items) > 0:
                    st.dataframe(duplicate_line_items[['period_date', 'line_item_name', 'amount']].head(10))
            
            # Show unique periods
            st.subheader("📅 Unique Periods")
            unique_periods = sorted(df['period_date'].unique())
            st.write(f"Found {len(unique_periods)} unique periods:")
            st.write(unique_periods)
            
            # Try aggregation
            st.subheader("🧪 Test Aggregation")
            try:
                from components.ai_assumptions_engine import aggregate_detailed_line_items_to_summary
                aggregated = aggregate_detailed_line_items_to_summary(db, scenario_id, user_id)
                if not aggregated.empty:
                    st.success(f"✅ Aggregation successful! {len(aggregated)} periods")
                    st.dataframe(aggregated)
                else:
                    st.warning("⚠️ Aggregation returned empty DataFrame")
            except Exception as e:
                st.error(f"❌ Aggregation failed: {str(e)}")
                import traceback
                with st.expander("Full Traceback"):
                    st.code(traceback.format_exc())
            
            # Show sample data
            st.subheader("📋 Sample Data (First 20 rows)")
            st.dataframe(df.head(20))
            
        else:
            st.warning("No data found")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        with st.expander("Full Traceback"):
            st.code(traceback.format_exc())
