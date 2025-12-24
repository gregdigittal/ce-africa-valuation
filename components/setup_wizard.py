"""
Setup Wizard - Guided Configuration with Column Mapping
========================================================
Step-by-step setup flow with flexible CSV imports.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

# Database handler
try:
    from db_connector import SupabaseHandler
except ImportError:
    SupabaseHandler = None

# Column mapper
try:
    from components.column_mapper import (
        render_import_with_mapping,
        render_column_mapper,
        validate_mapping,
        apply_mapping,
        process_import,
        FIELD_CONFIGS,
        CSV_TEMPLATES,
    )
    COLUMN_MAPPER_AVAILABLE = True
except ImportError:
    try:
        from column_mapper import (
            render_import_with_mapping,
            render_column_mapper,
            validate_mapping,
            apply_mapping,
            process_import,
            FIELD_CONFIGS,
            CSV_TEMPLATES,
        )
        COLUMN_MAPPER_AVAILABLE = True
    except ImportError:
        COLUMN_MAPPER_AVAILABLE = False

# =============================================================================
# STEP DEFINITIONS
# =============================================================================

STEPS = [
    ("basics", "1️⃣ Basics", "Global assumptions and settings"),
    ("customers", "2️⃣ Customers", "Customer master data"),
    ("fleet", "3️⃣ Fleet", "Wear profiles and installed machines"),
    ("working_capital", "4️⃣ Working Capital", "Debtors, suppliers, creditors"),
    ("pipeline", "5️⃣ Pipeline", "Sales opportunities"),
    ("historics", "6️⃣ Historics", "Line-level IS/BS/CF"),
    ("costs", "7️⃣ Costs", "Expense configuration"),
]

# =============================================================================
# HELPERS
# =============================================================================


def load_table_data(db, table: str, user_id: str, scenario_id: str = None) -> pd.DataFrame:
    """Load data from a table."""
    try:
        query = db.client.table(table).select("*")

        user_only_tables = ["customers", "aged_debtors", "aged_creditors", "wear_profiles"]

        scenario_only_tables = [
            "historic_financials",
            "historic_customer_revenue",
            "historic_expense_categories",
            "historical_balance_sheet",
            "historical_cashflow",
            "historical_trial_balance",
            "historical_income_statement_line_items",
            "historical_balance_sheet_line_items",
            "historical_cashflow_line_items",
            "expense_assumptions",
        ]

        user_and_scenario_tables = ["installed_base", "creditors", "prospects"]

        if table in user_only_tables:
            query = query.eq("user_id", user_id)
        elif table in scenario_only_tables:
            if scenario_id:
                query = query.eq("scenario_id", scenario_id)
        elif table in user_and_scenario_tables:
            query = query.eq("user_id", user_id)
            if scenario_id:
                query = query.eq("scenario_id", scenario_id)

        result = query.execute()
        return pd.DataFrame(result.data) if result.data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def clear_table_data(db, table: str, user_id: str, scenario_id: str = None) -> bool:
    """Clear data from a table."""
    try:
        query = db.client.table(table).delete()

        user_only_tables = ["customers", "aged_debtors", "aged_creditors", "wear_profiles"]

        scenario_only_tables = [
            "historic_financials",
            "historic_customer_revenue",
            "historic_expense_categories",
            "historical_balance_sheet",
            "historical_cashflow",
            "historical_trial_balance",
            "historical_income_statement_line_items",
            "historical_balance_sheet_line_items",
            "historical_cashflow_line_items",
        ]

        user_and_scenario_tables = ["installed_base", "creditors", "prospects"]

        if table in user_only_tables:
            query = query.eq("user_id", user_id)
        elif table in scenario_only_tables:
            if scenario_id:
                query = query.eq("scenario_id", scenario_id)
            else:
                st.warning("Scenario ID required to clear this table")
                return False
        elif table in user_and_scenario_tables:
            query = query.eq("user_id", user_id)
            if scenario_id:
                query = query.eq("scenario_id", scenario_id)

        query.execute()
        return True
    except Exception as e:
        st.error(f"Failed to clear {table}: {e}")
        return False


# =============================================================================
# STEP COMPLETION CHECKS
# =============================================================================


def check_basics_complete(db, scenario_id: str, user_id: str) -> bool:
    try:
        result = db.client.table("assumptions").select("data").eq("scenario_id", scenario_id).execute()
        if result.data and result.data[0].get("data"):
            assumptions = result.data[0]["data"]
            return bool(assumptions.get("wacc"))
        return False
    except Exception:
        return False


def check_customers_complete(db, scenario_id: str, user_id: str) -> bool:
    try:
        df = load_table_data(db, "customers", user_id)
        return len(df) > 0
    except Exception:
        return False


def check_fleet_complete(db, scenario_id: str, user_id: str) -> bool:
    try:
        wp = load_table_data(db, "wear_profiles", user_id)
        ib = load_table_data(db, "installed_base", user_id, scenario_id)
        return len(wp) > 0 or len(ib) > 0
    except Exception:
        return False


def check_working_capital_complete(db, scenario_id: str, user_id: str) -> bool:
    try:
        deb = load_table_data(db, "aged_debtors", user_id)
        cred = load_table_data(db, "aged_creditors", user_id)
        return len(deb) > 0 or len(cred) > 0
    except Exception:
        return False


def check_pipeline_complete(db, scenario_id: str, user_id: str) -> bool:
    try:
        df = load_table_data(db, "prospects", user_id, scenario_id)
        return len(df) > 0
    except Exception:
        return True  # optional


def check_historics_complete(db, scenario_id: str, user_id: str) -> bool:
    try:
        is_li_df = load_table_data(db, "historical_income_statement_line_items", user_id, scenario_id)
        bs_li_df = load_table_data(db, "historical_balance_sheet_line_items", user_id, scenario_id)
        cf_li_df = load_table_data(db, "historical_cashflow_line_items", user_id, scenario_id)
        return len(is_li_df) > 0 or len(bs_li_df) > 0 or len(cf_li_df) > 0
    except Exception:
        return False


def check_costs_complete(db, scenario_id: str, user_id: str) -> bool:
    return True


STEP_CHECKS = {
    "basics": check_basics_complete,
    "customers": check_customers_complete,
    "fleet": check_fleet_complete,
    "working_capital": check_working_capital_complete,
    "pipeline": check_pipeline_complete,
    "historics": check_historics_complete,
    "costs": check_costs_complete,
}

# =============================================================================
# STEP RENDERERS
# =============================================================================


def render_step_basics(db, scenario_id: str, user_id: str):
    st.subheader("⚙️ Global Assumptions")
    st.caption("Configure core model parameters")

    assumptions = {}
    try:
        result = db.client.table("assumptions").select("data").eq("scenario_id", scenario_id).execute()
        if result.data and result.data[0].get("data"):
            assumptions = result.data[0]["data"]
    except Exception:
        assumptions = {}

    if not assumptions:
        assumptions = {}

    with st.form("basics_form"):
        col1, col2 = st.columns(2)
        with col1:
            wacc = st.number_input(
                "WACC / Discount Rate",
                value=float(assumptions.get("wacc", 0.12)),
                min_value=0.0,
                max_value=0.50,
                step=0.01,
                format="%.3f",
            )
            duration = st.number_input(
                "Forecast Duration (Months)",
                value=int(assumptions.get("forecast_duration_months", 60)),
                min_value=12,
                max_value=120,
                step=12,
            )
            inflation = st.number_input(
                "Inflation Rate",
                value=float(assumptions.get("inflation_rate", 0.05)),
                min_value=0.0,
                max_value=0.30,
                step=0.01,
                format="%.3f",
            )
        with col2:
            margin_consumable = st.number_input(
                "Consumable Gross Margin %",
                value=float(assumptions.get("margin_consumable_pct", 0.35)),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.3f",
            )
            margin_refurb = st.number_input(
                "Refurbishment Gross Margin %",
                value=float(assumptions.get("margin_refurb_pct", 0.28)),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.3f",
            )
            terminal_growth = st.number_input(
                "Terminal Growth Rate",
                value=float(assumptions.get("terminal_growth_rate", 0.03)),
                min_value=0.0,
                max_value=0.10,
                step=0.005,
                format="%.3f",
            )

        st.markdown("---")
        st.markdown("##### Monte Carlo Settings")
        col3, col4 = st.columns(2)
        with col3:
            mc_iterations = st.number_input(
                "Monte Carlo Iterations",
                value=int(assumptions.get("mc_iterations", 1000)),
                min_value=100,
                max_value=10000,
                step=100,
            )
        with col4:
            mc_seed = st.number_input(
                "Random Seed (0 = random)",
                value=int(assumptions.get("mc_seed", 0)),
                min_value=0,
                max_value=99999,
            )

        if st.form_submit_button("💾 Save Assumptions", type="primary", use_container_width=True):
            assumptions_data = {
                "wacc": wacc,
                "forecast_duration_months": duration,
                "inflation_rate": inflation,
                "margin_consumable_pct": margin_consumable,
                "margin_refurb_pct": margin_refurb,
                "terminal_growth_rate": terminal_growth,
                "mc_iterations": mc_iterations,
                "mc_seed": mc_seed,
            }
            try:
                existing = db.client.table("assumptions").select("id").eq("scenario_id", scenario_id).execute()
                if existing.data:
                    db.client.table("assumptions").update({"data": assumptions_data}).eq("scenario_id", scenario_id).execute()
                else:
                    db.client.table("assumptions").insert(
                        {"scenario_id": scenario_id, "data": assumptions_data}
                    ).execute()
                st.success("✅ Assumptions saved!")
                st.rerun()
            except Exception as e:
                st.error(f"Error saving assumptions: {e}")


def render_step_customers(db, scenario_id: str, user_id: str):
    st.subheader("👥 Customers")
    st.caption("Import and manage your customer base")

    df = load_table_data(db, "customers", user_id)
    if not df.empty:
        st.success(f"✅ {len(df)} customers loaded")
        with st.expander("📋 View Current Customers"):
            display_cols = ["customer_code", "customer_name", "is_active"]
            available = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available] if available else df.head(20), use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear All Customers", key="clear_customers"):
            if clear_table_data(db, "customers", user_id):
                st.success("Cleared!")
                st.rerun()
    else:
        st.info("No customers loaded yet")

    st.markdown("---")
    if COLUMN_MAPPER_AVAILABLE:
        render_import_with_mapping(db, user_id, "customers")
    else:
        st.warning("Column mapper not available. Install column_mapper.py in components folder.")
        _render_fallback_customer_form(db, user_id)


def _render_fallback_customer_form(db, user_id: str):
    with st.expander("➕ Quick Add Customer"):
        with st.form("quick_add_customer"):
            col1, col2 = st.columns(2)
            code = col1.text_input("Customer Code", placeholder="e.g., ANG001")
            name = col2.text_input("Customer Name", placeholder="e.g., AngloGold Ashanti")
            if st.form_submit_button("Add Customer", type="primary"):
                if code and name:
                    try:
                        db.client.table("customers").insert(
                            {"user_id": user_id, "customer_code": code, "customer_name": name, "is_active": True}
                        ).execute()
                        st.success(f"✅ Added {name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
                else:
                    st.error("Code and Name required")


def render_step_fleet(db, scenario_id: str, user_id: str):
    st.subheader("🏗️ Fleet & Wear Profiles")
    st.caption("Import wear profiles and installed machines")

    wp_df = load_table_data(db, "wear_profiles", user_id)
    ib_df = load_table_data(db, "installed_base", user_id, scenario_id)

    col1, col2 = st.columns(2)
    col1.success(f"✅ {len(wp_df)} wear profiles" if not wp_df.empty else "⚠️ No wear profiles")
    col2.success(f"✅ {len(ib_df)} machines" if not ib_df.empty else "⚠️ No machines")

    st.markdown("---")
    tab_wp, tab_ib = st.tabs(["⚙️ Wear Profiles", "🏗️ Installed Base"])

    with tab_wp:
        st.markdown("### Wear Profiles")
        if not wp_df.empty:
            with st.expander("📋 View Current Profiles"):
                st.dataframe(wp_df, use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear Wear Profiles", key="clear_wear"):
                if clear_table_data(db, "wear_profiles", user_id):
                    st.success("Cleared!")
                    st.rerun()
        st.markdown("---")
        if COLUMN_MAPPER_AVAILABLE:
            render_import_with_mapping(db, user_id, "wear_profiles")
        else:
            st.warning("Column mapper not available")

    with tab_ib:
        st.markdown("### Installed Base")
        if not scenario_id:
            st.warning("⚠️ Select a scenario to manage installed base")
            return
        if not ib_df.empty:
            with st.expander("📋 View Current Machines"):
                st.dataframe(ib_df.head(20), use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear Installed Base", key="clear_fleet"):
                if clear_table_data(db, "installed_base", user_id, scenario_id):
                    st.success("Cleared!")
                    st.rerun()
        st.markdown("---")
        if COLUMN_MAPPER_AVAILABLE:
            render_import_with_mapping(db, user_id, "installed_base", scenario_id)
        else:
            st.warning("Column mapper not available")


def render_step_working_capital(db, scenario_id: str, user_id: str):
    st.subheader("💰 Working Capital")
    st.caption("Import aged debtors, suppliers, and aged creditors")

    deb_df = load_table_data(db, "aged_debtors", user_id)
    cred_df = load_table_data(db, "aged_creditors", user_id)
    sup_df = load_table_data(db, "creditors", user_id, scenario_id) if scenario_id else pd.DataFrame()

    col1, col2, col3 = st.columns(3)
    col1.metric("Debtors", f"{len(deb_df)} inv", f"R {deb_df['amount_due'].sum():,.0f}" if "amount_due" in deb_df else "")
    col2.metric("Suppliers", len(sup_df))
    col3.metric("Creditors", f"{len(cred_df)} inv", f"R {cred_df['amount_due'].sum():,.0f}" if "amount_due" in cred_df else "")

    st.markdown("---")
    tab_deb, tab_sup, tab_cred = st.tabs(["📈 Aged Debtors", "🏭 Suppliers", "📉 Aged Creditors"])

    with tab_deb:
        st.markdown("### Aged Debtors (Receivables)")
        if not deb_df.empty:
            with st.expander("📋 View Current Debtors"):
                st.dataframe(deb_df.head(20), use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear Debtors", key="clear_debtors"):
                if clear_table_data(db, "aged_debtors", user_id):
                    st.success("Cleared!")
                    st.rerun()
        st.markdown("---")
        if COLUMN_MAPPER_AVAILABLE:
            render_import_with_mapping(db, user_id, "aged_debtors")
        else:
            st.warning("Column mapper not available")

    with tab_sup:
        st.markdown("### Suppliers")
        if not scenario_id:
            st.warning("⚠️ Select a scenario to import suppliers")
        else:
            if not sup_df.empty:
                with st.expander("📋 View Current Suppliers"):
                    st.dataframe(sup_df, use_container_width=True, hide_index=True)
                if st.button("🗑️ Clear Suppliers", key="clear_suppliers"):
                    if clear_table_data(db, "creditors", user_id, scenario_id):
                        st.success("Cleared!")
                        st.rerun()
            st.markdown("---")
            if COLUMN_MAPPER_AVAILABLE:
                render_import_with_mapping(db, user_id, "suppliers", scenario_id)
            else:
                st.warning("Column mapper not available")

    with tab_cred:
        st.markdown("### Aged Creditors (Payables)")
        if not cred_df.empty:
            with st.expander("📋 View Current Creditors"):
                st.dataframe(cred_df.head(20), use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear Creditors", key="clear_creditors"):
                if clear_table_data(db, "aged_creditors", user_id):
                    st.success("Cleared!")
                    st.rerun()
        st.markdown("---")
        if COLUMN_MAPPER_AVAILABLE:
            render_import_with_mapping(db, user_id, "aged_creditors")
        else:
            st.warning("Column mapper not available")


def render_step_pipeline(db, scenario_id: str, user_id: str):
    st.subheader("🎯 Sales Pipeline")
    st.caption("Import sales opportunities and prospects (optional)")

    if not scenario_id:
        st.warning("⚠️ Select a scenario to manage pipeline")
        return

    df = load_table_data(db, "prospects", user_id, scenario_id)
    if not df.empty:
        st.success(f"✅ {len(df)} opportunities in pipeline")
        if "annual_liner_value" in df.columns and "confidence_pct" in df.columns:
            df["weighted"] = df["annual_liner_value"].fillna(0) * df["confidence_pct"].fillna(0.5)
            col1, col2 = st.columns(2)
            col1.metric("Total Pipeline", f"R {df['annual_liner_value'].sum():,.0f}")
            col2.metric("Weighted Value", f"R {df['weighted'].sum():,.0f}")
        with st.expander("📋 View Current Pipeline"):
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear Pipeline", key="clear_pipeline"):
            if clear_table_data(db, "prospects", user_id, scenario_id):
                st.success("Cleared!")
                st.rerun()
    else:
        st.info("No pipeline data - optional")

    st.markdown("---")
    if COLUMN_MAPPER_AVAILABLE:
        render_import_with_mapping(db, user_id, "prospects", scenario_id)
    else:
        st.warning("Column mapper not available")


def render_step_historics(db, scenario_id: str, user_id: str):
    """Render historical data step using only detailed line items (IS/BS/CF)."""
    st.subheader("📊 Historical Data (Line-Item Only)")
    st.caption("Import detailed line items for Income Statement, Balance Sheet, and Cash Flow.")

    if not scenario_id:
        st.warning("⚠️ Select a scenario to import historical data")
        return

    is_li_df = load_table_data(db, "historical_income_statement_line_items", user_id, scenario_id)
    bs_li_df = load_table_data(db, "historical_balance_sheet_line_items", user_id, scenario_id)
    cf_li_df = load_table_data(db, "historical_cashflow_line_items", user_id, scenario_id)

    col1, col2, col3 = st.columns(3)
    with col1:
        if not is_li_df.empty:
            periods = is_li_df["period_date"].nunique() if "period_date" in is_li_df else 0
            items = is_li_df["line_item_name"].nunique() if "line_item_name" in is_li_df else 0
            st.success(f"✅ P&L: {items} items, {periods} periods")
        else:
            st.info("⬜ P&L Line Items")
    with col2:
        if not bs_li_df.empty:
            periods = bs_li_df["period_date"].nunique() if "period_date" in bs_li_df else 0
            items = bs_li_df["line_item_name"].nunique() if "line_item_name" in bs_li_df else 0
            st.success(f"✅ BS: {items} items, {periods} periods")
        else:
            st.info("⬜ BS Line Items")
    with col3:
        if not cf_li_df.empty:
            periods = cf_li_df["period_date"].nunique() if "period_date" in cf_li_df else 0
            items = cf_li_df["line_item_name"].nunique() if "line_item_name" in cf_li_df else 0
            st.success(f"✅ CF: {items} items, {periods} periods")
        else:
            st.info("⬜ CF Line Items")

    st.markdown("---")
    sub_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])

    with sub_tabs[0]:
        st.markdown("### Income Statement Line Items")
        if not is_li_df.empty:
            with st.expander("📋 View Current Income Statement Line Items"):
                st.dataframe(is_li_df.head(100), use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear Income Statement Line Items", key="clear_is_li"):
                if clear_table_data(db, "historical_income_statement_line_items", user_id, scenario_id):
                    st.success("Cleared!")
                    st.rerun()
        st.markdown("---")
        if COLUMN_MAPPER_AVAILABLE:
            render_import_with_mapping(db, user_id, "historical_income_statement_line_items", scenario_id)
        else:
            st.warning("Column mapper not available")

    with sub_tabs[1]:
        st.markdown("### Balance Sheet Line Items")
        if not bs_li_df.empty:
            with st.expander("📋 View Current Balance Sheet Line Items"):
                st.dataframe(bs_li_df.head(100), use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear Balance Sheet Line Items", key="clear_bs_li"):
                if clear_table_data(db, "historical_balance_sheet_line_items", user_id, scenario_id):
                    st.success("Cleared!")
                    st.rerun()
        st.markdown("---")
        if COLUMN_MAPPER_AVAILABLE:
            render_import_with_mapping(db, user_id, "historical_balance_sheet_line_items", scenario_id)
        else:
            st.warning("Column mapper not available")

    with sub_tabs[2]:
        st.markdown("### Cash Flow Line Items")
        if not cf_li_df.empty:
            with st.expander("📋 View Current Cash Flow Line Items"):
                st.dataframe(cf_li_df.head(100), use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear Cash Flow Line Items", key="clear_cf_li"):
                if clear_table_data(db, "historical_cashflow_line_items", user_id, scenario_id):
                    st.success("Cleared!")
                    st.rerun()
        st.markdown("---")
        if COLUMN_MAPPER_AVAILABLE:
            render_import_with_mapping(db, user_id, "historical_cashflow_line_items", scenario_id)
        else:
            st.warning("Column mapper not available")


def render_step_costs(db, scenario_id: str, user_id: str):
    st.subheader("💵 Expense Configuration")
    st.caption("Configure OPEX functions (optional)")

    has_expense_methods = hasattr(db, "get_expense_assumptions")
    if not has_expense_methods:
        st.info("Expense forecasting uses default assumptions.")
        defaults = pd.DataFrame(
            [
                {"Category": "Personnel", "Type": "Fixed", "Monthly": "R500,000"},
                {"Category": "Logistics", "Type": "3% of Revenue", "Monthly": "Variable"},
                {"Category": "Professional", "Type": "Fixed", "Monthly": "R50,000"},
                {"Category": "Facilities", "Type": "Fixed", "Monthly": "R100,000"},
                {"Category": "Marketing", "Type": "Fixed", "Monthly": "R30,000"},
            ]
        )
        st.dataframe(defaults, use_container_width=True, hide_index=True)
        return

    try:
        expenses = db.get_expense_assumptions(scenario_id)
        if expenses:
            st.success(f"✅ {len(expenses)} expense items configured")
            df = pd.DataFrame(expenses)
            cols = ["expense_name", "expense_category", "function_type", "fixed_monthly"]
            st.dataframe(df[[c for c in cols if c in df.columns]], use_container_width=True, hide_index=True)
        else:
            st.info("No custom expense assumptions configured - using defaults")
            if st.button("📥 Load Default Expenses"):
                defaults = [
                    {
                        "expense_code": "personnel",
                        "expense_name": "Personnel",
                        "expense_category": "personnel",
                        "function_type": "fixed",
                        "fixed_monthly": 500000,
                    },
                    {
                        "expense_code": "logistics",
                        "expense_name": "Logistics",
                        "expense_category": "logistics",
                        "function_type": "variable",
                        "variable_rate": 0.03,
                    },
                    {
                        "expense_code": "professional",
                        "expense_name": "Professional Fees",
                        "expense_category": "professional",
                        "function_type": "fixed",
                        "fixed_monthly": 50000,
                    },
                    {
                        "expense_code": "facilities",
                        "expense_name": "Facilities",
                        "expense_category": "facilities",
                        "function_type": "fixed",
                        "fixed_monthly": 100000,
                    },
                    {
                        "expense_code": "marketing",
                        "expense_name": "Marketing",
                        "expense_category": "marketing",
                        "function_type": "fixed",
                        "fixed_monthly": 30000,
                    },
                ]
                try:
                    count = db.bulk_upsert_expense_assumptions(scenario_id, defaults)
                    if count > 0:
                        st.success(f"✅ Loaded {count} default expense items")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to load defaults: {e}")
    except Exception as e:
        st.warning(f"Could not load expenses: {e}")


# =============================================================================
# STEP RENDERER MAP
# =============================================================================

STEP_RENDERERS = {
    "basics": render_step_basics,
    "customers": render_step_customers,
    "fleet": render_step_fleet,
    "working_capital": render_step_working_capital,
    "pipeline": render_step_pipeline,
    "historics": render_step_historics,
    "costs": render_step_costs,
}

# =============================================================================
# PROGRESS INDICATOR
# =============================================================================


def render_progress_indicator(current_step: str, step_status: Dict[str, bool]):
    cols = st.columns(len(STEPS))
    for i, (step_id, step_name, _) in enumerate(STEPS):
        is_current = step_id == current_step
        is_complete = step_status.get(step_id, False)
        if is_complete:
            icon, color = "✅", "#10b981"
        elif is_current:
            icon, color = "🔵", "#3b82f6"
        else:
            icon, color = "⬜", "#64748b"
        with cols[i]:
            st.markdown(
                f"""
            <div style="text-align: center;">
                <div style="font-size: 1.2rem;">{icon}</div>
                <div style="font-size: 0.7rem; color: {color};">{step_name}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


# =============================================================================
# IMPORT STATUS SUMMARY
# =============================================================================


def render_import_status_summary(db, scenario_id: str, user_id: str):
    st.markdown("### 📋 Data Status")
    status_items = []
    cust = load_table_data(db, "customers", user_id)
    status_items.append(("👥 Customers", len(cust), "✅" if len(cust) > 0 else "⬜"))
    wp = load_table_data(db, "wear_profiles", user_id)
    status_items.append(("⚙️ Wear Profiles", len(wp), "✅" if len(wp) > 0 else "⬜"))
    if scenario_id:
        ib = load_table_data(db, "installed_base", user_id, scenario_id)
        status_items.append(("🏗️ Machines", len(ib), "✅" if len(ib) > 0 else "⬜"))
        pipe = load_table_data(db, "prospects", user_id, scenario_id)
        status_items.append(("🎯 Pipeline", len(pipe), "✅" if len(pipe) > 0 else "⬜"))
        is_li_df = load_table_data(db, "historical_income_statement_line_items", user_id, scenario_id)
        bs_li_df = load_table_data(db, "historical_balance_sheet_line_items", user_id, scenario_id)
        cf_li_df = load_table_data(db, "historical_cashflow_line_items", user_id, scenario_id)
        status_items.append(("📊 IS Line Items", len(is_li_df), "✅" if len(is_li_df) > 0 else "⬜"))
        status_items.append(("📋 BS Line Items", len(bs_li_df), "✅" if len(bs_li_df) > 0 else "⬜"))
        status_items.append(("💵 CF Line Items", len(cf_li_df), "✅" if len(cf_li_df) > 0 else "⬜"))

    cols = st.columns(len(status_items))
    for i, (label, count, icon) in enumerate(status_items):
        with cols[i]:
            st.markdown(f"**{icon} {label}**")
            st.caption(f"{count} records")


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================


def render_setup_wizard(db, scenario_id: str, user_id: str, initial_step: str = "basics"):
    st.header("⚙️ Setup Wizard")
    st.caption("Configure your scenario step by step with flexible CSV imports")

    if not COLUMN_MAPPER_AVAILABLE:
        st.warning(
            """
        ⚠️ Column Mapper Not Found

        Install `column_mapper.py` in the `components` folder to enable flexible CSV mapping.
        """
        )

    if "setup_step" not in st.session_state:
        st.session_state.setup_step = initial_step

    current_step = st.session_state.setup_step

    step_status = {}
    for step_id, _, _ in STEPS:
        check_fn = STEP_CHECKS.get(step_id)
        if check_fn:
            try:
                step_status[step_id] = check_fn(db, scenario_id, user_id)
            except Exception:
                step_status[step_id] = False
        else:
            step_status[step_id] = False

    render_progress_indicator(current_step, step_status)
    st.markdown("---")

    step_names = [name for _, name, _ in STEPS]
    step_ids = [sid for sid, _, _ in STEPS]
    tabs = st.tabs(step_names)

    for i, (step_id, _, _) in enumerate(STEPS):
        with tabs[i]:
            renderer = STEP_RENDERERS.get(step_id)
            if renderer:
                try:
                    renderer(db, scenario_id, user_id)
                except Exception as e:
                    st.error(f"Error rendering step: {e}")
            else:
                st.warning(f"Step '{step_id}' not implemented")

            st.markdown("---")
            col_prev, col_spacer, col_next = st.columns([1, 2, 1])
            with col_prev:
                if i > 0:
                    if st.button("← Previous", key=f"prev_{step_id}", use_container_width=True):
                        st.session_state.setup_step = step_ids[i - 1]
                        st.rerun()
            with col_next:
                if i < len(STEPS) - 1:
                    if st.button("Next →", key=f"next_{step_id}", type="primary", use_container_width=True):
                        st.session_state.setup_step = step_ids[i + 1]
                        st.rerun()
                else:
                    if st.button("✅ Finish Setup", key=f"finish_{step_id}", type="primary", use_container_width=True):
                        st.session_state.current_section = "forecast"
                        st.success("Setup complete! Ready to run forecast.")
                        st.rerun()


# =============================================================================
# COMPACT WIZARD (Alternative)
# =============================================================================


def render_compact_setup_wizard(db, scenario_id: str, user_id: str):
    st.header("⚙️ Quick Setup")
    render_import_status_summary(db, scenario_id, user_id)
    st.markdown("---")
    for step_id, step_name, step_desc in STEPS:
        with st.expander(f"{step_name} - {step_desc}"):
            renderer = STEP_RENDERERS.get(step_id)
            if renderer:
                try:
                    renderer(db, scenario_id, user_id)
                except Exception as e:
                    st.error(f"Error: {e}")