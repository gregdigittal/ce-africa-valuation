from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _classify_is_row(category: Optional[str], name: Optional[str]) -> str:
    c = (category or "").strip().lower()
    n = (name or "").strip().lower()

    if "revenue" in c or "sales" in c or "turnover" in c:
        return "revenue"
    if ("cost of sales" in c) or ("cogs" in c) or ("cost of goods" in c) or ("purchases" in c):
        return "cogs"
    if ("depreciation" in c) or ("amort" in c) or ("depreciation" in n) or ("amort" in n):
        return "depreciation"
    if ("finance cost" in c) or ("finance costs" in c) or ("interest" in c) or ("interest" in n):
        return "interest"
    if ("tax" in c) or ("taxation" in c) or ("tax" in n):
        return "tax"
    if ("other income" in c) or ("finance income" in c) or ("interest received" in n):
        return "other_income"
    if ("expense" in c) or ("operating" in c) or ("distribution" in c) or ("overhead" in c) or ("admin" in c):
        return "opex"
    return "opex"


def load_income_statement_history(db, scenario_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and aggregate detailed income statement line-items into canonical buckets.

    Returns:
      (df, diag)
      df columns: period_date, revenue, cogs, opex, depreciation, interest, tax, other_income, gross_profit, ebit
    """
    diag: Dict[str, Any] = {"source": "historical_income_statement_line_items", "rows": 0}
    try:
        resp = (
            db.client.table("historical_income_statement_line_items")
            .select("period_date, category, sub_category, line_item_name, amount")
            .eq("scenario_id", scenario_id)
            .order("period_date")
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        diag["error"] = str(e)
        return pd.DataFrame(), diag

    if not rows:
        diag["rows"] = 0
        return pd.DataFrame(), diag

    df = pd.DataFrame(rows)
    diag["rows"] = len(df)

    if "period_date" not in df.columns:
        return pd.DataFrame(), {**diag, "error": "Missing period_date column in response"}

    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce").fillna(0.0)

    df["bucket"] = df.apply(
        lambda r: _classify_is_row(r.get("category"), r.get("line_item_name")),
        axis=1,
    )

    # Sum by month+bucket
    grouped = df.groupby(["period_date", "bucket"], dropna=False)["amount"].sum().reset_index()
    pivot = grouped.pivot_table(index="period_date", columns="bucket", values="amount", aggfunc="sum").fillna(0.0)

    # Ensure expected columns exist
    for col in ["revenue", "cogs", "opex", "depreciation", "interest", "tax", "other_income"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    out = pivot.reset_index().copy()

    # Normalize sign conventions: costs should be positive numbers (we'll subtract later)
    for cost_col in ["cogs", "opex", "depreciation", "interest", "tax"]:
        out[cost_col] = out[cost_col].abs()

    out["gross_profit"] = out["revenue"] - out["cogs"]
    out["ebit"] = out["gross_profit"] - out["opex"] - out["depreciation"] + out["other_income"]

    out = out.sort_values("period_date")
    return out, diag


def load_forecast_data_api(
    db,
    scenario_id: str,
    user_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    API-safe version of `components.forecast_section.load_forecast_data` (no Streamlit caching).
    """
    options = options or {}
    data: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "user_id": user_id,
        "machines": [],
        "prospects": [],
        "expenses": [],
        "assumptions": {},
        "ai_assumptions": None,
        "historic_financials": pd.DataFrame(),
        "data_source": "machine_instances",
    }

    # Assumptions
    try:
        if hasattr(db, "get_scenario_assumptions"):
            data["assumptions"] = db.get_scenario_assumptions(scenario_id, user_id) or {}
    except Exception:
        data["assumptions"] = {}

    # Allow request-time overrides
    if "forecast_duration_months" in options:
        data["assumptions"]["forecast_duration_months"] = int(options["forecast_duration_months"])
    if options.get("forecast_method") in {"trend", "pipeline"}:
        data["assumptions"]["forecast_method"] = options["forecast_method"]
    if "use_trend_forecast" in options:
        data["assumptions"]["use_trend_forecast"] = bool(options["use_trend_forecast"])

    # Machines (prefer machine_instances)
    try:
        select_expr = (
            "*, sites(site_name, ore_type_id, customers(customer_name)), "
            "wear_profiles_v2(profile_name, liner_life_months, avg_consumable_revenue, "
            "refurb_interval_months, avg_refurb_revenue, ore_type_id, gross_margin_liner, gross_margin_refurb)"
        )
        machines = (
            db.client.table("machine_instances")
            .select(select_expr)
            .eq("scenario_id", scenario_id)
            .eq("status", "Active")
            .execute()
        )
        if machines.data:
            data["machines"] = machines.data
            data["data_source"] = "machine_instances"
    except Exception:
        pass

    if not data["machines"]:
        # Fallback: installed_base (legacy)
        try:
            ib = db.client.table("installed_base").select("*").eq("scenario_id", scenario_id).execute()
            if ib.data:
                data["machines"] = ib.data
                data["data_source"] = "installed_base"
        except Exception:
            pass

    # Prospects + Expenses
    try:
        resp = db.client.table("prospects").select("*").eq("scenario_id", scenario_id).execute()
        data["prospects"] = resp.data or []
    except Exception:
        data["prospects"] = []

    try:
        resp = (
            db.client.table("expense_assumptions")
            .select("*")
            .eq("scenario_id", scenario_id)
            .eq("is_active", True)
            .execute()
        )
        data["expenses"] = resp.data or []
    except Exception:
        data["expenses"] = []

    # History (only if trend is requested; keeps jobs fast by default)
    if bool(data["assumptions"].get("use_trend_forecast")) or data["assumptions"].get("forecast_method") == "trend":
        hist_df, _diag = load_income_statement_history(db, scenario_id)
        if not hist_df.empty:
            data["historic_financials"] = hist_df.rename(columns={"period_date": "period_date"})
            # Provide a simple revenue series for legacy paths
            data["historical_revenue"] = hist_df.set_index("period_date")["revenue"].sort_index()

    return data

