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


def _get_monthly_sales_pattern_api(db, scenario_id: str) -> Optional[np.ndarray]:
    """
    API-safe seasonality extraction.
    Prefers `granular_sales_history` (invoice-level) scoped to the scenario's user_id.
    Returns 12-element weights that sum to 1.0, or None.
    """
    try:
        scen = db.client.table("scenarios").select("user_id").eq("id", scenario_id).limit(1).execute()
        scen_user_id = None
        if scen.data and scen.data[0].get("user_id"):
            scen_user_id = scen.data[0]["user_id"]
        if not scen_user_id:
            return None
        resp = (
            db.client.table("granular_sales_history")
            .select("date, quantity, unit_price_sold")
            .eq("user_id", scen_user_id)
            .execute()
        )
        rows = resp.data or []
        if len(rows) < 12:
            return None
        sdf = pd.DataFrame(rows)
        sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")
        sdf = sdf.dropna(subset=["date"])
        if sdf.empty:
            return None
        sdf["month"] = sdf["date"].dt.month
        sdf["quantity"] = pd.to_numeric(sdf.get("quantity"), errors="coerce").fillna(0.0)
        sdf["unit_price_sold"] = pd.to_numeric(sdf.get("unit_price_sold"), errors="coerce").fillna(0.0)
        sdf["sales"] = sdf["quantity"] * sdf["unit_price_sold"]
        monthly = sdf.groupby("month")["sales"].sum()
        if int((monthly > 0).sum()) < 6:
            return None
        pattern = np.zeros(12, dtype=float)
        for m in range(1, 13):
            pattern[m - 1] = float(monthly.get(m, 0.0))
        s = float(pattern.sum())
        return (pattern / s) if s > 0 else None
    except Exception:
        return None


def _upsample_annual_or_ytd_is(
    df: pd.DataFrame,
    sales_pattern: Optional[np.ndarray],
    fiscal_year_end_month: int = 12,
    ytd_overrides: Optional[Dict[int, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    If income statement history has <=1 row per year (annual/YTD), upsample to monthly.
    If YTD (<12 months covered), annualize to 12 months before splitting (gross-up).
    """
    if df.empty or "period_date" not in df.columns:
        return df
    df = df.copy()
    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df = df.dropna(subset=["period_date"])
    if df.empty:
        return df
    counts = df["period_date"].dt.year.value_counts()
    if counts.max() > 1:
        return df

    fiscal_year_end_month = int(fiscal_year_end_month or 12)
    ytd_overrides = ytd_overrides or {}

    def _fy_end_year(dt: pd.Timestamp) -> int:
        return int(dt.year) if int(dt.month) <= fiscal_year_end_month else int(dt.year) + 1

    if sales_pattern is not None and len(sales_pattern) == 12:
        s = float(np.sum(sales_pattern))
        sales_pattern = (sales_pattern / s) if s > 0 else None
    else:
        sales_pattern = None

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        dt = pd.to_datetime(r["period_date"], errors="coerce")
        if pd.isna(dt):
            continue
        fy_end = pd.Timestamp(_fy_end_year(dt), fiscal_year_end_month, 1)
        fy_start = (fy_end - pd.DateOffset(months=11)).to_period("M").to_timestamp()
        fy_months = pd.date_range(start=fy_start, periods=12, freq="MS")

        dt_m = dt.to_period("M").to_timestamp()
        months_covered = int(sum(m <= dt_m for m in fy_months))
        months_covered = max(1, min(months_covered, 12))

        override = ytd_overrides.get(int(_fy_end_year(dt)))
        if override and override.get("ytd_end_calendar_month"):
            cal_m = int(override["ytd_end_calendar_month"])
            candidate = pd.Timestamp(fy_end.year if cal_m <= fiscal_year_end_month else fy_end.year - 1, cal_m, 1)
            if candidate in set(fy_months):
                months_covered = int(list(fy_months).index(candidate)) + 1

        annualize_factor = 12 / months_covered if months_covered < 12 else 1.0

        for m_start in fy_months:
            if sales_pattern is not None:
                w = float(sales_pattern[int(m_start.month) - 1])
            else:
                w = 1.0 / 12.0
            new = {"period_date": pd.to_datetime(m_start)}
            for c in df.columns:
                if c == "period_date":
                    continue
                val = float(r.get(c) or 0.0)
                new[c] = val * annualize_factor * w
            rows.append(new)

    if not rows:
        return df
    out = pd.DataFrame(rows)
    out["period_date"] = pd.to_datetime(out["period_date"], errors="coerce")
    return out.sort_values("period_date")


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
            # If annual/YTD-only imports exist, upsample to monthly (gross-up YTD to full year)
            try:
                assumptions = data.get("assumptions") or {}
                ip = (assumptions.get("import_period_settings") or {}) if isinstance(assumptions, dict) else {}
                fy_end_month = 12
                ytd_overrides: Dict[int, Dict[str, Any]] = {}
                if isinstance(ip, dict):
                    cfg_block = ip.get("historical_income_statement_line_items") or {}
                    fy_end_month = int(cfg_block.get("fiscal_year_end_month") or fy_end_month)
                    ytd_cfg = cfg_block.get("ytd") or None
                    if isinstance(ytd_cfg, dict) and ytd_cfg.get("year"):
                        ytd_overrides[int(ytd_cfg["year"])] = {
                            "year_end_month": int(ytd_cfg.get("year_end_month") or fy_end_month),
                            "ytd_end_calendar_month": int(ytd_cfg.get("ytd_end_calendar_month") or 0) or None,
                        }
                pattern = _get_monthly_sales_pattern_api(db, scenario_id)
                hist_df = _upsample_annual_or_ytd_is(hist_df, pattern, fiscal_year_end_month=fy_end_month, ytd_overrides=ytd_overrides)
            except Exception:
                pass
            data["historic_financials"] = hist_df.rename(columns={"period_date": "period_date"})
            # Provide a simple revenue series for legacy paths
            data["historical_revenue"] = hist_df.set_index("period_date")["revenue"].sort_index()

    return data


def get_historics_diagnostics_api(db, scenario_id: str, user_id: str) -> Dict[str, Any]:
    """
    Diagnostics helper for thin-client debugging.
    Returns expected vs found period coverage + monthly IS bucket totals (post-upsampling if annual/YTD-only).
    """
    assumptions: Dict[str, Any] = {}
    try:
        if hasattr(db, "get_scenario_assumptions"):
            assumptions = db.get_scenario_assumptions(scenario_id, user_id) or {}
    except Exception:
        assumptions = {}

    ip = (assumptions or {}).get("import_period_settings") or {}
    expected = []
    try:
        blk = (ip.get("historical_income_statement_line_items") or {}) if isinstance(ip, dict) else {}
        sel = blk.get("selected_periods") or []
        if isinstance(sel, list):
            expected = [str(x).strip() for x in sel if str(x).strip()]
    except Exception:
        expected = []

    hist_df, diag = load_income_statement_history(db, scenario_id)
    used_sales_pattern = False
    try:
        fy_end_month = 12
        ytd_overrides: Dict[int, Dict[str, Any]] = {}
        if isinstance(ip, dict):
            blk = ip.get("historical_income_statement_line_items") or {}
            fy_end_month = int(blk.get("fiscal_year_end_month") or fy_end_month)
            ytd_cfg = blk.get("ytd") or None
            if isinstance(ytd_cfg, dict) and ytd_cfg.get("year"):
                ytd_overrides[int(ytd_cfg["year"])] = {
                    "year_end_month": int(ytd_cfg.get("year_end_month") or fy_end_month),
                    "ytd_end_calendar_month": int(ytd_cfg.get("ytd_end_calendar_month") or 0) or None,
                }
        pattern = _get_monthly_sales_pattern_api(db, scenario_id)
        used_sales_pattern = bool(pattern is not None)
        hist_df = _upsample_annual_or_ytd_is(hist_df, pattern, fiscal_year_end_month=fy_end_month, ytd_overrides=ytd_overrides)
    except Exception:
        pass

    found = []
    monthly = []
    if not hist_df.empty and "period_date" in hist_df.columns:
        _p = pd.to_datetime(hist_df["period_date"], errors="coerce").dt.to_period("M").astype(str)
        found = sorted(set([x for x in _p.dropna().tolist() if isinstance(x, str) and x.strip()]))

        view = hist_df.copy()
        view["period"] = pd.to_datetime(view["period_date"], errors="coerce").dt.to_period("M").astype(str)
        keep_cols = [c for c in ["period", "revenue", "cogs", "opex", "depreciation", "interest", "tax", "other_income", "gross_profit", "ebit"] if c in view.columns]
        view = view[keep_cols].copy()
        for c in keep_cols:
            if c != "period":
                view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0.0)
        monthly = view.sort_values("period").to_dict(orient="records")

    missing = sorted(set(expected) - set(found)) if expected else []
    extra = sorted(set(found) - set(expected)) if expected else []

    return {
        "scenario_id": scenario_id,
        "expected_periods_is": expected,
        "found_periods_is": found,
        "missing_periods_is": missing,
        "extra_periods_is": extra,
        "income_statement_monthly": monthly,
        "used_sales_pattern": used_sales_pattern,
        "source_diag": diag,
    }

