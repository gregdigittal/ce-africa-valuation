from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Supabase helpers (avoid silent 1000-row truncation on large selects)
try:
    from supabase_pagination import fetch_all_rows
except Exception:
    fetch_all_rows = None


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
        q = (
            db.client.table("historical_income_statement_line_items")
            .select("id, period_date, category, sub_category, line_item_name, amount")
            .eq("scenario_id", scenario_id)
            .order("period_date")
        )
        if fetch_all_rows:
            rows = fetch_all_rows(q, order_by="id")
        else:
            resp = q.execute()
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
        q = (
            db.client.table("granular_sales_history")
            .select("date, quantity, unit_price_sold")
            .eq("user_id", scen_user_id)
        )
        if fetch_all_rows:
            # Safety cap to avoid runaway loads; seasonality converges quickly
            rows = fetch_all_rows(q, order_by="date", max_rows=50_000)
        else:
            resp = q.execute()
            rows = resp.data or []

        if len(rows) >= 12:
            sdf = pd.DataFrame(rows)
            sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")
            sdf = sdf.dropna(subset=["date"])
            if not sdf.empty:
                sdf["month"] = sdf["date"].dt.month
                sdf["quantity"] = pd.to_numeric(sdf.get("quantity"), errors="coerce").fillna(0.0)
                sdf["unit_price_sold"] = pd.to_numeric(sdf.get("unit_price_sold"), errors="coerce").fillna(0.0)
                sdf["sales"] = sdf["quantity"] * sdf["unit_price_sold"]
                monthly = sdf.groupby("month")["sales"].sum()
                if int((monthly > 0).sum()) >= 6:
                    pattern = np.zeros(12, dtype=float)
                    for m in range(1, 13):
                        pattern[m - 1] = float(monthly.get(m, 0.0))
                    s = float(pattern.sum())
                    if s > 0:
                        return pattern / s

        # Fallback: scenario-scoped sales orders (requires sales_orders table)
        try:
            so_q = (
                db.client.table("sales_orders")
                .select("id, order_date, total_amount")
                .eq("scenario_id", scenario_id)
                .eq("user_id", scen_user_id)
            )
            if fetch_all_rows:
                so_rows = fetch_all_rows(so_q, order_by="id", max_rows=50_000)
            else:
                so = so_q.execute()
                so_rows = so.data or []
            if len(so_rows) >= 12:
                odf = pd.DataFrame(so_rows)
                odf["order_date"] = pd.to_datetime(odf["order_date"], errors="coerce")
                odf = odf.dropna(subset=["order_date"])
                if not odf.empty:
                    odf["month"] = odf["order_date"].dt.month
                    odf["total_amount"] = pd.to_numeric(odf.get("total_amount"), errors="coerce").fillna(0.0)
                    monthly2 = odf.groupby("month")["total_amount"].sum()
                    if int((monthly2 > 0).sum()) >= 6:
                        pattern = np.zeros(12, dtype=float)
                        for m in range(1, 13):
                            pattern[m - 1] = float(monthly2.get(m, 0.0))
                        s = float(pattern.sum())
                        if s > 0:
                            return pattern / s
        except Exception:
            pass

        return None
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

    def _expected_periods(key: str) -> List[str]:
        try:
            blk = (ip.get(key) or {}) if isinstance(ip, dict) else {}
            sel = blk.get("selected_periods") or []
            if isinstance(sel, list):
                return [str(x).strip() for x in sel if str(x).strip()]
        except Exception:
            pass
        return []

    expected_is = _expected_periods("historical_income_statement_line_items")
    expected_bs = _expected_periods("historical_balance_sheet_line_items")
    expected_cf = _expected_periods("historical_cashflow_line_items")

    def _fetch_periods(table: str) -> List[str]:
        try:
            resp = (
                db.client.table(table)
                .select("period_date")
                .eq("scenario_id", scenario_id)
                .eq("user_id", user_id)
                .order("period_date")
                .execute()
            )
            rows = resp.data or []
            if not rows:
                return []
            df = pd.DataFrame(rows)
            if "period_date" not in df.columns:
                return []
            return (
                pd.to_datetime(df["period_date"], errors="coerce")
                .dt.to_period("M")
                .astype(str)
                .dropna()
                .tolist()
            )
        except Exception:
            return []

    def _bs_bucketize(category: str, name: str) -> str:
        s = f"{(category or '').strip().lower()} {(name or '').strip().lower()}".strip()
        if "asset" in s:
            return "assets"
        if any(k in s for k in ["liabil", "payable", "debt"]):
            return "liabilities"
        if any(k in s for k in ["equity", "retained", "share capital"]):
            return "equity"
        return "other"

    def _cf_bucketize(category: str, name: str) -> str:
        s = f"{(category or '').strip().lower()} {(name or '').strip().lower()}".strip()
        if any(k in s for k in ["operating", "operations", "cfo"]):
            return "operating"
        if any(k in s for k in ["investing", "cfi", "capex", "capital"]):
            return "investing"
        if any(k in s for k in ["financing", "cff", "debt", "equity", "dividend"]):
            return "financing"
        return "other"

    def _summarize_bs() -> List[Dict[str, Any]]:
        try:
            q = (
                db.client.table("historical_balance_sheet_line_items")
                .select("id, period_date, category, line_item_name, amount")
                .eq("scenario_id", scenario_id)
                .eq("user_id", user_id)
                .order("period_date")
            )
            if fetch_all_rows:
                rows = fetch_all_rows(q, order_by="id")
            else:
                resp = q.execute()
                rows = resp.data or []
            if not rows:
                return []
            df = pd.DataFrame(rows)
            df["period"] = pd.to_datetime(df["period_date"], errors="coerce").dt.to_period("M").astype(str)
            df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce").fillna(0.0)
            df["bucket"] = df.apply(lambda r: _bs_bucketize(r.get("category"), r.get("line_item_name")), axis=1)
            grouped = df.groupby(["period", "bucket"], dropna=False)["amount"].sum().unstack("bucket", fill_value=0.0).reset_index()
            for c in ["assets", "liabilities", "equity"]:
                if c not in grouped.columns:
                    grouped[c] = 0.0
            grouped["balance_check"] = grouped["assets"] - (grouped["liabilities"] + grouped["equity"])
            out = grouped.sort_values("period")
            for c in ["assets", "liabilities", "equity", "balance_check"]:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
            return out.to_dict(orient="records")
        except Exception:
            return []

    def _summarize_cf() -> List[Dict[str, Any]]:
        try:
            q = (
                db.client.table("historical_cashflow_line_items")
                .select("id, period_date, category, line_item_name, amount")
                .eq("scenario_id", scenario_id)
                .eq("user_id", user_id)
                .order("period_date")
            )
            if fetch_all_rows:
                rows = fetch_all_rows(q, order_by="id")
            else:
                resp = q.execute()
                rows = resp.data or []
            if not rows:
                return []
            df = pd.DataFrame(rows)
            df["period"] = pd.to_datetime(df["period_date"], errors="coerce").dt.to_period("M").astype(str)
            df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce").fillna(0.0)
            df["bucket"] = df.apply(lambda r: _cf_bucketize(r.get("category"), r.get("line_item_name")), axis=1)
            grouped = df.groupby(["period", "bucket"], dropna=False)["amount"].sum().unstack("bucket", fill_value=0.0).reset_index()
            for c in ["operating", "investing", "financing"]:
                if c not in grouped.columns:
                    grouped[c] = 0.0
            grouped["net_cash_flow"] = grouped["operating"] + grouped["investing"] + grouped["financing"]
            out = grouped.sort_values("period")
            for c in ["operating", "investing", "financing", "net_cash_flow"]:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
            return out.to_dict(orient="records")
        except Exception:
            return []

    # Found periods for each statement
    found_is = []
    found_bs = _fetch_periods("historical_balance_sheet_line_items")
    found_cf = _fetch_periods("historical_cashflow_line_items")

    monthly_bs = _summarize_bs()
    monthly_cf = _summarize_cf()

    # IS existing logic below (keeps post-upsampling buckets)
    expected = expected_is
    try:
        expected = expected_is
    except Exception:
        expected = expected_is

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
        found_is = found

        view = hist_df.copy()
        view["period"] = pd.to_datetime(view["period_date"], errors="coerce").dt.to_period("M").astype(str)
        keep_cols = [c for c in ["period", "revenue", "cogs", "opex", "depreciation", "interest", "tax", "other_income", "gross_profit", "ebit"] if c in view.columns]
        view = view[keep_cols].copy()
        for c in keep_cols:
            if c != "period":
                view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0.0)
        monthly = view.sort_values("period").to_dict(orient="records")

    missing = sorted(set(expected_is) - set(found_is)) if expected_is else []
    extra = sorted(set(found_is) - set(expected_is)) if expected_is else []
    missing_bs = sorted(set(expected_bs) - set(found_bs)) if expected_bs else []
    extra_bs = sorted(set(found_bs) - set(expected_bs)) if expected_bs else []
    missing_cf = sorted(set(expected_cf) - set(found_cf)) if expected_cf else []
    extra_cf = sorted(set(found_cf) - set(expected_cf)) if expected_cf else []

    return {
        "scenario_id": scenario_id,
        "expected_periods_is": expected_is,
        "found_periods_is": found_is,
        "missing_periods_is": missing,
        "extra_periods_is": extra,
        "income_statement_monthly": monthly,
        "expected_periods_bs": expected_bs,
        "found_periods_bs": found_bs,
        "missing_periods_bs": missing_bs,
        "extra_periods_bs": extra_bs,
        "balance_sheet_monthly": monthly_bs,
        "expected_periods_cf": expected_cf,
        "found_periods_cf": found_cf,
        "missing_periods_cf": missing_cf,
        "extra_periods_cf": extra_cf,
        "cash_flow_monthly": monthly_cf,
        "used_sales_pattern": used_sales_pattern,
        "source_diag": diag,
    }

