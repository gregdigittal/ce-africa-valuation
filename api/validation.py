from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import re


def validate_import_coverage(
    db,
    scenario_id: str,
    user_id: str,
    assumptions: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """
    Server-side historics gate.

    If `assumptions.import_period_settings` contains expected periods for IS/BS/CF line-item imports,
    verify those periods exist in the database for the scenario/user.

    Returns:
      (ok, messages)
    """
    msgs: List[str] = []
    assumptions = assumptions or {}
    ip = (assumptions or {}).get("import_period_settings") or {}
    if not isinstance(ip, dict) or not ip:
        return True, msgs

    def _expected_periods(key: str) -> List[str]:
        block = ip.get(key) or {}
        sel = block.get("selected_periods")
        if isinstance(sel, list) and sel:
            return [str(x).strip() for x in sel if str(x).strip()]
        return []

    def _month_starts(labels: List[str]) -> List[str]:
        out: List[str] = []
        for lbl in labels or []:
            s = str(lbl).strip()
            if re.match(r"^\d{4}-\d{2}$", s):
                out.append(f"{s}-01")
        return sorted(set(out))

    def _fetch_periods(table: str, month_starts: List[str]) -> List[str]:
        if not month_starts:
            return []
        resp = (
            db.client.table(table)
            .select("period_date")
            .eq("scenario_id", scenario_id)
            .eq("user_id", user_id)
            .in_("period_date", month_starts)
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

    def _revenue_nonzero(table: str, month_starts: List[str]) -> bool:
        if not month_starts:
            return True
        resp = (
            db.client.table(table)
            .select("category,line_item_name,amount")
            .eq("scenario_id", scenario_id)
            .eq("user_id", user_id)
            .in_("period_date", month_starts)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return False
        df = pd.DataFrame(rows)
        if df.empty:
            return False
        df["category"] = df.get("category", "").fillna("").astype(str).str.lower()
        df["line_item_name"] = df.get("line_item_name", "").fillna("").astype(str).str.lower()
        df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce").fillna(0.0)
        is_rev = df["category"].str.contains("revenue|sales|income|turnover", na=False) | df["line_item_name"].str.contains(
            "revenue|sales|income|turnover", na=False
        )
        return float(df.loc[is_rev, "amount"].abs().sum()) != 0.0

    checks = [
        ("historical_income_statement_line_items", "historical_income_statement_line_items", "Income Statement"),
        ("historical_balance_sheet_line_items", "historical_balance_sheet_line_items", "Balance Sheet"),
        ("historical_cashflow_line_items", "historical_cashflow_line_items", "Cash Flow"),
    ]

    ok = True
    for ip_key, table, label in checks:
        expected = _expected_periods(ip_key)
        if not expected:
            continue
        ms = _month_starts(expected)
        found = set(_fetch_periods(table, ms))
        missing = sorted(set(expected) - found)
        if missing:
            ok = False
            msgs.append(
                f"{label}: missing imported periods in DB: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}"
            )
        if table == "historical_income_statement_line_items" and not _revenue_nonzero(table, ms):
            ok = False
            msgs.append(
                "Income Statement: revenue totals appear to be zero across expected months (check category/line item labels and amount parsing)."
            )

    return ok, msgs

