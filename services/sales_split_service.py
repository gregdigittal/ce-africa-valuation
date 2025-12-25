"""
Sales Split Service
===================
Derive a Wear Parts vs Refurbishment/Service revenue split from historical sales orders.

This is used to:
1) Apportion historical Income Statement total revenue into Wear vs Refurb buckets
2) Provide better baseline mix assumptions for forecasting (especially Hybrid)

Design goals:
- Deterministic (no external LLM required)
- Fast enough for 1k+ sales lines
- Stores results into assumptions JSONB for reuse (avoids repeated recomputation)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Supabase helpers (avoid silent 1000-row truncation)
try:
    from supabase_pagination import fetch_all_rows
except Exception:
    fetch_all_rows = None

WEAR_KEYWORDS = [
    # common wear/parts terms
    "wear", "wear part", "wear parts", "wearpart", "wearparts",
    "liner", "liners", "mantle", "concave", "segment", "segments",
    "chocky", "tile", "tiles", "plate", "plates", "grate", "grates",
    "blow bar", "blowbar", "hammer", "hammers",
    "spare", "spares", "spare part", "spare parts", "part", "parts",
    "consumable", "consumables",
    # common hardware often present on parts invoices
    "bolt", "bolts", "nut", "nuts", "washer", "washers",
    "screw", "screws", "clamp", "clamps", "clamping",
    "deflector", "strip", "spring", "seal", "bearing",
]

SERVICE_KEYWORDS = [
    "refurb", "refurbish", "refurbishment",
    "service", "services",
    "repair", "repairs",
    "rebuild", "rebuilds",
    "overhaul", "overhauls",
    "maintenance",
    "labour", "labor",
    "installation", "commissioning",
    "inspection",
    "transport", "freight", "delivery", "shipping", "courier",
]


@dataclass(frozen=True)
class SplitResult:
    split_data: Dict[str, Any]
    monthly: pd.DataFrame


def _classify_text(text: str) -> Tuple[str, float, Dict[str, int]]:
    """
    Classify a sales line into wear/service/unknown using keyword scoring.
    Returns (label, confidence, diagnostics).
    """
    t = (text or "").strip().lower()
    if not t:
        return "unknown", 0.0, {"wear_score": 0, "service_score": 0}

    wear_score = sum(1 for k in WEAR_KEYWORDS if k in t)
    service_score = sum(1 for k in SERVICE_KEYWORDS if k in t)

    if wear_score == 0 and service_score == 0:
        return "unknown", 0.0, {"wear_score": 0, "service_score": 0}

    if wear_score > service_score:
        conf = float((wear_score - service_score) / max(wear_score + service_score, 1))
        return "wear", conf, {"wear_score": wear_score, "service_score": service_score}

    if service_score > wear_score:
        conf = float((service_score - wear_score) / max(wear_score + service_score, 1))
        return "service", conf, {"wear_score": wear_score, "service_score": service_score}

    # tie: ambiguous
    return "unknown", 0.1, {"wear_score": wear_score, "service_score": service_score}


def load_sales_orders_df(db, scenario_id: str, user_id: str) -> pd.DataFrame:
    """Load sales_orders rows for a scenario/user as a DataFrame."""
    if not hasattr(db, "client"):
        return pd.DataFrame()
    try:
        q = (
            db.client.table("sales_orders")
            .select("order_date,total_amount,item_code,description,customer_name,customer_code,order_number")
            .eq("scenario_id", scenario_id)
            .eq("user_id", user_id)
        )
        if fetch_all_rows:
            rows = fetch_all_rows(q, order_by="id")
        else:
            resp = q.execute()
            rows = resp.data if resp and getattr(resp, "data", None) else []
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def compute_sales_split_from_orders(
    sales_orders: pd.DataFrame,
    *,
    unknown_allocation: str = "pro_rata",
    default_wear_share: float = 0.70,
) -> SplitResult:
    """
    Compute a monthly wear vs service split from sales order lines.

    unknown_allocation:
      - "pro_rata": allocate unknown in proportion to classified wear/service
      - "wear": allocate unknown to wear
      - "service": allocate unknown to service
      - "ignore": exclude unknown from split (share based on classified only)
    """
    df = sales_orders.copy()
    if df.empty:
        empty_split = {
            "method": "heuristic_keywords_v1",
            "generated_at": datetime.utcnow().isoformat(),
            "error": "no_sales_orders",
        }
        return SplitResult(split_data=empty_split, monthly=pd.DataFrame())

    df["order_date"] = pd.to_datetime(df.get("order_date"), errors="coerce")
    df = df.dropna(subset=["order_date"])
    if df.empty:
        empty_split = {
            "method": "heuristic_keywords_v1",
            "generated_at": datetime.utcnow().isoformat(),
            "error": "no_valid_dates",
        }
        return SplitResult(split_data=empty_split, monthly=pd.DataFrame())

    df["total_amount"] = pd.to_numeric(df.get("total_amount"), errors="coerce").fillna(0.0)
    df["period"] = df["order_date"].dt.to_period("M").dt.to_timestamp()

    # Combine text fields for classification
    txt = (
        df.get("item_code", "").fillna("").astype(str)
        + " "
        + df.get("description", "").fillna("").astype(str)
    ).str.strip()

    labels = []
    confs = []
    wear_scores = []
    svc_scores = []
    for t in txt.tolist():
        lab, conf, diag = _classify_text(t)
        labels.append(lab)
        confs.append(conf)
        wear_scores.append(int(diag.get("wear_score", 0)))
        svc_scores.append(int(diag.get("service_score", 0)))

    df["split_label"] = labels
    df["split_conf"] = confs
    df["wear_score"] = wear_scores
    df["service_score"] = svc_scores

    # Aggregate to period (avoid groupby.apply warnings + faster)
    df["wear_amt"] = np.where(df["split_label"] == "wear", df["total_amount"], 0.0)
    df["service_amt"] = np.where(df["split_label"] == "service", df["total_amount"], 0.0)
    df["unknown_amt"] = np.where(df["split_label"] == "unknown", df["total_amount"], 0.0)
    df["wear_row"] = (df["split_label"] == "wear").astype(int)
    df["service_row"] = (df["split_label"] == "service").astype(int)
    df["unknown_row"] = (df["split_label"] == "unknown").astype(int)

    agg = (
        df.groupby("period", dropna=False)
        .agg(
            total=("total_amount", "sum"),
            wear=("wear_amt", "sum"),
            service=("service_amt", "sum"),
            unknown=("unknown_amt", "sum"),
            rows=("split_label", "size"),
            wear_rows=("wear_row", "sum"),
            service_rows=("service_row", "sum"),
            unknown_rows=("unknown_row", "sum"),
        )
        .reset_index()
        .sort_values("period")
    )

    # Overall ratio from classified totals
    total_wear = float(agg["wear"].sum())
    total_service = float(agg["service"].sum())
    total_unknown = float(agg["unknown"].sum())
    classified_total = total_wear + total_service
    if classified_total > 0:
        overall_wear_share = total_wear / classified_total
    else:
        overall_wear_share = float(np.clip(default_wear_share, 0.0, 1.0))

    # Allocation helper
    def _alloc_unknown(wear_amt: float, svc_amt: float, unknown_amt: float) -> Tuple[float, float]:
        if unknown_amt == 0:
            return wear_amt, svc_amt
        if unknown_allocation == "wear":
            return wear_amt + unknown_amt, svc_amt
        if unknown_allocation == "service":
            return wear_amt, svc_amt + unknown_amt
        if unknown_allocation == "ignore":
            return wear_amt, svc_amt
        # pro_rata
        denom = wear_amt + svc_amt
        if denom <= 0:
            return wear_amt + unknown_amt * overall_wear_share, svc_amt + unknown_amt * (1 - overall_wear_share)
        wear_share = wear_amt / denom
        return wear_amt + unknown_amt * wear_share, svc_amt + unknown_amt * (1 - wear_share)

    wear_final = []
    svc_final = []
    wear_share_final = []
    svc_share_final = []
    for _, r in agg.iterrows():
        w, s = _alloc_unknown(float(r["wear"]), float(r["service"]), float(r["unknown"]))
        tot = float(r["total"]) if float(r["total"]) != 0 else 0.0
        wear_final.append(w)
        svc_final.append(s)
        if tot != 0:
            wear_share_final.append(float(np.clip(w / tot, 0.0, 1.0)))
            svc_share_final.append(float(np.clip(s / tot, 0.0, 1.0)))
        else:
            wear_share_final.append(float(np.clip(overall_wear_share, 0.0, 1.0)))
            svc_share_final.append(float(np.clip(1 - overall_wear_share, 0.0, 1.0)))

    agg["wear_final"] = wear_final
    agg["service_final"] = svc_final
    agg["wear_share"] = wear_share_final
    agg["service_share"] = svc_share_final

    # Serialize to dict
    by_period: Dict[str, Any] = {}
    for _, r in agg.iterrows():
        key = pd.to_datetime(r["period"]).strftime("%Y-%m")
        by_period[key] = {
            "wear_share": float(r["wear_share"]),
            "service_share": float(r["service_share"]),
            "total": float(r["total"]),
            "wear": float(r["wear_final"]),
            "service": float(r["service_final"]),
            "unknown": float(r["unknown"]),
            "rows": int(r["rows"]),
        }

    split_data: Dict[str, Any] = {
        "method": "heuristic_keywords_v1",
        "generated_at": datetime.utcnow().isoformat(),
        "unknown_allocation": unknown_allocation,
        "overall": {
            "wear_share": float(overall_wear_share),
            "service_share": float(1 - overall_wear_share),
            "wear": float(total_wear),
            "service": float(total_service),
            "unknown": float(total_unknown),
            "classified_total": float(classified_total),
        },
        "by_period": by_period,
        "classification_summary": {
            "rows": int(len(df)),
            "wear_rows": int((df["split_label"] == "wear").sum()),
            "service_rows": int((df["split_label"] == "service").sum()),
            "unknown_rows": int((df["split_label"] == "unknown").sum()),
        },
    }

    return SplitResult(split_data=split_data, monthly=agg)


def save_sales_split_to_assumptions(db, scenario_id: str, user_id: str, split_data: Dict[str, Any]) -> bool:
    """Persist split into assumptions.data.historical_revenue_split."""
    try:
        existing = {}
        if hasattr(db, "get_scenario_assumptions"):
            existing = db.get_scenario_assumptions(scenario_id, user_id) or {}
        existing["historical_revenue_split"] = split_data
        if hasattr(db, "update_assumptions"):
            return bool(db.update_assumptions(scenario_id, user_id, existing))
        return False
    except Exception:
        return False


