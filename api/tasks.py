from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from api.forecast_data import load_forecast_data_api
from api.monte_carlo import run_monte_carlo_simple
from api.supabase_handler import SupabaseAPIHandler
from api.serialize import to_jsonable
from api.valuation import run_dcf_valuation
from forecast_engine import ForecastEngine


def _compact_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep job payloads small; Streamlit can request full snapshots later.
    """
    keep_keys = [
        "success",
        "error",
        "forecast_method_used",
        "assumptions_source",
        "ai_assumptions_used",
        "data_source",
        "manufacturing_included",
        "manufacturing_strategy",
        "summary",
    ]
    out: Dict[str, Any] = {k: results.get(k) for k in keep_keys if k in results}

    # Small top-line arrays (first 24 months) for quick inspection
    try:
        out["timeline"] = (results.get("timeline") or [])[:24]
        rev = results.get("revenue") or {}
        costs = results.get("costs") or {}
        profit = results.get("profit") or {}
        out["revenue_preview"] = {
            "total": (rev.get("total") or [])[:24],
            "consumables": (rev.get("consumables") or [])[:24],
            "refurb": (rev.get("refurb") or [])[:24],
            "pipeline": (rev.get("pipeline") or [])[:24],
        }
        out["costs_preview"] = {
            "cogs": (costs.get("cogs") or [])[:24],
            "opex": (costs.get("opex") or [])[:24],
        }
        out["profit_preview"] = {
            "gross": (profit.get("gross") or [])[:24],
            "ebit": (profit.get("ebit") or [])[:24],
        }
    except Exception:
        pass

    return out


def run_forecast_task(
    scenario_id: str,
    user_id: Optional[str] = None,
    options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Background job entrypoint.

    Runs the existing forecast engine in a non-Streamlit worker.
    """
    options = options or {}
    started = datetime.now(tz=timezone.utc).isoformat()

    db = SupabaseAPIHandler()
    data = load_forecast_data_api(db, scenario_id=scenario_id, user_id=user_id, options=options)

    engine = ForecastEngine()
    results = engine.run_forecast(data, manufacturing_scenario=None, progress_callback=None)

    # Optional Monte Carlo (API mode)
    mc_results: Optional[Dict[str, Any]] = None
    if bool(options.get("run_monte_carlo")):
        mc_results = run_monte_carlo_simple(
            results,
            iterations=int(options.get("mc_iterations", 1000)),
            fleet_cv=float(options.get("mc_fleet_cv", 0.10)),
            prospect_cv=float(options.get("mc_prospect_cv", 0.30)),
            cost_cv=float(options.get("mc_cost_cv", 0.10)),
            seed=int(options.get("mc_seed", 42)),
            progress_callback=None,
        )

    # Deterministic valuation (DCF)
    valuation_data: Optional[Dict[str, Any]] = None
    enterprise_value: Optional[float] = None
    if bool(options.get("run_valuation", True)):
        try:
            valuation_data_raw, ev = run_dcf_valuation(results, data.get("assumptions") or {}, net_debt=float(options.get("net_debt", 0.0) or 0.0))
            valuation_data = to_jsonable(valuation_data_raw)  # type: ignore[assignment]
            enterprise_value = float(ev)
        except Exception:
            valuation_data = {"error": "valuation_failed"}

    # Persist as snapshot (preferred transport for big payloads)
    snapshot_name = str(options.get("snapshot_name") or f"API Forecast {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    snapshot_type = str(options.get("snapshot_type") or "base")

    assumptions_data = to_jsonable(data.get("assumptions") or {})
    forecast_data = to_jsonable(results)
    monte_carlo_data = to_jsonable(mc_results) if mc_results is not None else None

    snapshot_id = db.insert_forecast_snapshot(
        scenario_id=scenario_id,
        user_id=user_id,
        snapshot_name=snapshot_name,
        snapshot_type=snapshot_type,
        assumptions_data=assumptions_data if isinstance(assumptions_data, dict) else {"value": assumptions_data},
        forecast_data=forecast_data if isinstance(forecast_data, dict) else {"value": forecast_data},
        monte_carlo_data=monte_carlo_data if isinstance(monte_carlo_data, dict) else ({"value": monte_carlo_data} if monte_carlo_data is not None else None),
        valuation_data=valuation_data if isinstance(valuation_data, dict) else ({"value": valuation_data} if valuation_data is not None else None),
        enterprise_value=enterprise_value,
    )

    return {
        "scenario_id": scenario_id,
        "user_id": user_id,
        "options": options,
        "started_at": started,
        "finished_at": datetime.now(tz=timezone.utc).isoformat(),
        "result": _compact_results(results),
        "mc": (
            {
                "success": mc_results.get("success"),
                "iterations": mc_results.get("iterations"),
                "error": mc_results.get("error"),
            }
            if isinstance(mc_results, dict)
            else None
        ),
        "valuation": valuation_data,
        "enterprise_value": enterprise_value,
        "snapshot_id": snapshot_id,
    }

