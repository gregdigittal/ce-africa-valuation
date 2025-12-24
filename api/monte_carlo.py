from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def run_monte_carlo_simple(
    base_results: Dict[str, Any],
    *,
    iterations: int = 1000,
    fleet_cv: float = 0.10,
    prospect_cv: float = 0.30,
    cost_cv: float = 0.10,
    seed: int = 42,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    API-safe Monte Carlo simulation (matches existing UI logic).

    Uses lognormal multiplicative noise:
      - fleet revenue streams: fleet_cv
      - prospect/pipeline stream: prospect_cv
      - cogs/opex: cost_cv
    """
    mc_results: Dict[str, Any] = {
        "success": False,
        "iterations": int(iterations),
        "percentiles": {"revenue": {}, "gross_profit": {}, "ebit": {}},
        "distributions": {},
    }

    try:
        n_iterations = int(iterations)
        np.random.seed(int(seed))

        base_consumables = np.array(base_results["revenue"]["consumables"], dtype=float)
        base_refurb = np.array(base_results["revenue"]["refurb"], dtype=float)
        base_pipeline = np.array(base_results["revenue"]["pipeline"], dtype=float)
        base_cogs = np.array(base_results["costs"]["cogs"], dtype=float)
        base_opex = np.array(base_results["costs"]["opex"], dtype=float)

        n_months = len(base_consumables)

        all_revenue = np.zeros((n_iterations, n_months))
        all_gp = np.zeros((n_iterations, n_months))
        all_ebit = np.zeros((n_iterations, n_months))

        for i in range(n_iterations):
            if progress_callback and i % 200 == 0:
                progress_callback(i / max(n_iterations, 1), f"Monte Carlo {i}/{n_iterations}")

            consumables_sim = base_consumables * np.random.lognormal(0, float(fleet_cv), n_months)
            refurb_sim = base_refurb * np.random.lognormal(0, float(fleet_cv), n_months)
            pipeline_sim = base_pipeline * np.random.lognormal(0, float(prospect_cv), n_months)

            revenue_sim = consumables_sim + refurb_sim + pipeline_sim

            cogs_sim = base_cogs * np.random.lognormal(0, float(cost_cv), n_months)
            opex_sim = base_opex * np.random.lognormal(0, float(cost_cv), n_months)

            gp_sim = revenue_sim - cogs_sim
            ebit_sim = gp_sim - opex_sim

            all_revenue[i] = revenue_sim
            all_gp[i] = gp_sim
            all_ebit[i] = ebit_sim

        mc_results["percentiles"]["revenue"] = {
            "p10": np.percentile(all_revenue, 10, axis=0).tolist(),
            "p25": np.percentile(all_revenue, 25, axis=0).tolist(),
            "p50": np.percentile(all_revenue, 50, axis=0).tolist(),
            "p75": np.percentile(all_revenue, 75, axis=0).tolist(),
            "p90": np.percentile(all_revenue, 90, axis=0).tolist(),
            "mean": np.mean(all_revenue, axis=0).tolist(),
            "std": np.std(all_revenue, axis=0).tolist(),
        }

        mc_results["percentiles"]["gross_profit"] = {
            "p10": np.percentile(all_gp, 10, axis=0).tolist(),
            "p50": np.percentile(all_gp, 50, axis=0).tolist(),
            "p90": np.percentile(all_gp, 90, axis=0).tolist(),
        }

        mc_results["percentiles"]["ebit"] = {
            "p10": np.percentile(all_ebit, 10, axis=0).tolist(),
            "p50": np.percentile(all_ebit, 50, axis=0).tolist(),
            "p90": np.percentile(all_ebit, 90, axis=0).tolist(),
        }

        mc_results["distributions"] = {
            "total_revenue": np.sum(all_revenue, axis=1).tolist(),
            "total_gp": np.sum(all_gp, axis=1).tolist(),
            "total_ebit": np.sum(all_ebit, axis=1).tolist(),
        }

        mc_results["success"] = True
        return mc_results
    except Exception as e:
        mc_results["error"] = str(e)
        return mc_results

