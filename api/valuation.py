from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ValuationResult:
    enterprise_value: float
    equity_value: float
    pv_discrete_cf: float
    pv_terminal_value: float
    terminal_value_perpetuity: float
    tv_as_pct_ev: float
    implied_ev_ebitda: float
    implied_ev_revenue: float
    dcf_table: pd.DataFrame


class ValuationEngine:
    """
    API-safe DCF valuation engine.

    Note: This intentionally avoids Streamlit/UI dependencies.
    """

    def __init__(self, assumptions: Dict[str, Any]):
        self.assumptions = assumptions or {}
        self.wacc = self._get_rate("wacc", 0.12)
        self.tax_rate = self._get_rate("tax_rate", 0.27)
        self.g_rate = self._get_rate("terminal_growth_rate", 0.03)
        self.exit_multiple = float(self.assumptions.get("exit_multiple_ebitda", 5.0) or 5.0)
        self.capex_pct = self._get_rate("capex_pct_revenue", 0.05)
        self.depreciation_pct = self._get_rate("depreciation_pct_revenue", 0.03)
        self.nwc_pct = self._get_rate("nwc_pct_revenue", 0.10)

    def _get_rate(self, key: str, default: float) -> float:
        value = self.assumptions.get(key, default)
        if value is None:
            return default
        value_f = float(value)
        return value_f / 100 if value_f > 1 else value_f

    def calculate_valuation(self, forecast_df: pd.DataFrame, net_debt: float = 0.0) -> ValuationResult:
        if forecast_df is None or forecast_df.empty:
            empty = pd.DataFrame()
            return ValuationResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, empty)

        df = forecast_df.copy()

        # Standardize revenue column
        if "total_revenue" in df.columns and "revenue_total" not in df.columns:
            df["revenue_total"] = df["total_revenue"]
        if "revenue" in df.columns and "revenue_total" not in df.columns:
            df["revenue_total"] = df["revenue"]

        if "revenue_total" not in df.columns or float(df["revenue_total"].sum()) == 0.0:
            empty = pd.DataFrame()
            return ValuationResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, empty)

        # EBITDA
        if "ebitda" not in df.columns:
            if "total_gross_profit" in df.columns and "total_opex" in df.columns:
                df["ebitda"] = df["total_gross_profit"] - df["total_opex"]
            elif "gross_profit" in df.columns and "opex" in df.columns:
                df["ebitda"] = df["gross_profit"] - df["opex"]
            else:
                df["ebitda"] = df["revenue_total"] * 0.15

        # D&A + CapEx
        if "depreciation" not in df.columns:
            df["depreciation"] = df["revenue_total"] * self.depreciation_pct
        if "capex" not in df.columns:
            df["capex"] = df["revenue_total"] * self.capex_pct

        # EBIT + NOPAT
        df["ebit"] = df["ebitda"] - df["depreciation"]
        df["tax_expense"] = np.where(df["ebit"] > 0, df["ebit"] * self.tax_rate, 0.0)
        df["nopat"] = df["ebit"] - df["tax_expense"]

        # Working capital
        df["nwc_balance"] = df["revenue_total"] * self.nwc_pct
        df["delta_nwc"] = df["nwc_balance"].diff().fillna(df["nwc_balance"].iloc[0] * 0.1)

        # UFCF
        df["ufcf"] = df["nopat"] + df["depreciation"] - df["capex"] - df["delta_nwc"]

        # Discount factors (monthly)
        df["period_index"] = np.arange(1, len(df) + 1)
        df["discount_factor"] = 1.0 / ((1.0 + self.wacc) ** (df["period_index"] / 12.0))
        df["pv_ufcf"] = df["ufcf"] * df["discount_factor"]
        pv_discrete = float(df["pv_ufcf"].sum())

        # Terminal value (annualize last 12 months or fewer)
        n_months = min(12, len(df))
        last_period = df.tail(n_months)
        final_year_ufcf = float(last_period["ufcf"].sum()) * (12.0 / n_months)
        final_year_ebitda = float(last_period["ebitda"].sum()) * (12.0 / n_months)
        final_discount_factor = float(df["discount_factor"].iloc[-1])

        if self.wacc <= self.g_rate:
            tv_perpetuity = 0.0
        else:
            tv_perpetuity = (final_year_ufcf * (1.0 + self.g_rate)) / (self.wacc - self.g_rate)
        pv_tv = float(tv_perpetuity * final_discount_factor)

        enterprise_value = pv_discrete + pv_tv
        equity_value = enterprise_value - float(net_debt or 0.0)

        total_revenue = float(df["revenue_total"].sum())
        total_ebitda = float(df["ebitda"].sum())
        years = max(len(df) / 12.0, 1e-9)
        annual_ebitda = total_ebitda / years
        annual_revenue = total_revenue / years

        tv_pct = (pv_tv / enterprise_value * 100.0) if enterprise_value > 0 else 0.0
        ev_ebitda = (enterprise_value / annual_ebitda) if annual_ebitda > 0 else 0.0
        ev_revenue = (enterprise_value / annual_revenue) if annual_revenue > 0 else 0.0

        return ValuationResult(
            enterprise_value=float(enterprise_value),
            equity_value=float(equity_value),
            pv_discrete_cf=float(pv_discrete),
            pv_terminal_value=float(pv_tv),
            terminal_value_perpetuity=float(tv_perpetuity),
            tv_as_pct_ev=float(tv_pct),
            implied_ev_ebitda=float(ev_ebitda),
            implied_ev_revenue=float(ev_revenue),
            dcf_table=df,
        )


def build_forecast_df_from_results(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert the engine's dict output into a DataFrame compatible with ValuationEngine.
    """
    timeline = results.get("timeline_dates") or []
    if not timeline:
        # fallback: use timeline (YYYY-MM) as month start
        timeline = results.get("timeline") or []
        timeline = [f"{m}-01" for m in timeline]

    revenue = results.get("revenue") or {}
    costs = results.get("costs") or {}
    profit = results.get("profit") or {}

    df = pd.DataFrame(
        {
            "period_date": pd.to_datetime(timeline, errors="coerce"),
            "total_revenue": pd.to_numeric(revenue.get("total", []), errors="coerce"),
            "total_cogs": pd.to_numeric(costs.get("cogs", []), errors="coerce"),
            "total_opex": pd.to_numeric(costs.get("opex", []), errors="coerce"),
            "total_gross_profit": pd.to_numeric(profit.get("gross", []), errors="coerce"),
            "ebit": pd.to_numeric(profit.get("ebit", []), errors="coerce"),
        }
    )
    df = df.dropna(subset=["period_date"]).reset_index(drop=True)
    return df


def run_dcf_valuation(
    results: Dict[str, Any],
    assumptions: Dict[str, Any],
    net_debt: float = 0.0,
) -> Tuple[Dict[str, Any], float]:
    """
    Returns (valuation_data, enterprise_value).
    """
    forecast_df = build_forecast_df_from_results(results)
    engine = ValuationEngine(assumptions)
    out = engine.calculate_valuation(forecast_df, net_debt=net_debt)

    valuation_data: Dict[str, Any] = {
        "enterprise_value": out.enterprise_value,
        "equity_value": out.equity_value,
        "pv_discrete_cf": out.pv_discrete_cf,
        "pv_terminal_value": out.pv_terminal_value,
        "terminal_value_perpetuity": out.terminal_value_perpetuity,
        "tv_as_pct_ev": out.tv_as_pct_ev,
        "implied_ev_ebitda": out.implied_ev_ebitda,
        "implied_ev_revenue": out.implied_ev_revenue,
        # Keep the DCF table small; full detail can be recomputed.
        "dcf_table": out.dcf_table.tail(24).to_dict("records"),
    }
    return valuation_data, float(out.enterprise_value)

