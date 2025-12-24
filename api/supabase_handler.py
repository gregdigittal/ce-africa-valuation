from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional

from supabase import Client, create_client


@dataclass(frozen=True)
class SupabaseEnv:
    url: str
    key: str


def _get_supabase_env() -> SupabaseEnv:
    """
    API-safe Supabase config loader (no Streamlit secrets).

    Expected env vars:
    - SUPABASE_URL
    - SUPABASE_SERVICE_ROLE_KEY (preferred for server-side jobs) OR SUPABASE_ANON_KEY / SUPABASE_KEY
    """
    url = os.getenv("SUPABASE_URL", "").strip()
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
        or os.getenv("SUPABASE_KEY", "").strip()
    )
    if not url or not key:
        raise RuntimeError(
            "Missing Supabase env vars. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "(or SUPABASE_ANON_KEY / SUPABASE_KEY)."
        )
    return SupabaseEnv(url=url, key=key)


class SupabaseAPIHandler:
    """
    Minimal DB handler for the API/worker.

    Provides `.client` (supabase-py Client) and a few helper methods used by the engine.
    """

    def __init__(self):
        env = _get_supabase_env()
        self.client: Client = create_client(env.url, env.key)

    def get_scenario_assumptions(self, scenario_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            resp = self.client.table("assumptions").select("data").eq("scenario_id", scenario_id).execute()
            if resp.data:
                return resp.data[0].get("data", {}) or {}
            return {}
        except Exception:
            return {}

    def get_scenario_user_id(self, scenario_id: str) -> Optional[str]:
        try:
            resp = self.client.table("scenarios").select("user_id").eq("id", scenario_id).limit(1).execute()
            if resp.data and resp.data[0].get("user_id"):
                return str(resp.data[0]["user_id"])
            return None
        except Exception:
            return None

    def insert_forecast_snapshot(
        self,
        *,
        scenario_id: str,
        user_id: Optional[str],
        snapshot_name: str,
        snapshot_type: str,
        assumptions_data: Dict[str, Any],
        forecast_data: Dict[str, Any],
        monte_carlo_data: Optional[Dict[str, Any]] = None,
        snapshot_date: Optional[date] = None,
    ) -> Optional[str]:
        """
        Insert a minimal snapshot record and return its ID.
        """
        try:
            effective_user_id = user_id or self.get_scenario_user_id(scenario_id)
            if not effective_user_id:
                # Can't write to forecast_snapshots without user_id (NOT NULL)
                return None

            payload: Dict[str, Any] = {
                "user_id": effective_user_id,
                "scenario_id": scenario_id,
                "snapshot_name": snapshot_name,
                "snapshot_type": snapshot_type,
                "assumptions_data": assumptions_data,
                "forecast_data": forecast_data,
            }
            if monte_carlo_data is not None:
                payload["monte_carlo_data"] = monte_carlo_data
            if snapshot_date is not None:
                payload["snapshot_date"] = snapshot_date.isoformat()

            # Return inserted row id
            resp = self.client.table("forecast_snapshots").insert(payload).execute()
            if resp.data and resp.data[0].get("id"):
                return str(resp.data[0]["id"])
            return None
        except Exception:
            return None

