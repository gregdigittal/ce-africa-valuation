from __future__ import annotations

import os
from dataclasses import dataclass
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

