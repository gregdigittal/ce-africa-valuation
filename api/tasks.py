from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


def run_forecast_task(scenario_id: str, user_id: str | None = None, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Background job entrypoint.

    MVP: return a stub payload. Next sprint: call the real forecast engine and
    persist results/snapshots in Supabase.
    """
    options = options or {}
    started = datetime.now(tz=timezone.utc).isoformat()
    # TODO: integrate with existing forecast logic (forecast_engine.py / components/forecast_section.py)
    return {
        "scenario_id": scenario_id,
        "user_id": user_id,
        "options": options,
        "started_at": started,
        "status": "ok",
        "message": "Forecast job executed (stub). Next sprint will call engine and return real results.",
    }

