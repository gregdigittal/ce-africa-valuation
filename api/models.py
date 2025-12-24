from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ForecastRunRequest(BaseModel):
    scenario_id: str = Field(..., description="Scenario UUID")
    user_id: Optional[str] = Field(None, description="User UUID (optional for dev)")
    options: Dict[str, Any] = Field(default_factory=dict, description="Run options (horizon, toggles, etc.)")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    enqueued_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None

