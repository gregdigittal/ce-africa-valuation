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


class SnapshotListItem(BaseModel):
    id: str
    snapshot_name: Optional[str] = None
    snapshot_date: Optional[str] = None
    snapshot_type: Optional[str] = None
    created_at: Optional[str] = None
    total_revenue_forecast: Optional[float] = None
    total_gross_profit_forecast: Optional[float] = None
    enterprise_value: Optional[float] = None
    is_locked: Optional[bool] = None


class SnapshotGetResponse(BaseModel):
    id: str
    scenario_id: str
    user_id: str
    snapshot_name: Optional[str] = None
    snapshot_date: Optional[str] = None
    snapshot_type: Optional[str] = None
    created_at: Optional[str] = None
    assumptions_data: Optional[Dict[str, Any]] = None
    forecast_data: Optional[Dict[str, Any]] = None
    monte_carlo_data: Optional[Dict[str, Any]] = None
    valuation_data: Optional[Dict[str, Any]] = None
    enterprise_value: Optional[float] = None
    summary_stats: Optional[Dict[str, Any]] = None
    total_revenue_forecast: Optional[float] = None
    total_gross_profit_forecast: Optional[float] = None
    is_locked: Optional[bool] = None
    notes: Optional[str] = None

