from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Request

from api.auth import parse_bearer_token, verify_supabase_jwt
from api.jobs import fetch_job, job_to_status
from api.models import ForecastRunRequest, JobStatusResponse, SnapshotGetResponse, SnapshotListItem
from api.queue import get_queue
from api.supabase_handler import SupabaseAPIHandler
from api.tasks import run_forecast_task


app = FastAPI(title="CE Africa Valuation API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/v1/forecasts/run")
def enqueue_forecast(req: ForecastRunRequest) -> dict:
    q = get_queue("default")
    job = q.enqueue(
        run_forecast_task,
        req.scenario_id,
        req.user_id,
        req.options,
        job_timeout=60 * 30,  # 30 minutes for now
        result_ttl=60 * 60,   # keep 1 hour
    )
    return {"job_id": job.id}


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    try:
        job = fetch_job(job_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_to_status(job)


@app.get("/v1/scenarios/{scenario_id}/snapshots", response_model=list[SnapshotListItem])
def list_snapshots(
    scenario_id: str,
    request: Request,
    user_id: str | None = Query(None, description="User UUID (fallback when no Authorization header)"),
    limit: int = Query(50, ge=1, le=200),
):
    effective_user_id = user_id
    try:
        token = parse_bearer_token(request.headers.get("Authorization"))
        if token:
            effective_user_id = verify_supabase_jwt(token).user_id
    except Exception:
        # If JWT is present but invalid, reject
        if request.headers.get("Authorization"):
            raise HTTPException(status_code=401, detail="Invalid Authorization token")

    if not effective_user_id:
        raise HTTPException(status_code=401, detail="Missing user_id (provide Authorization header or user_id query param)")

    db = SupabaseAPIHandler()
    return db.list_snapshots(scenario_id=scenario_id, user_id=effective_user_id, limit=limit)


@app.get("/v1/snapshots/{snapshot_id}", response_model=SnapshotGetResponse)
def get_snapshot(
    snapshot_id: str,
    request: Request,
    user_id: str | None = Query(None, description="User UUID (fallback when no Authorization header)"),
):
    effective_user_id = user_id
    try:
        token = parse_bearer_token(request.headers.get("Authorization"))
        if token:
            effective_user_id = verify_supabase_jwt(token).user_id
    except Exception:
        if request.headers.get("Authorization"):
            raise HTTPException(status_code=401, detail="Invalid Authorization token")

    if not effective_user_id:
        raise HTTPException(status_code=401, detail="Missing user_id (provide Authorization header or user_id query param)")

    db = SupabaseAPIHandler()
    snap = db.get_snapshot(snapshot_id=snapshot_id, user_id=effective_user_id)
    if not snap:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return snap

