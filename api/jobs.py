from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from rq.job import Job

from api.queue import get_redis


def job_to_status(job: Job) -> Dict[str, Any]:
    def _iso(dt: Optional[datetime]) -> Optional[str]:
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    # Job.get_status() returns string statuses: queued/started/finished/failed/deferred/scheduled
    status = job.get_status()
    payload: Dict[str, Any] = {
        "job_id": job.id,
        "status": status,
        "enqueued_at": _iso(job.enqueued_at),
        "started_at": _iso(job.started_at),
        "ended_at": _iso(job.ended_at),
        "result": None,
        "error": None,
    }

    if status == "failed":
        payload["error"] = str(job.exc_info or "Job failed")
    elif status == "finished":
        payload["result"] = job.result

    return payload


def fetch_job(job_id: str) -> Job:
    return Job.fetch(job_id, connection=get_redis())

