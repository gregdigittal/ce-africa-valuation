## API service (FastAPI + RQ)

This folder scaffolds the migration of forecasting/modeling into a Python API.

### Local dev (minimal)

1) Start Redis (required for background jobs)

If you have Docker:

```bash
docker run --rm -p 6379:6379 redis:7
```

2) Start the API

```bash
cd "/Users/gregmorris/Development Projects/CE Africa/ce-africa-valuation"
python3 -m uvicorn api.main:app --reload --port 8000
```

3) Start the worker

```bash
cd "/Users/gregmorris/Development Projects/CE Africa/ce-africa-valuation"
python3 -m api.worker
```

### Endpoints

- `GET /health`: liveness check
- `POST /v1/forecasts/run`: enqueue a forecast job (currently a stub)
- `GET /v1/jobs/{job_id}`: job status/result

