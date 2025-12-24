## API service (FastAPI + RQ)

This folder scaffolds the migration of forecasting/modeling into a Python API.

### Local dev (minimal)

### Environment variables

Set these in your shell before starting the API/worker:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` (preferred for server-side jobs)
  - Alternatively: `SUPABASE_ANON_KEY` or `SUPABASE_KEY`
- `REDIS_URL` (optional, defaults to `redis://localhost:6379/0`)

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
- `POST /v1/forecasts/run`: enqueue a forecast job (runs existing engine in worker)
- `GET /v1/jobs/{job_id}`: job status/result

### Quick test (curl)

```bash
curl -s -X POST "http://localhost:8000/v1/forecasts/run" \
  -H "Content-Type: application/json" \
  -d '{"scenario_id":"<scenario-uuid>","user_id":"<user-uuid>","options":{"forecast_method":"pipeline","forecast_duration_months":60}}'
```

Then:

```bash
curl -s "http://localhost:8000/v1/jobs/<job-id>"
```

