# AI Assumptions Refactor Recommendation (Online)
**Date:** 2025-12-25  
**Scope:** Streamlit UI + persistence + forecast consumption (Installed Base model)  
**Goal:** Make it obvious what is **required** vs **optional**, reduce duplication, and make the “applied assumptions” unambiguous.

---

## 1) What’s wrong today (user-facing)

### 1.1 “Required vs optional” is inconsistent
- `app_refactored.py` treats AI Assumptions as **required before Forecast** (workflow stages + gating), but the user manual describes it as **optional**.
- There is a “Skip (Use Defaults)” bypass, which makes it *functionally optional* but still *presented as required*.

### 1.2 Multiple configuration surfaces with unclear precedence
The AI Assumptions section currently mixes:
- **Run Analysis** (auto-saves analysis results)
- **Configure Assumptions** (Unified line-item config; also sets `assumptions_saved=True`)
- **Legacy** tabs (Trend Forecast + Distributions)
- **Manufacturing** assumptions
- **Save & Apply** (explicit save for AI assumptions)

Users can save *something* in multiple places, but it’s unclear:
- which “save” unblocks Forecast
- which values are actually used during forecasting
- which parts are optional (e.g., distributions vs trend config vs manufacturing suggestions)

### 1.3 “Save” semantics are confusing
The engine auto-saves analysis results (`ai_assumptions_auto_saved=True`) but still requires an explicit “Save Assumptions” to mark `assumptions_saved=True`. This creates a state where:
- analysis is complete
- results exist
- but the app still blocks forecast unless the user presses the right save button (or uses Skip)

---

## 2) What’s wrong today (technical)

### 2.1 `components/ai_assumptions_engine.py` is monolithic
It contains:
- UI rendering
- historical data loading
- analytics and distribution fitting
- DB persistence
- workflow state side-effects
- manufacturing recommendations

This makes it hard to reason about correctness, hard to test, and easy to introduce regressions.

### 2.2 Business logic is duplicated across modules
Examples:
- “bucketizing” / classification of historics exists in multiple files (`setup_wizard.py`, `forecast_section.py`, `ai_assumptions_engine.py`) and can drift.
- “effective assumption” logic exists in multiple places (`forecast_engine.py` + AI integration).

### 2.3 Service layer imports UI modules
`services/assumptions_service.py` imports functions and classes directly from `components/ai_assumptions_engine.py`. This inverts layering and prevents clean reuse by API/worker code.

---

## 3) Target user workflow (clear required vs optional)

### 3.1 Required to run a forecast (minimum viable)
Required inputs should be limited to:
1. **Setup complete**: scenario selected + installed base/fleet present
2. **Base assumptions saved** (manual): WACC, tax, inflation, base margins (or defaults accepted)
3. **Forecast method chosen**:
   - Pipeline (Installed Base + Prospects) OR
   - Trend/Line-item forecasting

**Important:** Pipeline forecasting should not require AI analysis.

### 3.2 Optional add-ons (explicitly labeled)
Optional features should be opt-in and clearly describe impact:
- **AI Suggestions (Optional)**: generate recommended values + distributions from historicals
- **Monte Carlo (Optional)**: uses distributions (AI-derived or manual CV defaults)
- **Trend/Line-item Forecasting (Optional)**: configure line-item trends/correlations; only required if user selects Trend method
- **Manufacturing Strategy (Optional)**: only impacts forecast if enabled

### 3.3 One “Applied Assumptions” view
Add a single summary panel that always answers:
- What’s the active forecast method?
- Which assumptions are currently applied?
- For each assumption: **value + source** (Manual / AI / Default) + optional distribution
- Last updated timestamp
- Warnings for missing prerequisites (e.g., “Trend method selected but no unified config saved”)

---

## 4) Target persistence model (single source of truth)

Use `assumptions.data` JSONB, but enforce separation of concerns:

- **`manual_inputs`**: the base manual inputs (today’s top-level keys can remain for backward compatibility, but new code should treat this as the canonical block)
- **`ai_analysis`**: auto-saved analysis outputs (raw stats, fit scores, candidate distributions)
- **`assumptions_active`**: the committed set used by forecasting
- **`unified_line_item_config`**: only if Trend/Line-item method is used
- **`forecast_method`**: `pipeline` or `trend`
- **`monte_carlo`**: enabled flag + config (iterations, etc.)

This eliminates ambiguity between “analysis exists” and “assumptions applied”.

---

## 5) Refactor architecture (recommended module split)

### 5.1 Create a small AI Assumptions package
Move logic out of `components/ai_assumptions_engine.py` into:

- `components/ai_assumptions/models.py`
  - `Assumption`, `AssumptionsSet`, `DistributionParams`, serialization helpers

- `components/ai_assumptions/data_sources.py`
  - Load historical summary + detailed line items
  - Load sales seasonality inputs (e.g., `granular_sales_history`, `sales_orders`)
  - No Streamlit calls

- `components/ai_assumptions/analysis.py`
  - Historical stats, distribution fitting, diagnostics (pure Python)

- `components/ai_assumptions/persistence.py`
  - Read/write to `assumptions.data` with explicit schema versioning

- `components/ai_assumptions/ui.py`
  - Streamlit rendering only (tabs, forms, preview tables)

### 5.2 Service layer becomes the canonical API
Update `services/assumptions_service.py` to depend on `components/ai_assumptions/*` modules (not UI).
This enables:
- API/worker reuse
- unit testing of analysis + persistence

---

## 6) Refactor plan (incremental, low-risk)

### Phase 1 (clarity + correctness)
1. **Unify gating logic** for Forecast:
   - Allow Forecast when pipeline method + manual assumptions exist
   - Require unified config only when trend method selected
   - Require AI assumptions only when Monte Carlo is enabled *and* no distributions are configured
2. **Rename UI labels**:
   - “AI Assumptions” → “Assumptions & Forecast Config”
   - Make “Optional” explicit on tabs
3. Add the **Applied Assumptions** summary panel.

### Phase 2 (codebase cleanup)
1. Extract models + analysis into separate modules
2. Move DB reads/writes into a dedicated persistence module
3. Remove duplicated bucket logic by centralizing classification utilities

### Phase 3 (deprecation)
1. Hide legacy tabs by default (already exists), then remove after migration tooling is stable.
2. Ensure unified config is the only trend configuration path.

---

## 7) Acceptance criteria (what “done” looks like)

1. A new user can run a Pipeline forecast without touching AI Assumptions (if they have basic manual inputs).
2. If Trend method is selected, the UI clearly blocks until unified config is saved (with a direct “Go configure” action).
3. There is exactly one “Save/Apply” path that defines what the forecast will use, and it is visible in one summary.
4. No duplicated classification logic across Setup/Forecast/AI modules (single shared implementation).


