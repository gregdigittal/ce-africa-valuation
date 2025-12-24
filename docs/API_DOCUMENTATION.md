# API Documentation

**Version:** 2.0  
**Last Updated:** December 17, 2025

---

## Overview

This document describes the API and key functions available in the CE Africa Valuation Platform.

---

## Forecast Engine

### `ForecastEngine`

Core forecast calculation engine (Sprint 21).

#### Methods

##### `run_forecast(data, manufacturing_scenario=None, progress_callback=None)`

Run complete forecast calculation.

**Parameters**:
- `data` (Dict): Forecast data containing:
  - `assumptions`: Dict of forecast assumptions
  - `ai_assumptions`: Optional AI assumptions
  - `machines`: List of machine records
  - `prospects`: List of prospect records
  - `expenses`: List of expense assumptions
- `manufacturing_scenario` (Optional): Manufacturing strategy scenario
- `progress_callback` (Optional[Callable]): Progress callback function

**Returns**: Dict with forecast results

**Example**:
```python
from forecast_engine import ForecastEngine

engine = ForecastEngine()
results = engine.run_forecast(data)
```

---

## Services Layer

### `ScenarioService`

Business logic for scenario management.

#### Methods

- `get_user_scenarios(user_id) -> List[Dict]`
- `create_scenario(user_id, name, status="draft") -> Optional[Dict]`
- `update_scenario(scenario_id, user_id, name=None, status=None) -> Optional[Dict]`
- `delete_scenario(scenario_id, user_id) -> bool`
- `get_scenario(scenario_id, user_id) -> Optional[Dict]`

### `ForecastService`

Business logic for forecast operations.

#### Methods

- `run_forecast(scenario_id, user_id, manufacturing_scenario=None, progress_callback=None) -> Dict`
- `validate_forecast_prerequisites(scenario_id, user_id) -> tuple[bool, Optional[str]]`

### `AssumptionsService`

Business logic for assumptions management.

#### Methods

- `get_assumptions(scenario_id, user_id, include_ai=True) -> Dict`
- `save_assumptions(scenario_id, user_id, assumptions, assumption_type="manual") -> bool`
- `merge_assumptions(manual, ai) -> Dict`

### `WorkflowService`

Business logic for workflow state management.

#### Methods

- `mark_stage_complete(scenario_id, stage_id, user_id) -> bool`
- `is_stage_complete(scenario_id, stage_id, user_id) -> bool`
- `get_workflow_progress(scenario_id, user_id) -> Dict[str, bool]`
- `check_prerequisites(scenario_id, stage_id, prerequisites, user_id) -> tuple[bool, List[str]]`

### `SessionManager`

Centralized session state management.

#### Methods

- `get(key, default=None) -> Any`
- `set(key, value) -> None`
- `delete(key) -> None`
- `exists(key) -> bool`
- `get_scenario_id() -> Optional[str]`
- `set_scenario_id(scenario_id) -> None`
- `get_forecast_results(scenario_id=None) -> Optional[Dict]`
- `set_forecast_results(results, scenario_id=None) -> None`
- `get_workflow_progress(scenario_id=None) -> Dict[str, bool]`
- `validate_state() -> tuple[bool, List[str]]`

---

## Database Handler

### `SupabaseHandler`

Database operations handler.

#### Key Methods

**Scenario Management**:
- `get_user_scenarios(user_id) -> List[Dict]`
- `create_scenario(user_id, name, status) -> Optional[Dict]`
- `update_scenario(scenario_id, user_id, name, status) -> Optional[Dict]`
- `delete_scenario(scenario_id, user_id) -> bool`

**Assumptions**:
- `get_scenario_assumptions(scenario_id, user_id) -> Dict`
- `update_assumptions(scenario_id, user_id, assumptions) -> bool`

**Machine Data**:
- `get_machine_instances(user_id, scenario_id, site_id=None) -> List[Dict]`
- `get_installed_base(scenario_id) -> List[Dict]`

**Historical Data**:
- `get_historic_financials(scenario_id) -> List[Dict]`
- `get_historical_financials(scenario_id) -> List[Dict]`

---

## Component Functions

### Forecast Section

#### `run_forecast(db, scenario_id, user_id, progress_callback=None, manufacturing_scenario=None)`

Run forecast (uses ForecastEngine internally).

**Parameters**:
- `db`: Database handler
- `scenario_id`: Scenario ID
- `user_id`: User ID
- `progress_callback`: Optional progress callback
- `manufacturing_scenario`: Optional manufacturing strategy

**Returns**: Forecast results dictionary

#### `load_forecast_data(db, scenario_id, user_id) -> Dict`

Load all data needed for forecasting.

**Returns**: Dictionary with machines, prospects, expenses, assumptions

#### `save_snapshot(db, scenario_id, user_id, forecast_results, mc_results=None, snapshot_name=None) -> bool`

Save forecast snapshot.

### What-If Agent

#### `render_whatif_agent(db, scenario_id, user_id, baseline_forecast=None)`

Render What-If Agent UI.

#### `calculate_adjusted_forecast(baseline, adjustments) -> Optional[Dict]`

Apply adjustments to baseline forecast.

**Parameters**:
- `baseline`: Baseline forecast results
- `adjustments`: Dict of adjustment percentages

**Returns**: Adjusted forecast results

#### `run_sensitivity_analysis(baseline, parameter_ranges=None) -> Dict`

Run sensitivity analysis.

---

## Data Structures

### Forecast Results

```python
{
    'success': bool,
    'error': Optional[str],
    'timeline': List[str],  # YYYY-MM format
    'timeline_dates': List[str],  # YYYY-MM-DD format
    'revenue': {
        'consumables': List[float],
        'refurb': List[float],
        'pipeline': List[float],
        'total': List[float]
    },
    'costs': {
        'cogs': List[float],
        'cogs_buy': List[float],
        'cogs_make': List[float],
        'opex': List[float],
        'total': List[float]
    },
    'profit': {
        'gross': List[float],
        'ebit': List[float]
    },
    'summary': {
        'total_revenue': float,
        'total_ebit': float,
        'avg_gross_margin': float,
        'avg_ebit_margin': float,
        # ... more summary stats
    }
}
```

---

## Error Handling

All service methods return:
- `None` or `False` on failure
- Appropriate data type on success

Check return values and handle errors appropriately.

---

## Examples

### Running a Forecast

```python
from services.forecast_service import ForecastService
from db_connector import SupabaseHandler

db = SupabaseHandler()
service = ForecastService(db)

results = service.run_forecast('scenario_id', 'user_id')
if results['success']:
    print(f"Total Revenue: {results['summary']['total_revenue']}")
```

### Using Session Manager

```python
from services.session_manager import SessionManager

# Set scenario
SessionManager.set_scenario_id('scenario_123')

# Get forecast results
results = SessionManager.get_forecast_results('scenario_123')

# Validate state
is_valid, issues = SessionManager.validate_state()
```
