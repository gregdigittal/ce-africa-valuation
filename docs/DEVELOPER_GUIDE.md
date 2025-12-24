# Developer Guide

**Version:** 2.0  
**Last Updated:** December 17, 2025

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Development Setup](#development-setup)
4. [Code Standards](#code-standards)
5. [Adding New Features](#adding-new-features)
6. [Testing](#testing)
7. [Debugging](#debugging)

---

## Architecture Overview

The platform follows a layered architecture (Sprint 21):

```
┌─────────────────────────────────────┐
│         UI Layer (Streamlit)        │
│  components/*.py, app_refactored.py│
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│        Service Layer                 │
│      services/*.py                   │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      Business Logic Layer           │
│    forecast_engine.py                │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      Data Access Layer              │
│    db_connector.py                  │
└─────────────────────────────────────┘
```

### Key Principles

1. **Separation of Concerns**: UI, business logic, and data access are separated
2. **Service Layer**: Business logic in service classes
3. **Testability**: Core logic is unit-testable
4. **Session Management**: Centralized via SessionManager

---

## Project Structure

```
ce-africa-valuation/
├── app_refactored.py          # Main application entry point
├── forecast_engine.py          # Core forecast calculation (Sprint 21)
├── funding_engine.py           # Funding calculations
├── db_connector.py             # Database operations
├── supabase_utils.py           # Supabase utilities
│
├── components/                 # UI components
│   ├── forecast_section.py     # Forecast UI
│   ├── ai_assumptions_engine.py
│   ├── whatif_agent.py        # What-If analysis
│   ├── vertical_integration.py # Manufacturing
│   └── ...
│
├── services/                   # Service layer (Sprint 21)
│   ├── scenario_service.py
│   ├── forecast_service.py
│   ├── assumptions_service.py
│   ├── workflow_service.py
│   └── session_manager.py
│
├── tests/                      # Test suite (Sprint 22)
│   ├── test_forecast_engine.py
│   ├── test_services.py
│   └── test_integration.py
│
└── docs/                       # Documentation (Sprint 22)
    ├── USER_MANUAL.md
    ├── API_DOCUMENTATION.md
    ├── DEVELOPER_GUIDE.md
    ├── SETUP_GUIDE.md
    └── TROUBLESHOOTING.md
```

---

## Development Setup

### Prerequisites

- Python 3.11+
- Supabase account and database
- Git

### Setup Steps

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd ce-africa-valuation
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets**
   Create `secrets.toml`:
   ```toml
   [supabase]
   url = "your-supabase-url"
   anon_key = "your-anon-key"
   ```

4. **Run Application**
   ```bash
   streamlit run app_refactored.py
   ```

### Development Tools

- **Testing**: `pytest tests/`
- **Linting**: Use your IDE's Python linter
- **Type Checking**: Consider adding `mypy` for type checking

---

## Code Standards

### Python Style

- Follow PEP 8
- Use type hints where possible
- Document functions with docstrings
- Keep functions focused (single responsibility)

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ForecastEngine`)
- **Functions**: `snake_case` (e.g., `run_forecast`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `TABLE_SCENARIOS`)
- **Private methods**: `_leading_underscore`

### File Organization

- One class per file (for large classes)
- Related functions grouped together
- Imports at top, organized by:
  1. Standard library
  2. Third-party
  3. Local imports

### Error Handling

```python
try:
    result = operation()
    return result
except SpecificError as e:
    # Handle specific error
    logger.error(f"Error: {e}")
    return None
except Exception as e:
    # Handle general error
    logger.error(f"Unexpected error: {e}")
    return None
```

---

## Adding New Features

### 1. Plan the Feature

- Define requirements
- Identify affected components
- Plan data model changes (if needed)

### 2. Create Service (if needed)

If business logic is involved, create a service:

```python
# services/new_feature_service.py
class NewFeatureService:
    def __init__(self, db_handler):
        self.db = db_handler
    
    def do_something(self, param):
        # Business logic here
        pass
```

### 3. Create UI Component

```python
# components/new_feature.py
def render_new_feature(db, scenario_id, user_id):
    """Render new feature UI."""
    st.header("New Feature")
    # UI code here
```

### 4. Integrate into App

Add to `app_refactored.py`:
- Import component
- Add to navigation
- Add to workflow (if applicable)

### 5. Write Tests

```python
# tests/test_new_feature.py
def test_new_feature_service():
    service = NewFeatureService(mock_db)
    result = service.do_something('param')
    assert result is not None
```

---

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_forecast_engine.py

# With coverage
pytest --cov=forecast_engine --cov=services
```

### Writing Tests

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **Use Fixtures**: Share test data via `conftest.py`

### Test Structure

```python
class TestFeature:
    def test_basic_functionality(self):
        """Test basic feature works."""
        # Arrange
        # Act
        # Assert
```

---

## Debugging

### Common Issues

1. **Session State Issues**
   - Use `SessionManager.get_state_summary()` to inspect state
   - Clear cache: `SessionManager.clear_cache()`

2. **Database Errors**
   - Check RLS policies
   - Verify `user_id` is set correctly
   - Check connection string

3. **Import Errors**
   - Verify Python path
   - Check `__init__.py` files exist
   - Verify relative imports

### Debugging Tools

- **Streamlit Debug**: Use `st.write()` for inspection
- **Python Debugger**: `import pdb; pdb.set_trace()`
- **Logging**: Add logging statements

---

## Best Practices

1. **Use Services**: Don't access database directly from UI
2. **Validate Inputs**: Check user inputs before processing
3. **Handle Errors**: Provide user-friendly error messages
4. **Cache Data**: Use session state for expensive operations
5. **Document Code**: Write clear docstrings
6. **Write Tests**: Test new features
7. **Follow Workflow**: Use the established workflow system

---

## Contributing

1. Create feature branch
2. Make changes
3. Write/update tests
4. Update documentation
5. Submit for review

---

## Resources

- **Backlog**: `UPDATED_BACKLOG.md`
- **API Docs**: `docs/API_DOCUMENTATION.md`
- **Database Schema**: `docs/DATABASE_SCHEMA.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
