# Setup Guide

**Version:** 2.0  
**Last Updated:** December 17, 2025

---

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Supabase account and database
- Git (optional, for version control)

---

## Installation Steps

### 1. Clone or Download Repository

```bash
git clone <repository-url>
cd ce-africa-valuation
```

Or download and extract the ZIP file.

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:

```bash
pip install streamlit pandas numpy plotly scipy supabase python-dateutil
```

### 4. Configure Supabase

Create a `.streamlit/secrets.toml` file:

```toml
[supabase]
url = "https://your-project.supabase.co"
anon_key = "your-anon-key"
# OR
service_role_key = "your-service-role-key"

[dev]
user_id = "your-dev-user-id"  # Optional, for development
```

**Getting Supabase Credentials**:
1. Go to your Supabase project dashboard
2. Settings → API
3. Copy "Project URL" and "anon public" key

### 5. Database Setup

Run database migrations (if provided):

```bash
# Example (adjust path as needed)
psql $DATABASE_URL -f migrations/initial_schema.sql
```

Or use Supabase SQL Editor to run migrations manually.

### 6. Launch Application

```bash
streamlit run app_refactored.py
```

The application will open in your browser at `http://localhost:8501`

---

## Configuration

### Environment Variables

Optional environment variables:

- `DATABASE_URL`: PostgreSQL connection string
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_KEY`: Supabase API key

### Application Settings

Configure in `app_refactored.py` or via Streamlit config:

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#D4A537"
backgroundColor = "#09090B"
secondaryBackgroundColor = "#0F0F11"
```

---

## Verification

### Test Database Connection

1. Launch application
2. Check for database connection errors
3. Try creating a test scenario

### Test Components

Run component verification:

```bash
python scripts/verify_components.py
```

### Run Tests

```bash
pytest tests/
```

---

## Troubleshooting

### Common Setup Issues

**"Module not found" errors**:
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

**"Supabase connection failed"**:
- Check `secrets.toml` exists and is correct
- Verify Supabase URL and key
- Check network connectivity

**"Database table not found"**:
- Run database migrations
- Check table names match schema
- Verify RLS policies are set

**"Port already in use"**:
- Change port: `streamlit run app_refactored.py --server.port 8502`
- Or stop other Streamlit instances

---

## Next Steps

After setup:

1. Read the [User Manual](USER_MANUAL.md)
2. Create your first scenario
3. Import test data
4. Run a forecast

---

## Support

For setup issues:
- Check [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review [Developer Guide](DEVELOPER_GUIDE.md)
- Check Supabase documentation
