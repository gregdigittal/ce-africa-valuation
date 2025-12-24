#!/usr/bin/env python3
"""
Python Migration Runner
Runs SQL migrations against Supabase/PostgreSQL database without requiring psql.

Usage:
    python run_migration.py migrations_add_detailed_line_items.sql
    
Or with explicit database URL:
    DATABASE_URL='postgresql://...' python run_migration.py migrations_add_detailed_line_items.sql
"""

import sys
import os
from pathlib import Path

def get_database_url():
    """Get database URL from environment or secrets file."""
    # First, try environment variable
    if 'DATABASE_URL' in os.environ:
        return os.environ['DATABASE_URL']
    
    # Try to read from secrets.toml
    secrets_path = Path(__file__).parent / '.streamlit' / 'secrets.toml'
    if secrets_path.exists():
        try:
            # Try toml first (if available)
            try:
                import toml
                secrets = toml.load(secrets_path)
            except ImportError:
                # Fallback: simple TOML parsing for our use case
                secrets = {}
                with open(secrets_path, 'r') as f:
                    content = f.read()
                    # Simple regex to extract supabase URL
                    import re
                    url_match = re.search(r'url\s*=\s*"([^"]+)"', content)
                    if url_match:
                        secrets = {'supabase': {'url': url_match.group(1)}}
            
            supabase_url = secrets.get('supabase', {}).get('url', '')
            
            if supabase_url:
                # Extract database connection from Supabase URL
                # Supabase URL format: https://xxxxx.supabase.co
                # Database connection: postgresql://postgres:PASSWORD@db.xxxxx.supabase.co:5432/postgres
                
                # Try to get password from environment or prompt
                password = os.environ.get('SUPABASE_DB_PASSWORD', '')
                
                if not password:
                    print("⚠️  Database password not found in environment.")
                    print("   Please set SUPABASE_DB_PASSWORD environment variable, or")
                    print("   provide DATABASE_URL directly.")
                    print("")
                    print("   Example:")
                    print("   export SUPABASE_DB_PASSWORD='your-password'")
                    print("   python run_migration.py migrations_add_detailed_line_items.sql")
                    print("")
                    print("   Or find your password at:")
                    print("   https://supabase.com/dashboard → Your Project → Settings → Database")
                    print("")
                    password = input("Enter your Supabase database password: ").strip()
                
                if password:
                    # Extract project ref from URL
                    # URL: https://qxbngbmpstwebjkbpdcj.supabase.co
                    # Ref: qxbngbmpstwebjkbpdcj
                    project_ref = supabase_url.replace('https://', '').replace('.supabase.co', '')
                    return f"postgresql://postgres:{password}@db.{project_ref}.supabase.co:5432/postgres"
        except Exception as e:
            print(f"⚠️  Error reading secrets: {e}")
    
    return None

def run_migration(migration_file: str, database_url: str = None):
    """Run a SQL migration file."""
    if database_url is None:
        database_url = get_database_url()
    
    if not database_url:
        print("❌ Error: Database URL not found")
        print("")
        print("Please provide DATABASE_URL environment variable:")
        print("  export DATABASE_URL='postgresql://postgres:PASSWORD@host:5432/postgres'")
        print("  python run_migration.py migrations_add_detailed_line_items.sql")
        return False
    
    # Read migration file
    migration_path = Path(migration_file)
    if not migration_path.exists():
        print(f"❌ Error: Migration file not found: {migration_file}")
        return False
    
    print(f"🔄 Running migration: {migration_file}")
    print(f"📊 Database: {database_url.split('@')[0]}@***")
    print("")
    
    # Read SQL file
    try:
        with open(migration_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
    except Exception as e:
        print(f"❌ Error reading migration file: {e}")
        return False
    
    # Try to use psycopg2 (PostgreSQL adapter)
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        # Parse connection string
        # Format: postgresql://user:password@host:port/database
        import urllib.parse
        parsed = urllib.parse.urlparse(database_url)
        
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path.lstrip('/'),
            user=parsed.username,
            password=parsed.password
        )
        
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Execute SQL (split by semicolons, but handle DO $$ blocks)
        # Simple approach: execute the entire file
        cursor.execute(sql_content)
        
        cursor.close()
        conn.close()
        
        print("✅ Migration completed successfully!")
        return True
        
    except ImportError:
        print("❌ Error: psycopg2 not installed")
        print("")
        print("Install it with:")
        print("  pip install psycopg2-binary")
        print("")
        print("Or use the Supabase dashboard to run the migration manually:")
        print("  1. Go to https://supabase.com/dashboard")
        print("  2. Select your project")
        print("  3. Go to SQL Editor")
        print("  4. Paste the contents of migrations_add_detailed_line_items.sql")
        print("  5. Run the query")
        return False
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        print("")
        print("Alternative: Run the migration via Supabase Dashboard:")
        print("  1. Go to https://supabase.com/dashboard")
        print("  2. Select your project")
        print("  3. Go to SQL Editor")
        print("  4. Paste the contents of migrations_add_detailed_line_items.sql")
        print("  5. Run the query")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: Migration file not specified")
        print("")
        print("Usage:")
        print("  python run_migration.py <migration_file.sql>")
        print("")
        print("Example:")
        print("  python run_migration.py migrations_add_detailed_line_items.sql")
        sys.exit(1)
    
    migration_file = sys.argv[1]
    success = run_migration(migration_file)
    sys.exit(0 if success else 1)
