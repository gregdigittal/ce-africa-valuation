#!/bin/bash

# Migration Runner Script
# Usage: ./run_migration_psql.sh <migration_file.sql>

if [ -z "$1" ]; then
    echo "❌ Error: Migration file not specified"
    echo "Usage: ./run_migration_psql.sh <migration_file.sql>"
    exit 1
fi

MIGRATION_FILE="$1"

if [ ! -f "$MIGRATION_FILE" ]; then
    echo "❌ Error: Migration file not found: $MIGRATION_FILE"
    exit 1
fi

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ Error: DATABASE_URL environment variable not set"
    echo ""
    echo "Please set it first:"
    echo "  export DATABASE_URL='postgresql://postgres:password@host:5432/postgres'"
    echo ""
    echo "Or for Supabase:"
    echo "  export DATABASE_URL='postgresql://postgres:YOUR_PASSWORD@db.xxxxx.supabase.co:5432/postgres'"
    exit 1
fi

echo "🔄 Running migration: $MIGRATION_FILE"
echo "📊 Database: $(echo $DATABASE_URL | sed 's/:[^:]*@/:***@/')"
echo ""

# Run the migration
psql "$DATABASE_URL" -f "$MIGRATION_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Migration completed successfully!"
else
    echo ""
    echo "❌ Migration failed. Please check the error messages above."
    exit 1
fi
