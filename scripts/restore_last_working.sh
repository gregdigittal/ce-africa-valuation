#!/bin/bash
# Restore to Last Known Working Model
# Automatically finds and restores the most recent backup

set -e

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_ROOT="$PROJECT_ROOT/.backups"

echo "🔄 Restoring to Last Known Working Model..."
echo ""

# Find the most recent backup
if [ ! -d "$BACKUP_ROOT" ]; then
    echo "❌ No backups directory found!"
    exit 1
fi

# Find the most recent backup directory (excluding compressed archives)
LATEST_BACKUP=$(find "$BACKUP_ROOT" -maxdepth 1 -type d -name "pre_*" -o -name "*backup*" | sort -r | head -1)

if [ -z "$LATEST_BACKUP" ] || [ ! -d "$LATEST_BACKUP" ]; then
    echo "❌ No backup found to restore from!"
    echo "   Checked: $BACKUP_ROOT"
    exit 1
fi

BACKUP_NAME=$(basename "$LATEST_BACKUP")
echo "📦 Found backup: $BACKUP_NAME"
echo ""

# Confirm before restoring
read -p "⚠️  This will overwrite current files. Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "❌ Restore cancelled."
    exit 0
fi

echo ""
echo "🔄 Restoring files..."

# Restore root files
echo "  📄 Restoring root files..."
for file in app_refactored.py db_connector.py supabase_utils.py forecast_engine.py funding_engine.py requirements.txt; do
    if [ -f "$LATEST_BACKUP/$file" ]; then
        cp "$LATEST_BACKUP/$file" "$PROJECT_ROOT/$file"
        echo "    ✅ $file"
    fi
done

# Restore components directory
if [ -d "$LATEST_BACKUP/components" ]; then
    echo "  📁 Restoring components..."
    # Remove existing components and restore from backup
    rm -rf "$PROJECT_ROOT/components"/*.py
    cp -r "$LATEST_BACKUP/components"/*.py "$PROJECT_ROOT/components/" 2>/dev/null || true
    echo "    ✅ components/ directory"
fi

# Restore services directory
if [ -d "$LATEST_BACKUP/services" ]; then
    echo "  📁 Restoring services..."
    rm -rf "$PROJECT_ROOT/services"/*.py
    cp -r "$LATEST_BACKUP/services"/*.py "$PROJECT_ROOT/services/" 2>/dev/null || true
    echo "    ✅ services/ directory"
fi

# Restore migrations
if [ -d "$LATEST_BACKUP/migrations" ]; then
    echo "  📄 Restoring migrations..."
    cp "$LATEST_BACKUP/migrations"/*.sql "$PROJECT_ROOT/" 2>/dev/null || true
    echo "    ✅ migrations/"
fi

# Restore tests
if [ -d "$LATEST_BACKUP/tests" ]; then
    echo "  🧪 Restoring tests..."
    cp -r "$LATEST_BACKUP/tests"/* "$PROJECT_ROOT/tests/" 2>/dev/null || true
    echo "    ✅ tests/ directory"
fi

# Restore scripts (but not this restore script itself)
if [ -d "$LATEST_BACKUP/scripts" ]; then
    echo "  📜 Restoring scripts..."
    for script in "$LATEST_BACKUP/scripts"/*.sh; do
        if [ -f "$script" ] && [ "$(basename "$script")" != "restore_last_working.sh" ]; then
            cp "$script" "$PROJECT_ROOT/scripts/"
        fi
    done
    echo "    ✅ scripts/ directory"
fi

echo ""
echo "✅ Restore completed successfully!"
echo ""
echo "📋 Restored from: $BACKUP_NAME"
echo "📅 Backup date: $(cat "$LATEST_BACKUP/BACKUP_INFO.txt" 2>/dev/null | grep "Backup Date" | cut -d: -f2- || echo "Unknown")"
echo ""
echo "⚠️  Note: secrets.toml was NOT restored (for security)."
echo "   If you need to restore secrets, do so manually."
echo ""
echo "🧪 Next steps:"
echo "   1. Test the application: streamlit run app_refactored.py"
echo "   2. Check for any errors"
echo "   3. Verify all features work as expected"
