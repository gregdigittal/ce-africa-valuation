#!/bin/bash
# Backup Before Changes Script
# Creates a backup before making changes to prevent regression

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="$PROJECT_ROOT/.backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="pre_change_${TIMESTAMP}"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"

echo "Creating backup before changes..."
echo "Backup location: $BACKUP_PATH"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Backup critical files
echo "Backing up critical files..."

# Root files
for file in app_refactored.py db_connector.py supabase_utils.py funding_engine.py linear_theme.py; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        cp "$PROJECT_ROOT/$file" "$BACKUP_PATH/"
        echo "  ✅ $file"
    fi
done

# Components directory
if [ -d "$PROJECT_ROOT/components" ]; then
    mkdir -p "$BACKUP_PATH/components"
    cp -r "$PROJECT_ROOT/components"/*.py "$BACKUP_PATH/components/" 2>/dev/null || true
    echo "  ✅ components/ directory"
fi

# Create backup metadata
cat > "$BACKUP_PATH/BACKUP_INFO.txt" << EOF
Backup Created: $(date)
Purpose: Pre-change backup
Location: $BACKUP_PATH

Files backed up:
- Root Python files
- All component files

To restore:
  cp $BACKUP_PATH/*.py $PROJECT_ROOT/
  cp $BACKUP_PATH/components/*.py $PROJECT_ROOT/components/
EOF

echo ""
echo "✅ Backup complete: $BACKUP_PATH"
echo ""
echo "To restore this backup:"
echo "  cp $BACKUP_PATH/*.py $PROJECT_ROOT/"
echo "  cp $BACKUP_PATH/components/*.py $PROJECT_ROOT/components/"
