#!/bin/bash
# Backup script for CE Africa Valuation Platform
# Creates a timestamped backup of the entire codebase

set -e

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_ROOT="$PROJECT_ROOT/.backups"

# Create backup directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="pre_sprint_ai_assumptions_${TIMESTAMP}"
BACKUP_DIR="$BACKUP_ROOT/$BACKUP_NAME"

echo "Creating backup: $BACKUP_NAME"
echo "Backup location: $BACKUP_DIR"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Copy all important files and directories
echo "Copying files..."

# Core application files
cp "$PROJECT_ROOT/app_refactored.py" "$BACKUP_DIR/" 2>/dev/null || true
cp "$PROJECT_ROOT/db_connector.py" "$BACKUP_DIR/" 2>/dev/null || true
cp "$PROJECT_ROOT/supabase_utils.py" "$BACKUP_DIR/" 2>/dev/null || true
cp "$PROJECT_ROOT/forecast_engine.py" "$BACKUP_DIR/" 2>/dev/null || true
cp "$PROJECT_ROOT/funding_engine.py" "$BACKUP_DIR/" 2>/dev/null || true
cp "$PROJECT_ROOT/requirements.txt" "$BACKUP_DIR/" 2>/dev/null || true

# Components directory
if [ -d "$PROJECT_ROOT/components" ]; then
    echo "Copying components..."
    cp -r "$PROJECT_ROOT/components" "$BACKUP_DIR/"
fi

# Services directory
if [ -d "$PROJECT_ROOT/services" ]; then
    echo "Copying services..."
    cp -r "$PROJECT_ROOT/services" "$BACKUP_DIR/"
fi

# Migration files
echo "Copying migrations..."
mkdir -p "$BACKUP_DIR/migrations"
cp "$PROJECT_ROOT"/migrations_*.sql "$BACKUP_DIR/migrations/" 2>/dev/null || true

# Scripts directory
if [ -d "$PROJECT_ROOT/scripts" ]; then
    echo "Copying scripts..."
    cp -r "$PROJECT_ROOT/scripts" "$BACKUP_DIR/"
fi

# Documentation
echo "Copying documentation..."
mkdir -p "$BACKUP_DIR/docs"
cp "$PROJECT_ROOT"/*.md "$BACKUP_DIR/docs/" 2>/dev/null || true
if [ -d "$PROJECT_ROOT/docs" ]; then
    cp -r "$PROJECT_ROOT/docs" "$BACKUP_DIR/"
fi

# Configuration files (excluding secrets)
echo "Copying configuration..."
if [ -d "$PROJECT_ROOT/.streamlit" ]; then
    mkdir -p "$BACKUP_DIR/.streamlit"
    cp "$PROJECT_ROOT/.streamlit/config.toml" "$BACKUP_DIR/.streamlit/" 2>/dev/null || true
    # Don't copy secrets.toml for security
    echo "# secrets.toml not backed up for security reasons" > "$BACKUP_DIR/.streamlit/secrets.toml.example"
fi

# Tests
if [ -d "$PROJECT_ROOT/tests" ]; then
    echo "Copying tests..."
    cp -r "$PROJECT_ROOT/tests" "$BACKUP_DIR/"
fi

# Create backup info file
cat > "$BACKUP_DIR/BACKUP_INFO.txt" << EOF
CE Africa Valuation Platform - Backup
=====================================
Backup Date: $(date)
Backup Name: $BACKUP_NAME
Backup Reason: Pre-Sprint AI Assumptions Enhancements

Files Included:
- All Python source files
- Components directory
- Services directory
- Migration SQL files
- Documentation
- Configuration files (excluding secrets)
- Tests

To restore:
1. Copy files from this backup to project root
2. Restore secrets.toml manually if needed
3. Run migrations if database schema changed

Git Status at backup:
$(cd "$PROJECT_ROOT" && git status --short 2>/dev/null || echo "Not a git repository or git not available")
EOF

# Create a compressed archive
echo "Creating compressed archive..."
cd "$BACKUP_ROOT"
tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME" 2>/dev/null || zip -r "${BACKUP_NAME}.zip" "$BACKUP_NAME" 2>/dev/null || true

echo ""
echo "✅ Backup completed successfully!"
echo "📁 Backup location: $BACKUP_DIR"
if [ -f "$BACKUP_ROOT/${BACKUP_NAME}.tar.gz" ] || [ -f "$BACKUP_ROOT/${BACKUP_NAME}.zip" ]; then
    echo "📦 Compressed archive created"
fi
echo ""
echo "Backup info saved to: $BACKUP_DIR/BACKUP_INFO.txt"
