#!/bin/bash
# Component Backup Script
# Creates a backup of all components before major changes

BACKUP_DIR=".backups/components_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating component backup to $BACKUP_DIR..."

# Backup components directory
if [ -d "components" ]; then
    cp -r components "$BACKUP_DIR/"
    echo "✅ Backed up components/"
fi

# Backup root-level Python files
for file in app_refactored.py db_connector.py supabase_utils.py funding_engine.py linear_theme.py component_registry.py; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/"
        echo "✅ Backed up $file"
    fi
done

# Create backup manifest
cat > "$BACKUP_DIR/BACKUP_MANIFEST.txt" << EOF
Component Backup Manifest
========================
Date: $(date)
Backup Directory: $BACKUP_DIR

Files Backed Up:
$(find "$BACKUP_DIR" -type f -name "*.py" | sed 's|^.*/||' | sort)

Component Verification:
$(cd "$(dirname "$0")" && python3 verify_components.py 2>&1 | tail -20)
EOF

echo ""
echo "✅ Backup complete: $BACKUP_DIR"
echo "📋 Manifest: $BACKUP_DIR/BACKUP_MANIFEST.txt"
