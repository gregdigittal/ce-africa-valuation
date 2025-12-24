#!/usr/bin/env python3
"""
Component Inventory Tracker
==========================
Generates a current inventory of all components and their status.
Run this to track component availability over time.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def get_component_info(component_path: Path) -> dict:
    """Get information about a component file."""
    info = {
        'exists': component_path.exists(),
        'size': component_path.stat().st_size if component_path.exists() else 0,
        'modified': datetime.fromtimestamp(component_path.stat().st_mtime).isoformat() if component_path.exists() else None,
    }
    return info

def scan_components() -> dict:
    """Scan all components and generate inventory."""
    inventory = {
        'timestamp': datetime.now().isoformat(),
        'root_files': {},
        'components': {},
    }
    
    # Scan root files
    root_files = [
        'app_refactored.py',
        'db_connector.py',
        'supabase_utils.py',
        'funding_engine.py',
        'linear_theme.py',
    ]
    
    for filename in root_files:
        filepath = project_root / filename
        inventory['root_files'][filename] = get_component_info(filepath)
    
    # Scan components directory
    components_dir = project_root / 'components'
    if components_dir.exists():
        for component_file in sorted(components_dir.glob('*.py')):
            component_name = component_file.stem
            inventory['components'][component_name] = get_component_info(component_file)
    
    return inventory

def generate_report(inventory: dict) -> str:
    """Generate a human-readable report."""
    report = []
    report.append("=" * 60)
    report.append("Component Inventory Report")
    report.append("=" * 60)
    report.append(f"Generated: {inventory['timestamp']}")
    report.append("")
    
    # Root files
    report.append("Root Files:")
    for filename, info in inventory['root_files'].items():
        status = "✅" if info['exists'] else "❌"
        size_kb = info['size'] / 1024 if info['exists'] else 0
        report.append(f"  {status} {filename} ({size_kb:.1f} KB)")
    report.append("")
    
    # Components
    report.append("Components:")
    for component_name, info in sorted(inventory['components'].items()):
        status = "✅" if info['exists'] else "❌"
        size_kb = info['size'] / 1024 if info['exists'] else 0
        modified = info['modified'][:10] if info['modified'] else "N/A"
        report.append(f"  {status} {component_name}.py ({size_kb:.1f} KB, modified: {modified})")
    report.append("")
    
    # Summary
    total_root = len(inventory['root_files'])
    existing_root = sum(1 for info in inventory['root_files'].values() if info['exists'])
    total_components = len(inventory['components'])
    existing_components = sum(1 for info in inventory['components'].values() if info['exists'])
    
    report.append("Summary:")
    report.append(f"  Root files: {existing_root}/{total_root} ({existing_root/total_root*100:.0f}%)")
    report.append(f"  Components: {existing_components}/{total_components} ({existing_components/total_components*100:.0f}%)")
    report.append("=" * 60)
    
    return "\n".join(report)

def main():
    """Generate component inventory."""
    inventory = scan_components()
    
    # Print report
    print(generate_report(inventory))
    
    # Save JSON inventory
    inventory_file = project_root / '.component_inventory.json'
    with open(inventory_file, 'w') as f:
        json.dump(inventory, f, indent=2)
    
    print(f"\nInventory saved to: {inventory_file}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
