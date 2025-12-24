#!/usr/bin/env python3
"""
Component Verification Script
============================
Verifies that all required components exist and are importable.
Run this before commits to prevent regression.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Required components and their expected functions
REQUIRED_COMPONENTS = {
    'command_center': ['render_command_center'],
    'setup_wizard': ['render_setup_wizard'],
    'forecast_section': ['render_forecast_section'],
    'scenario_comparison': ['render_scenario_comparison'],
    'ai_assumptions_engine': ['render_ai_assumptions_section', 'get_saved_assumptions'],
    'vertical_integration': ['render_vertical_integration_section'],
    'ai_trend_analysis': ['render_trend_analysis_section'],
    'funding_ui': ['render_funding_section'],
    'user_management': ['render_user_management'],
    'ui_components': ['inject_custom_css', 'section_header'],
    'financial_statements': ['render_financial_statements'],
    'enhanced_navigation': ['render_enhanced_sidebar_nav'],
    'ai_assumptions_integration': ['render_ai_assumptions_summary'],
    'column_mapper': ['render_column_mapper', 'render_import_with_mapping', 'FIELD_CONFIGS'],
}

# Optional components (graceful fallbacks if missing)
OPTIONAL_COMPONENTS = {
    'workflow_navigator': ['render_workflow_navigator_simple'],
    'workflow_guidance': ['render_contextual_help', 'get_stage_tips', 'render_prerequisite_warning', 'render_whats_next_widget'],
    'whatif_agent': ['render_whatif_agent'],
}

# Required root files
REQUIRED_ROOT_FILES = [
    'app_refactored.py',
    'db_connector.py',
    'supabase_utils.py',
    'funding_engine.py',
    'linear_theme.py',
]

def check_file_exists(filepath: Path) -> bool:
    """Check if a file exists."""
    return filepath.exists() and filepath.is_file()

def check_component_import(component_name: str, functions: list) -> tuple[bool, list]:
    """Check if a component can be imported and has required functions."""
    errors = []
    try:
        module = __import__(f'components.{component_name}', fromlist=functions)
        for func_name in functions:
            if not hasattr(module, func_name):
                errors.append(f"Missing function: {func_name}")
        return len(errors) == 0, errors
    except ImportError as e:
        return False, [f"Import error: {str(e)}"]
    except Exception as e:
        return False, [f"Error: {str(e)}"]

def check_root_file_import(filename: str) -> tuple[bool, str]:
    """Check if a root file can be imported."""
    try:
        module_name = filename.replace('.py', '')
        __import__(module_name)
        return True, ""
    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Run component verification."""
    print("=" * 60)
    print("Component Verification")
    print("=" * 60)
    print()
    
    all_passed = True
    errors = []
    
    # Check root files exist
    print("Checking root files...")
    for filename in REQUIRED_ROOT_FILES:
        filepath = project_root / filename
        if check_file_exists(filepath):
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} - MISSING")
            errors.append(f"Missing root file: {filename}")
            all_passed = False
    print()
    
    # Check root files are importable
    print("Checking root file imports...")
    for filename in REQUIRED_ROOT_FILES:
        if filename == 'app_refactored.py':
            # Skip app_refactored as it requires Streamlit context
            print(f"  ⏭️  {filename} (skipped - requires Streamlit)")
            continue
        success, error_msg = check_root_file_import(filename)
        if success:
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename}: {error_msg}")
            errors.append(f"{filename}: {error_msg}")
            all_passed = False
    print()
    
    # Check components exist
    print("Checking component files...")
    for component_name in REQUIRED_COMPONENTS.keys():
        filepath = project_root / 'components' / f'{component_name}.py'
        if check_file_exists(filepath):
            print(f"  ✅ components/{component_name}.py")
        else:
            print(f"  ❌ components/{component_name}.py - MISSING")
            errors.append(f"Missing component: {component_name}.py")
            all_passed = False
    print()
    
    # Check component imports (skip if file doesn't exist)
    print("Checking component imports...")
    for component_name, functions in REQUIRED_COMPONENTS.items():
        filepath = project_root / 'components' / f'{component_name}.py'
        if not check_file_exists(filepath):
            continue  # Already reported as missing
        
        success, component_errors = check_component_import(component_name, functions)
        if success:
            print(f"  ✅ {component_name} (all functions available)")
        else:
            print(f"  ❌ {component_name}:")
            for error in component_errors:
                print(f"      - {error}")
            errors.extend([f"{component_name}: {e}" for e in component_errors])
            all_passed = False
    
    # Check optional components (warn but don't fail)
    print("\nChecking optional components...")
    for component_name, functions in OPTIONAL_COMPONENTS.items():
        filepath = project_root / 'components' / f'{component_name}.py'
        if check_file_exists(filepath):
            success, component_errors = check_component_import(component_name, functions)
            if success:
                print(f"  ✅ {component_name} (optional, available)")
            else:
                print(f"  ⚠️  {component_name} (optional, has issues):")
                for error in component_errors:
                    print(f"      - {error}")
        else:
            print(f"  ⏭️  {component_name} (optional, not present)")
    print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("=" * 60)
        return 0
    else:
        print("❌ VERIFICATION FAILED")
        print("=" * 60)
        print("\nErrors found:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix these issues before committing.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
