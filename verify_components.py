#!/usr/bin/env python3
"""
Component Verification Script
=============================
Verifies all components are present and importable.
Run this before commits or as part of CI/CD.

Usage:
    python verify_components.py
    python verify_components.py --strict  # Fail if any optional components missing
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from component_registry import (
    verify_all_components,
    get_missing_required_components,
    COMPONENT_REGISTRY,
    ComponentStatus,
    print_component_status
)

def main(strict: bool = False):
    """Main verification function"""
    print("🔍 Verifying Component Integrity...\n")
    
    # Verify all components
    results = verify_all_components()
    
    # Check required components
    missing_required = get_missing_required_components()
    
    # Print status
    all_ok = print_component_status()
    
    # Check optional components if strict mode
    if strict:
        missing_optional = [
            name for name, result in results.items()
            if COMPONENT_REGISTRY[name].status == ComponentStatus.OPTIONAL
            and (not result['importable'] or not result['function_exists'])
        ]
        
        if missing_optional:
            print(f"\n⚠️  Strict mode: {len(missing_optional)} optional components missing")
            print(f"   {', '.join(missing_optional)}")
    
    # Exit code
    if missing_required:
        print("\n❌ VERIFICATION FAILED: Required components missing!")
        return 1
    elif strict and missing_optional:
        print("\n⚠️  VERIFICATION WARNING: Optional components missing (strict mode)")
        return 2
    else:
        print("\n✅ VERIFICATION PASSED: All components OK")
        return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify component integrity')
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail if optional components are missing'
    )
    
    args = parser.parse_args()
    exit_code = main(strict=args.strict)
    sys.exit(exit_code)
