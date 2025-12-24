"""
Component Registry
==================
Central registry of all required components and their dependencies.
Used to verify system integrity and prevent regressions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

class ComponentStatus(Enum):
    """Component status enumeration"""
    REQUIRED = "required"  # Must be present for system to function
    OPTIONAL = "optional"  # Nice to have, has fallback
    ENHANCEMENT = "enhancement"  # Future feature

@dataclass
class ComponentInfo:
    """Information about a component"""
    name: str
    module_path: str
    function_name: str
    status: ComponentStatus
    description: str
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0"
    sprint: Optional[str] = None
    fallback_available: bool = False

# =============================================================================
# COMPONENT REGISTRY
# =============================================================================

COMPONENT_REGISTRY: Dict[str, ComponentInfo] = {
    # Core Infrastructure
    'db_connector': ComponentInfo(
        name='Database Connector',
        module_path='db_connector',
        function_name='SupabaseHandler',
        status=ComponentStatus.REQUIRED,
        description='Supabase database handler',
        sprint='Sprint 1',
        fallback_available=False
    ),
    'supabase_utils': ComponentInfo(
        name='Supabase Utils',
        module_path='supabase_utils',
        function_name='get_user_id',
        status=ComponentStatus.REQUIRED,
        description='Supabase utility functions',
        sprint='Sprint 1',
        fallback_available=False
    ),
    
    # Core Components
    'command_center': ComponentInfo(
        name='Command Center',
        module_path='components.command_center',
        function_name='render_command_center',
        status=ComponentStatus.REQUIRED,
        description='Dashboard home with scenario health',
        sprint='Sprint 16',
        fallback_available=False
    ),
    'setup_wizard': ComponentInfo(
        name='Setup Wizard',
        module_path='components.setup_wizard',
        function_name='render_setup_wizard',
        status=ComponentStatus.REQUIRED,
        description='Guided configuration wizard',
        sprint='Sprint 2',
        fallback_available=False
    ),
    'forecast_section': ComponentInfo(
        name='Forecast Section',
        module_path='components.forecast_section',
        function_name='render_forecast_section',
        status=ComponentStatus.REQUIRED,
        description='Financial forecasting and analysis',
        sprint='Sprint 4',
        fallback_available=False,
        dependencies=['ai_assumptions_engine']
    ),
    'financial_statements': ComponentInfo(
        name='Financial Statements',
        module_path='components.financial_statements',
        function_name='render_financial_statements',
        status=ComponentStatus.REQUIRED,
        description='Income statement, balance sheet, cash flow',
        sprint='Sprint 4',
        fallback_available=False
    ),
    'scenario_comparison': ComponentInfo(
        name='Scenario Comparison',
        module_path='components.scenario_comparison',
        function_name='render_scenario_comparison',
        status=ComponentStatus.REQUIRED,
        description='Multi-scenario comparison and variance analysis',
        sprint='Sprint 6',
        fallback_available=False
    ),
    
    # AI Components
    'ai_assumptions_engine': ComponentInfo(
        name='AI Assumptions Engine',
        module_path='components.ai_assumptions_engine',
        function_name='render_ai_assumptions_section',
        status=ComponentStatus.REQUIRED,
        description='AI-powered assumption derivation (required before Forecast)',
        sprint='Sprint 14',
        fallback_available=True  # Has stub functions
    ),
    'ai_assumptions_integration': ComponentInfo(
        name='AI Assumptions Integration',
        module_path='components.ai_assumptions_integration',
        function_name='render_ai_assumptions_summary',
        status=ComponentStatus.REQUIRED,
        description='AI assumptions UI integration',
        sprint='Sprint 14',
        fallback_available=True
    ),
    'ai_trend_analysis': ComponentInfo(
        name='AI Trend Analysis',
        module_path='components.ai_trend_analysis',
        function_name='render_trend_analysis_section',
        status=ComponentStatus.OPTIONAL,
        description='Automated trend detection and analysis',
        sprint='Sprint 14',
        fallback_available=True
    ),
    
    # Business Logic Components
    'vertical_integration': ComponentInfo(
        name='Manufacturing Strategy',
        module_path='components.vertical_integration',
        function_name='render_vertical_integration_section',
        status=ComponentStatus.REQUIRED,
        description='Make vs Buy analysis and manufacturing strategy',
        sprint='Sprint 13',
        fallback_available=True,
        dependencies=['forecast_section']
    ),
    'funding_ui': ComponentInfo(
        name='Funding & Returns',
        module_path='components.funding_ui',
        function_name='render_funding_section',
        status=ComponentStatus.REQUIRED,
        description='Debt/equity management and IRR analysis',
        sprint='Sprint 11',
        fallback_available=True,
        dependencies=['funding_engine', 'forecast_section']
    ),
    'funding_engine': ComponentInfo(
        name='Funding Engine',
        module_path='funding_engine',
        function_name='FundingEngine',
        status=ComponentStatus.REQUIRED,
        description='Funding calculation engine',
        sprint='Sprint 11',
        fallback_available=False,
        dependencies=[]
    ),
    
    # UI Components
    'ui_components': ComponentInfo(
        name='UI Components',
        module_path='components.ui_components',
        function_name='inject_custom_css',
        status=ComponentStatus.REQUIRED,
        description='Reusable UI component library',
        sprint='Sprint 16',
        fallback_available=True,
        dependencies=['linear_theme']
    ),
    'linear_theme': ComponentInfo(
        name='Linear Theme',
        module_path='linear_theme',
        function_name='apply_ce_africa_branding',
        status=ComponentStatus.REQUIRED,
        description='Linear design system theme',
        sprint='Sprint 16',
        fallback_available=False
    ),
    'enhanced_navigation': ComponentInfo(
        name='Enhanced Navigation',
        module_path='components.enhanced_navigation',
        function_name='render_enhanced_sidebar_nav',
        status=ComponentStatus.OPTIONAL,
        description='Enhanced navigation components',
        sprint='Sprint 16.5',
        fallback_available=True
    ),
    
    # User Management
    'user_management': ComponentInfo(
        name='User Management',
        module_path='components.user_management',
        function_name='render_user_management',
        status=ComponentStatus.OPTIONAL,
        description='Access control and user permissions',
        sprint='Sprint 7',
        fallback_available=True
    ),
    
    # Missing/Planned Components
    'whatif_agent': ComponentInfo(
        name='What-If Agent',
        module_path='components.whatif_agent',
        function_name='render_whatif_agent',
        status=ComponentStatus.ENHANCEMENT,
        description='What-if scenario modeling and sensitivity analysis',
        sprint='Sprint 17',
        fallback_available=True
    ),
    'workflow_navigator': ComponentInfo(
        name='Workflow Navigator',
        module_path='components.workflow_navigator',
        function_name='render_workflow_navigator_simple',
        status=ComponentStatus.OPTIONAL,
        description='Visual workflow progress and navigation',
        sprint='Sprint 16.5',
        fallback_available=True
    ),
    'workflow_guidance': ComponentInfo(
        name='Workflow Guidance',
        module_path='components.workflow_guidance',
        function_name='render_contextual_help',
        status=ComponentStatus.OPTIONAL,
        description='Contextual help and workflow guidance',
        sprint='Sprint 16.5',
        fallback_available=True
    ),
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_required_components() -> List[str]:
    """Get list of required component names"""
    return [
        name for name, info in COMPONENT_REGISTRY.items()
        if info.status == ComponentStatus.REQUIRED
    ]

def get_component_dependencies(component_name: str) -> List[str]:
    """Get dependencies for a component"""
    if component_name not in COMPONENT_REGISTRY:
        return []
    return COMPONENT_REGISTRY[component_name].dependencies

def get_all_dependencies(component_name: str) -> List[str]:
    """Get all transitive dependencies for a component"""
    deps = set()
    to_check = [component_name]
    
    while to_check:
        current = to_check.pop()
        if current in deps:
            continue
        deps.add(current)
        direct_deps = get_component_dependencies(current)
        to_check.extend(direct_deps)
    
    deps.discard(component_name)  # Remove self
    return list(deps)

def verify_component(component_name: str) -> Dict[str, Any]:
    """
    Verify a component exists and is importable.
    
    Returns:
        Dict with 'exists', 'importable', 'function_exists', 'error'
    """
    if component_name not in COMPONENT_REGISTRY:
        return {
            'exists': False,
            'importable': False,
            'function_exists': False,
            'error': f'Component {component_name} not in registry'
        }
    
    info = COMPONENT_REGISTRY[component_name]
    result = {
        'exists': True,
        'importable': False,
        'function_exists': False,
        'error': None
    }
    
    try:
        # Try to import the module
        module_parts = info.module_path.split('.')
        if len(module_parts) == 1:
            mod = __import__(info.module_path)
        else:
            mod = __import__(info.module_path, fromlist=[module_parts[-1]])
        
        result['importable'] = True
        
        # Try to get the function/class
        if hasattr(mod, info.function_name):
            result['function_exists'] = True
        else:
            result['error'] = f'Function {info.function_name} not found in {info.module_path}'
            
    except ImportError as e:
        result['error'] = f'Import error: {str(e)}'
    except Exception as e:
        result['error'] = f'Unexpected error: {str(e)}'
    
    return result

def verify_all_components() -> Dict[str, Dict[str, Any]]:
    """Verify all components in the registry"""
    results = {}
    for name in COMPONENT_REGISTRY:
        results[name] = verify_component(name)
    return results

def get_missing_required_components() -> List[str]:
    """Get list of missing required components"""
    results = verify_all_components()
    missing = []
    
    for name, result in results.items():
        info = COMPONENT_REGISTRY[name]
        if info.status == ComponentStatus.REQUIRED:
            if not result['importable'] or not result['function_exists']:
                missing.append(name)
    
    return missing

def print_component_status():
    """Print status of all components"""
    results = verify_all_components()
    
    print("=" * 80)
    print("COMPONENT REGISTRY STATUS")
    print("=" * 80)
    
    required_missing = []
    optional_missing = []
    
    for name, result in results.items():
        info = COMPONENT_REGISTRY[name]
        status_icon = "✅" if result['importable'] and result['function_exists'] else "❌"
        status_text = info.status.value.upper()
        
        print(f"\n{status_icon} {name} ({status_text})")
        print(f"   Module: {info.module_path}")
        print(f"   Function: {info.function_name}")
        
        if result['importable'] and result['function_exists']:
            print(f"   Status: OK")
        else:
            print(f"   Status: MISSING")
            if result['error']:
                print(f"   Error: {result['error']}")
            
            if info.status == ComponentStatus.REQUIRED:
                required_missing.append(name)
            elif info.status == ComponentStatus.OPTIONAL:
                optional_missing.append(name)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Components: {len(COMPONENT_REGISTRY)}")
    print(f"Required Missing: {len(required_missing)}")
    if required_missing:
        print(f"  ⚠️  {', '.join(required_missing)}")
    print(f"Optional Missing: {len(optional_missing)}")
    if optional_missing:
        print(f"  ℹ️  {', '.join(optional_missing)}")
    
    if required_missing:
        print("\n🚨 CRITICAL: Required components are missing!")
        return False
    else:
        print("\n✅ All required components are present")
        return True

if __name__ == "__main__":
    print_component_status()
