"""
Command Center Component
========================
Dashboard showing scenario health, setup status, and quick navigation.

Version: 6.0 (December 2025)
Changes: Added fallback to installed_base table when sites/machine_instances are empty
"""

import streamlit as st
from typing import Dict, Any, Optional, Callable
from datetime import datetime


# =============================================================================
# UTILITY FUNCTIONS  
# =============================================================================

def format_currency(value: float, prefix: str = 'R') -> str:
    """Format currency with smart abbreviation."""
    if value is None:
        return f"{prefix}0"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"{prefix}{value/1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"{prefix}{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{prefix}{value/1_000:.0f}K"
    else:
        return f"{prefix}{value:,.0f}"


def get_grade_color(grade: str) -> str:
    """Return color for health grade."""
    colors = {
        'A': '#10b981',  # Green
        'B': '#22c55e',  # Light Green
        'C': '#f59e0b',  # Amber
        'D': '#f97316',  # Orange
        'F': '#ef4444',  # Red
    }
    return colors.get(grade, '#64748b')


# =============================================================================
# SETUP HEALTH CALCULATION
# =============================================================================

def calculate_setup_health(db, scenario_id: str, user_id: str) -> Dict[str, Any]:
    """
    Calculate setup completeness score and identify missing items.
    
    Includes fallback logic to check installed_base table when 
    sites/machine_instances tables are empty.
    
    Returns:
        Dict with: score, grade, checks, actions_needed, stats, using_legacy_fleet
    """
    checks = {}
    actions_needed = []
    stats = {}
    using_legacy_fleet = False
    
    # 1. Assumptions (20 points)
    try:
        assumptions = db.get_scenario_assumptions(scenario_id, user_id)
        has_assumptions = bool(assumptions and assumptions.get('wacc'))
        checks['assumptions'] = has_assumptions
        stats['assumptions'] = 'Configured' if has_assumptions else 'Not set'
        
        if not has_assumptions:
            actions_needed.append({
                'item': 'Set Global Assumptions',
                'description': 'Configure WACC, margins, and forecast duration',
                'section': 'setup',
                'step': 'basics',
                'priority': 'high'
            })
    except Exception as e:
        checks['assumptions'] = False
        stats['assumptions'] = 'Error'
    
    # 2. Customers (15 points)
    customer_count = 0
    try:
        customers = db.get_customers(user_id)
        customer_count = len(customers) if customers else 0
        checks['customers'] = customer_count > 0
        stats['customers'] = customer_count
        
        if customer_count == 0:
            actions_needed.append({
                'item': 'Add Customers',
                'description': 'Import or add customer records',
                'section': 'setup',
                'step': 'customers',
                'priority': 'high'
            })
    except Exception as e:
        checks['customers'] = False
        stats['customers'] = 0
    
    # 3. Sites (10 points) - WITH FALLBACK TO installed_base
    site_count = 0
    try:
        sites = db.get_sites(user_id)
        site_count = len(sites) if sites else 0
        
        # FALLBACK: If sites table is empty, check installed_base for unique site_names
        if site_count == 0:
            try:
                result = db.client.table('installed_base').select('site_name').eq('scenario_id', scenario_id).execute()
                if result.data:
                    unique_sites = set(r.get('site_name') for r in result.data if r.get('site_name'))
                    if unique_sites:
                        site_count = len(unique_sites)
                        using_legacy_fleet = True
            except Exception:
                pass  # installed_base may not exist
        
        checks['sites'] = site_count > 0
        stats['sites'] = site_count
        
        if site_count == 0 and customer_count > 0:
            actions_needed.append({
                'item': 'Add Sites',
                'description': 'Define customer site locations',
                'section': 'setup',
                'step': 'fleet',
                'priority': 'medium'
            })
    except Exception as e:
        checks['sites'] = False
        stats['sites'] = 0
    
    # 4. Machine Instances (20 points) - WITH FALLBACK TO installed_base
    machine_count = 0
    try:
        machines = db.get_machine_instances(user_id, scenario_id)
        machine_count = len(machines) if machines else 0
        
        # FALLBACK: If machine_instances is empty, check installed_base
        if machine_count == 0:
            try:
                result = db.client.table('installed_base').select('id').eq('scenario_id', scenario_id).execute()
                if result.data:
                    machine_count = len(result.data)
                    using_legacy_fleet = True
            except Exception:
                pass  # installed_base may not exist
        
        checks['machines'] = machine_count > 0
        stats['machines'] = machine_count
        
        if machine_count == 0:
            actions_needed.append({
                'item': 'Add Fleet',
                'description': 'Import or add machine instances',
                'section': 'setup',
                'step': 'fleet',
                'priority': 'high'
            })
    except Exception as e:
        checks['machines'] = False
        stats['machines'] = 0
    
    # 5. Wear Profiles (20 points)
    try:
        # Try new wear profiles first
        profiles = None
        if hasattr(db, 'get_wear_profiles_v2'):
            profiles = db.get_wear_profiles_v2(user_id)
        if not profiles:
            # Fall back to legacy profiles
            profiles_dict = db.get_wear_profiles(user_id)
            profile_count = len(profiles_dict) if profiles_dict else 0
        else:
            profile_count = len(profiles)
        
        checks['wear_profiles'] = profile_count > 0
        stats['wear_profiles'] = profile_count
        
        if profile_count == 0:
            actions_needed.append({
                'item': 'Configure Wear Profiles',
                'description': 'Set liner life and revenue per machine model',
                'section': 'setup',
                'step': 'fleet',
                'priority': 'high'
            })
    except Exception as e:
        checks['wear_profiles'] = False
        stats['wear_profiles'] = 0
    
    # 6. Pipeline/Prospects (10 points - optional but good to have)
    prospect_count = 0
    try:
        prospects = db.get_prospects(user_id, scenario_id)
        prospect_count = len(prospects) if prospects else 0
        checks['pipeline'] = prospect_count > 0
        stats['pipeline'] = prospect_count
        
        # Pipeline is optional - only suggest if they have fleet but no pipeline
        if prospect_count == 0 and machine_count > 0:
            actions_needed.append({
                'item': 'Add Pipeline Prospects',
                'description': 'Track sales opportunities for forecasting',
                'section': 'setup',
                'step': 'pipeline',
                'priority': 'low'
            })
    except Exception as e:
        checks['pipeline'] = False
        stats['pipeline'] = 0
    
    # 7. Expense Assumptions (5 points - optional)
    try:
        expense_count = 0
        if hasattr(db, 'get_expense_assumptions'):
            expenses = db.get_expense_assumptions(scenario_id)
            expense_count = len(expenses) if expenses else 0
        
        checks['expenses'] = expense_count > 0
        stats['expenses'] = expense_count
        
        if expense_count == 0 and checks['assumptions']:
            actions_needed.append({
                'item': 'Configure Expenses',
                'description': 'Set up OPEX functions for forecasting',
                'section': 'setup',
                'step': 'costs',
                'priority': 'low'
            })
    except Exception as e:
        checks['expenses'] = False
        stats['expenses'] = 0
    
    # Calculate weighted score
    weights = {
        'assumptions': 20,
        'customers': 15,
        'sites': 10,
        'machines': 20,
        'wear_profiles': 20,
        'pipeline': 10,
        'expenses': 5
    }
    
    score = sum(weights[k] for k, v in checks.items() if v)
    
    # Determine grade
    if score >= 90:
        grade = 'A'
    elif score >= 75:
        grade = 'B'
    elif score >= 60:
        grade = 'C'
    elif score >= 40:
        grade = 'D'
    else:
        grade = 'F'
    
    # Sort actions by priority
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    actions_needed.sort(key=lambda x: priority_order.get(x['priority'], 3))
    
    return {
        'score': score,
        'grade': grade,
        'checks': checks,
        'actions_needed': actions_needed,
        'stats': stats,
        'using_legacy_fleet': using_legacy_fleet
    }


# =============================================================================
# FORECAST SUMMARY
# =============================================================================

def get_forecast_summary(db, scenario_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Load latest forecast snapshot summary."""
    try:
        if hasattr(db, 'get_forecast_snapshots'):
            snapshots = db.get_forecast_snapshots(user_id, scenario_id)
            if snapshots:
                latest = snapshots[0]  # Already sorted by created_at desc
                return {
                    'name': latest.get('snapshot_name', 'Unnamed'),
                    'created_at': latest.get('created_at'),
                    'total_revenue': latest.get('total_revenue_forecast', 0) or latest.get('total_revenue', 0),
                    'gross_profit': latest.get('total_gross_profit_forecast', 0) or latest.get('gross_profit', 0),
                    'ebit': latest.get('ebit', 0),
                    'summary_stats': latest.get('summary_stats', {}),
                    'has_monte_carlo': latest.get('valuation_data') is not None
                }
        return None
    except Exception as e:
        return None


# =============================================================================
# PIPELINE SUMMARY
# =============================================================================

def get_pipeline_summary(db, scenario_id: str, user_id: str) -> Dict[str, Any]:
    """Aggregate pipeline prospects by stage."""
    try:
        prospects = db.get_prospects(user_id, scenario_id)
        
        if not prospects:
            return {'total': 0, 'stages': {}, 'weighted_value': 0}
        
        stages = {}
        total_value = 0
        weighted_value = 0
        
        for p in prospects:
            stage = p.get('pipeline_stage', 'lead')
            stage_key = stage.lower()
            
            if stage_key not in stages:
                stages[stage_key] = {'count': 0, 'value': 0, 'weighted': 0}
            
            # Annual liner value
            annual_value = float(p.get('annual_liner_value', 0) or 0)
            # Confidence is stored as decimal (0.65 = 65%)
            confidence = float(p.get('confidence_pct', 0) or 0)
            
            stages[stage_key]['count'] += 1
            stages[stage_key]['value'] += annual_value
            stages[stage_key]['weighted'] += annual_value * confidence
            
            total_value += annual_value
            weighted_value += annual_value * confidence
        
        return {
            'total': len(prospects),
            'stages': stages,
            'total_value': total_value,
            'weighted_value': weighted_value
        }
    except Exception as e:
        return {'total': 0, 'stages': {}, 'weighted_value': 0}


# =============================================================================
# RENDER COMPONENTS
# =============================================================================

def render_health_score(health: Dict[str, Any]):
    """Render the setup health score gauge."""
    score = health['score']
    grade = health['grade']
    color = get_grade_color(grade)
    
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, {color}22, {color}11);
        border-radius: 1rem;
        border: 1px solid {color}44;
    ">
        <div style="font-size: 4rem; font-weight: bold; color: {color};">{score}</div>
        <div style="font-size: 1.5rem; color: {color}; margin-bottom: 0.5rem;">Grade: {grade}</div>
        <div style="font-size: 0.875rem; opacity: 0.7;">Setup Health Score</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show legacy fleet notice if applicable
    if health.get('using_legacy_fleet'):
        st.info("📋 Using imported fleet data from installed_base table")


def render_setup_checklist(health: Dict[str, Any]):
    """Render the setup checklist with counts."""
    st.markdown("#### ✅ Setup Checklist")
    
    checks = health['checks']
    stats = health['stats']
    
    items = [
        ('assumptions', '⚙️ Assumptions', stats.get('assumptions', 'Not set')),
        ('customers', '👥 Customers', stats.get('customers', 0)),
        ('sites', '📍 Sites', stats.get('sites', 0)),
        ('machines', '🏭 Machines', stats.get('machines', 0)),
        ('wear_profiles', '⚡ Wear Profiles', stats.get('wear_profiles', 0)),
        ('pipeline', '🎯 Pipeline', stats.get('pipeline', 0)),
        ('expenses', '💵 Expenses', stats.get('expenses', 0)),
    ]
    
    for key, label, count in items:
        is_done = checks.get(key, False)
        icon = "✅" if is_done else "⬜"
        count_str = str(count) if isinstance(count, int) else count
        
        st.markdown(f"{icon} **{label}**: {count_str}")


def render_actions_needed(health: Dict[str, Any], on_navigate: Callable = None):
    """Render priority action items."""
    actions = health['actions_needed']
    
    if not actions:
        st.success("🎉 All setup complete! Ready to forecast.")
        return
    
    st.markdown("#### ⚡ Actions Needed")
    
    for action in actions:
        priority = action['priority']
        priority_colors = {
            'high': '#ef4444',
            'medium': '#f59e0b',
            'low': '#3b82f6'
        }
        color = priority_colors.get(priority, '#64748b')
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"""
            <div style="
                padding: 0.75rem;
                background: {color}11;
                border-left: 3px solid {color};
                border-radius: 0.25rem;
                margin-bottom: 0.5rem;
            ">
                <div style="font-weight: 600;">{action['item']}</div>
                <div style="font-size: 0.875rem; opacity: 0.7;">{action['description']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if on_navigate:
                if st.button("Fix →", key=f"fix_{action['item']}", use_container_width=True):
                    on_navigate(action['section'], action['step'])


def render_quick_nav_cards(on_navigate: Callable = None):
    """Render quick navigation cards."""
    st.markdown("#### 🚀 Quick Navigation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⚙️ Setup Wizard", use_container_width=True, type="secondary"):
            if on_navigate:
                on_navigate('setup', 'basics')
    
    with col2:
        if st.button("📊 Run Forecast", use_container_width=True, type="secondary"):
            if on_navigate:
                on_navigate('forecast', None)
    
    with col3:
        if st.button("🔄 Compare Scenarios", use_container_width=True, type="secondary"):
            if on_navigate:
                on_navigate('compare', None)


def render_forecast_preview(summary: Optional[Dict[str, Any]], on_navigate: Callable = None):
    """Render forecast snapshot preview."""
    st.markdown("#### 📈 Latest Forecast")
    
    if not summary:
        st.info("No forecast snapshots yet. Run a forecast to see results here.")
        if on_navigate:
            if st.button("▶️ Run Forecast Now", type="primary"):
                on_navigate('forecast', None)
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Revenue", format_currency(summary.get('total_revenue', 0)))
    
    with col2:
        st.metric("Gross Profit", format_currency(summary.get('gross_profit', 0)))
    
    with col3:
        st.metric("EBIT", format_currency(summary.get('ebit', 0)))
    
    # Snapshot info
    created = summary.get('created_at', '')
    if created:
        try:
            dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
            st.caption(f"📅 {summary.get('name', 'Latest')} • {dt.strftime('%Y-%m-%d %H:%M')}")
        except:
            st.caption(f"📅 {summary.get('name', 'Latest')}")
    
    if summary.get('has_monte_carlo'):
        st.caption("🎲 Monte Carlo analysis included")


def render_pipeline_preview(summary: Dict[str, Any]):
    """Render pipeline summary."""
    st.markdown("#### 🎯 Pipeline Overview")
    
    if summary['total'] == 0:
        st.info("No prospects in pipeline yet.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Prospects", summary['total'])
    
    with col2:
        st.metric("Weighted Value", format_currency(summary.get('weighted_value', 0)))
    
    # Stage breakdown
    stages = summary.get('stages', {})
    if stages:
        stage_names = {
            'lead': '🔵 Lead',
            'qualified': '🟡 Qualified',
            'proposal': '🟠 Proposal',
            'negotiation': '🟢 Negotiation'
        }
        
        for stage_key, stage_data in stages.items():
            name = stage_names.get(stage_key, stage_key.title())
            st.caption(f"{name}: {stage_data['count']} ({format_currency(stage_data['weighted'])})")


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_command_center(
    db,
    scenario_id: str,
    user_id: str,
    on_navigate: Callable = None
):
    """
    Main render function for Command Center.
    
    Args:
        db: SupabaseHandler instance
        scenario_id: Current scenario UUID
        user_id: Current user UUID  
        on_navigate: Callback function(section, step) for navigation
    """
    # Branded header
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div style="flex: 0 0 auto; margin-right: 1rem;">
            <span style="font-size: 2rem;">🏠</span>
        </div>
        <div style="flex: 1;">
            <h1 style="color: #D4A537; margin: 0; font-size: 1.75rem; font-weight: 700;">Command Center</h1>
            <p style="color: #B0B0B0; margin: 0; font-size: 0.875rem;">Crusher Equipment Africa - Scenario Health Dashboard</p>
        </div>
        <div style="flex: 0 0 auto;">
            <div style="display: flex; align-items: center;">
                <div style="width: 40px; height: 2px; background: linear-gradient(90deg, transparent, #D4A537);"></div>
                <span style="color: #D4A537; padding: 0 0.5rem;">⚙️</span>
                <div style="width: 40px; height: 2px; background: linear-gradient(90deg, #D4A537, transparent);"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate setup health (with fallback logic)
    health = calculate_setup_health(db, scenario_id, user_id)
    
    # Load additional data
    forecast_summary = get_forecast_summary(db, scenario_id, user_id)
    pipeline_summary = get_pipeline_summary(db, scenario_id, user_id)
    
    # Layout
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        render_health_score(health)
        st.markdown("")
        render_setup_checklist(health)
    
    with col_right:
        render_actions_needed(health, on_navigate)
        st.markdown("---")
        render_quick_nav_cards(on_navigate)
    
    st.markdown("---")
    
    # Bottom row
    col_forecast, col_pipeline = st.columns(2)
    
    with col_forecast:
        render_forecast_preview(forecast_summary, on_navigate)
    
    with col_pipeline:
        render_pipeline_preview(pipeline_summary)
