"""
Natural Language Scenario Planner
==================================
Allows users to describe business scenarios in natural language,
which are then parsed and applied to configure all assumptions.

Date: December 20, 2025

Example Scenarios:
- "Aggressive 35% revenue growth with constant margins"
- "Conservative growth with cost reduction focus"
- "Turnaround scenario with declining revenue but improved efficiency"
"""

import streamlit as st
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime


# =============================================================================
# SCENARIO PARAMETERS
# =============================================================================

@dataclass
class ScenarioParameters:
    """Extracted parameters from a scenario description."""
    # Revenue
    revenue_growth_rate: float = 0.0  # Annual %
    revenue_trend_type: str = 'linear'  # linear, exponential, flat
    
    # Margins
    margin_change: float = 0.0  # Change in margin (pp)
    margin_constant: bool = True
    
    # COGS
    cogs_growth_ratio: float = 1.0  # Ratio to revenue growth (1.0 = same)
    
    # Overheads/OPEX
    opex_growth_ratio: float = 1.0  # Ratio to revenue growth
    opex_correlation: float = 0.7   # Correlation with revenue
    
    # Working Capital
    stock_days_change: float = 0.0  # Days change
    stock_growth_premium: float = 0.0  # Extra stock as % of revenue growth
    receivable_days_change: float = 0.0
    payable_days_change: float = 0.0
    
    # Fixed Assets
    capex_as_pct_revenue: float = 0.05
    
    # Monte Carlo
    apply_mc: bool = True
    mc_cv: float = 0.15  # Coefficient of variation
    
    # Scenario metadata
    scenario_name: str = 'Custom'
    scenario_type: str = 'custom'  # aggressive, conservative, turnaround, custom
    confidence_level: str = 'medium'


# =============================================================================
# PREDEFINED SCENARIO TEMPLATES
# =============================================================================

SCENARIO_TEMPLATES = {
    'aggressive_growth': {
        'name': '🚀 Aggressive Growth',
        'description': 'High revenue growth with scaling operations',
        'example': 'Aggressive compound annual revenue growth of 35% with margins remaining constant. Overheads scale at 60% of revenue growth.',
        'params': ScenarioParameters(
            revenue_growth_rate=35.0,
            revenue_trend_type='exponential',
            margin_constant=True,
            cogs_growth_ratio=1.0,
            opex_growth_ratio=0.6,
            opex_correlation=0.75,
            stock_growth_premium=10.0,
            apply_mc=True,
            mc_cv=0.20,
            scenario_name='Aggressive Growth',
            scenario_type='aggressive'
        )
    },
    'conservative_growth': {
        'name': '📈 Conservative Growth',
        'description': 'Steady growth with cost discipline',
        'example': 'Conservative 8% annual growth with tight cost control. Overheads grow at half the rate of revenue. Maintain current working capital ratios.',
        'params': ScenarioParameters(
            revenue_growth_rate=8.0,
            revenue_trend_type='linear',
            margin_constant=True,
            cogs_growth_ratio=1.0,
            opex_growth_ratio=0.5,
            opex_correlation=0.5,
            stock_growth_premium=0.0,
            apply_mc=True,
            mc_cv=0.12,
            scenario_name='Conservative Growth',
            scenario_type='conservative'
        )
    },
    'turnaround': {
        'name': '🔄 Turnaround',
        'description': 'Declining revenue with cost restructuring',
        'example': 'Revenue declining 10% but margins improving through cost cuts. Overheads reduced by 20%. Working capital optimized.',
        'params': ScenarioParameters(
            revenue_growth_rate=-10.0,
            revenue_trend_type='linear',
            margin_change=5.0,
            margin_constant=False,
            cogs_growth_ratio=0.8,
            opex_growth_ratio=-2.0,  # Negative = cutting faster than revenue decline
            opex_correlation=0.3,
            stock_days_change=-10,
            receivable_days_change=-5,
            apply_mc=True,
            mc_cv=0.25,
            scenario_name='Turnaround',
            scenario_type='turnaround'
        )
    },
    'stable': {
        'name': '⚖️ Stable / Maintenance',
        'description': 'Flat revenue with inflation adjustments',
        'example': 'Revenue flat in real terms, growing only at inflation (5%). All costs grow at same rate. Maintain current ratios.',
        'params': ScenarioParameters(
            revenue_growth_rate=5.0,
            revenue_trend_type='linear',
            margin_constant=True,
            cogs_growth_ratio=1.0,
            opex_growth_ratio=1.0,
            opex_correlation=0.6,
            apply_mc=True,
            mc_cv=0.10,
            scenario_name='Stable',
            scenario_type='stable'
        )
    },
    'high_investment': {
        'name': '🏗️ High Investment Phase',
        'description': 'Growth with significant capex and working capital build',
        'example': 'Growing 20% with significant investment. Need 15% more stock than revenue growth. Capex at 10% of revenue for capacity.',
        'params': ScenarioParameters(
            revenue_growth_rate=20.0,
            revenue_trend_type='exponential',
            margin_constant=True,
            cogs_growth_ratio=1.0,
            opex_growth_ratio=0.7,
            opex_correlation=0.7,
            stock_growth_premium=15.0,
            capex_as_pct_revenue=0.10,
            apply_mc=True,
            mc_cv=0.18,
            scenario_name='High Investment',
            scenario_type='aggressive'
        )
    }
}


# =============================================================================
# NATURAL LANGUAGE PARSER
# =============================================================================

def parse_scenario_description(description: str) -> ScenarioParameters:
    """
    Parse a natural language scenario description into parameters.
    
    This uses pattern matching for common phrases. In production,
    this would be enhanced with an LLM for more sophisticated parsing.
    """
    params = ScenarioParameters()
    desc_lower = description.lower()
    
    # ==========================================================================
    # Revenue Growth
    # ==========================================================================
    # Look for percentage patterns
    growth_patterns = [
        r'(\d+(?:\.\d+)?)\s*%\s*(?:annual|yearly|compound|cagr)?\s*(?:revenue\s+)?growth',
        r'revenue\s+growth\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%',
        r'grow(?:ing|th)?\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*%',
        r'growth\s+(?:rate\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*%',
    ]
    
    for pattern in growth_patterns:
        match = re.search(pattern, desc_lower)
        if match:
            params.revenue_growth_rate = float(match.group(1))
            break
    
    # Check for declining revenue
    if any(word in desc_lower for word in ['decline', 'declining', 'decrease', 'decreasing', 'negative', 'shrink']):
        params.revenue_growth_rate = -abs(params.revenue_growth_rate)
    
    # Trend type
    if any(word in desc_lower for word in ['aggressive', 'compound', 'exponential', 'accelerat']):
        params.revenue_trend_type = 'exponential'
    elif any(word in desc_lower for word in ['flat', 'constant', 'stable', 'no growth']):
        params.revenue_trend_type = 'flat'
    else:
        params.revenue_trend_type = 'linear'
    
    # ==========================================================================
    # Margins
    # ==========================================================================
    if any(phrase in desc_lower for phrase in ['margin remain', 'margins remain', 'constant margin', 'margin constant', 'maintain margin']):
        params.margin_constant = True
        params.margin_change = 0.0
    elif 'margin improv' in desc_lower or 'improving margin' in desc_lower:
        params.margin_constant = False
        margin_match = re.search(r'margin\s+improv\w*\s+(?:by\s+)?(\d+(?:\.\d+)?)', desc_lower)
        if margin_match:
            params.margin_change = float(margin_match.group(1))
        else:
            params.margin_change = 2.0  # Default improvement
    
    # ==========================================================================
    # Overheads / OPEX
    # ==========================================================================
    # Look for overhead scaling ratio
    overhead_patterns = [
        r'overhead[s]?\s+(?:scale|grow|increase)\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*%\s+(?:of\s+)?(?:revenue\s+)?growth',
        r'overhead[s]?\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?growth',
        r'opex\s+(?:scale|grow)\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*%',
    ]
    
    for pattern in overhead_patterns:
        match = re.search(pattern, desc_lower)
        if match:
            params.opex_growth_ratio = float(match.group(1)) / 100
            break
    
    # Override if specific language found
    if 'overhead' in desc_lower and 'not scale' in desc_lower:
        params.opex_growth_ratio = 0.25
    elif 'overhead' in desc_lower and 'do not scale proportionate' in desc_lower:
        # Extract the ratio
        match = re.search(r'(\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?growth', desc_lower)
        if match:
            params.opex_growth_ratio = float(match.group(1)) / 100
    elif 'tight cost' in desc_lower or 'cost control' in desc_lower or 'cost discipline' in desc_lower:
        params.opex_growth_ratio = 0.5
    elif 'cost reduction' in desc_lower or 'reduce cost' in desc_lower or 'cut cost' in desc_lower:
        params.opex_growth_ratio = 0.3
    
    # ==========================================================================
    # Stock / Inventory
    # ==========================================================================
    stock_patterns = [
        r'hold\s+(\d+(?:\.\d+)?)\s*%\s+more\s+stock',
        r'stock\s+(\d+(?:\.\d+)?)\s*%\s+(?:more|higher)',
        r'inventory\s+(\d+(?:\.\d+)?)\s*%\s+(?:more|higher)',
    ]
    
    for pattern in stock_patterns:
        match = re.search(pattern, desc_lower)
        if match:
            params.stock_growth_premium = float(match.group(1))
            break
    
    # Stock days
    stock_days_match = re.search(r'stock\s+days?\s+(?:to\s+)?(\d+)', desc_lower)
    if stock_days_match:
        params.stock_days_change = float(stock_days_match.group(1))
    
    # ==========================================================================
    # Supplier Terms
    # ==========================================================================
    if any(phrase in desc_lower for phrase in ['supplier', 'terms remain', 'terms constant', 'payable']):
        if 'remain constant' in desc_lower or 'terms remain' in desc_lower:
            params.payable_days_change = 0.0
    
    # ==========================================================================
    # Scenario Type Classification
    # ==========================================================================
    if params.revenue_growth_rate >= 25:
        params.scenario_type = 'aggressive'
        params.scenario_name = 'Aggressive Growth'
        params.mc_cv = 0.20
    elif params.revenue_growth_rate < 0:
        params.scenario_type = 'turnaround'
        params.scenario_name = 'Turnaround'
        params.mc_cv = 0.25
    elif params.revenue_growth_rate < 10:
        params.scenario_type = 'conservative'
        params.scenario_name = 'Conservative Growth'
        params.mc_cv = 0.12
    else:
        params.scenario_type = 'growth'
        params.scenario_name = 'Growth Scenario'
        params.mc_cv = 0.15
    
    return params


# =============================================================================
# APPLY SCENARIO TO LINE ITEMS
# =============================================================================

# Industry default correlations (from ai_assumptions_engine)
INDUSTRY_DEFAULTS = {
    'Personnel': {'correlation': 0.80, 'growth_ratio': 0.7},
    'Admin': {'correlation': 0.50, 'growth_ratio': 0.5},
    'Facilities': {'correlation': 0.35, 'growth_ratio': 0.3},
    'Sales & Marketing': {'correlation': 0.75, 'growth_ratio': 0.8},
    'IT & Technology': {'correlation': 0.45, 'growth_ratio': 0.4},
    'Travel & Entertainment': {'correlation': 0.75, 'growth_ratio': 0.7},
    'Depreciation': {'correlation': 0.55, 'growth_ratio': 0.5},
    'Insurance': {'correlation': 0.25, 'growth_ratio': 0.2},
    'Materials': {'correlation': 0.92, 'growth_ratio': 1.0},
    'Direct Labour': {'correlation': 0.88, 'growth_ratio': 0.95},
    'Direct Costs': {'correlation': 0.90, 'growth_ratio': 1.0},
    'Other Operating': {'correlation': 0.60, 'growth_ratio': 0.6},
}


def apply_scenario_to_config(config, params: ScenarioParameters) -> Tuple[int, List[str]]:
    """
    Apply scenario parameters to the unified config.
    
    Returns:
        Tuple of (items_updated, list of changes made)
    """
    changes = []
    count = 0
    
    for key, item in config.line_items.items():
        category = item.category.lower()
        sub_group = getattr(item, 'sub_group', '') or ''
        sub_group_lower = sub_group.lower()
        name_lower = item.line_item_name.lower()
        
        # =======================================================================
        # Revenue Items
        # =======================================================================
        if 'revenue' in category or 'sales' in category or 'income' in category:
            item.trend_type = params.revenue_trend_type
            item.trend_growth_rate = params.revenue_growth_rate
            item.use_distribution = params.apply_mc
            item.distribution_cv = params.mc_cv
            item.correlate_with_revenue = True
            item.revenue_correlation = 1.0
            count += 1
            changes.append(f"Revenue: {params.revenue_growth_rate:.0f}% {params.revenue_trend_type} growth")
        
        # =======================================================================
        # COGS Items
        # =======================================================================
        elif 'cogs' in category or 'cost of' in category or 'direct' in category:
            # COGS scales with revenue * ratio
            cogs_growth = params.revenue_growth_rate * params.cogs_growth_ratio
            item.trend_type = params.revenue_trend_type
            item.trend_growth_rate = cogs_growth
            item.use_distribution = params.apply_mc
            item.distribution_cv = params.mc_cv * 0.8  # Lower variance for COGS
            item.correlate_with_revenue = True
            item.revenue_correlation = 0.92
            count += 1
        
        # =======================================================================
        # OPEX / Overhead Items
        # =======================================================================
        elif 'opex' in category or 'operating' in category or 'expense' in category or 'overhead' in category:
            # Get sub-group specific settings or use scenario default
            defaults = INDUSTRY_DEFAULTS.get(sub_group, {'correlation': params.opex_correlation, 'growth_ratio': params.opex_growth_ratio})
            
            # Apply scenario override for overhead growth ratio
            growth_ratio = params.opex_growth_ratio if params.opex_growth_ratio != 1.0 else defaults['growth_ratio']
            opex_growth = params.revenue_growth_rate * growth_ratio
            
            item.trend_type = 'linear'  # OPEX usually linear
            item.trend_growth_rate = opex_growth
            item.use_distribution = params.apply_mc
            item.distribution_cv = params.mc_cv
            item.correlate_with_revenue = True
            item.revenue_correlation = defaults['correlation']
            count += 1
            
            if sub_group:
                changes.append(f"{sub_group}: {opex_growth:.1f}% growth, {defaults['correlation']:.0%} correlation")
        
        # =======================================================================
        # Working Capital Items
        # =======================================================================
        elif 'working capital' in category or getattr(item, 'is_working_capital', False):
            # Stock/Inventory
            if 'stock' in name_lower or 'inventory' in name_lower:
                # Stock grows at revenue + premium
                stock_growth = params.revenue_growth_rate + params.stock_growth_premium
                item.trend_type = 'linear'
                item.trend_growth_rate = stock_growth
                item.correlate_with_revenue = True
                item.revenue_correlation = 0.90
                item.use_distribution = params.apply_mc
                count += 1
                changes.append(f"Stock: {stock_growth:.1f}% growth (revenue + {params.stock_growth_premium:.0f}% premium)")
            
            # Receivables
            elif 'receivable' in name_lower:
                item.trend_growth_rate = params.revenue_growth_rate
                item.correlate_with_revenue = True
                item.revenue_correlation = 0.95
                if params.receivable_days_change != 0:
                    item.days_metric = getattr(item, 'days_metric', 45) + params.receivable_days_change
                count += 1
            
            # Payables
            elif 'payable' in name_lower:
                item.trend_growth_rate = params.revenue_growth_rate * params.cogs_growth_ratio
                item.correlate_with_revenue = True
                item.revenue_correlation = 0.90
                if params.payable_days_change != 0:
                    item.days_metric = getattr(item, 'days_metric', 30) + params.payable_days_change
                count += 1
        
        # =======================================================================
        # Fixed Assets
        # =======================================================================
        elif 'fixed' in category or 'asset' in category:
            # Fixed assets grow slower
            item.trend_type = 'flat' if params.revenue_growth_rate < 10 else 'linear'
            item.trend_growth_rate = params.revenue_growth_rate * 0.3
            item.correlate_with_revenue = True
            item.revenue_correlation = 0.35
            count += 1
    
    return count, changes


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_scenario_planner(db, scenario_id: str, user_id: str, config):
    """Render the natural language scenario planner UI."""
    st.markdown("### 🎯 Scenario Planner")
    st.caption("Describe your business scenario in natural language and we'll configure all assumptions")

    # ---------------------------------------------------------------------
    # Persist scenario selection + text to DB so it survives navigation/sessions.
    # ---------------------------------------------------------------------
    scenario_input_key = f"scenario_input_{scenario_id}"
    selected_template_key = f"scenario_template_{scenario_id}"

    saved_state: Dict[str, Any] = {}
    try:
        if hasattr(db, "get_scenario_assumptions"):
            _assum = db.get_scenario_assumptions(scenario_id, user_id) or {}
            saved_state = (_assum.get("scenario_planner") or {}) if isinstance(_assum, dict) else {}
    except Exception:
        saved_state = {}

    # Initialize session state from DB on first render
    try:
        if selected_template_key not in st.session_state and isinstance(saved_state, dict) and saved_state.get("selected_template"):
            st.session_state[selected_template_key] = saved_state.get("selected_template")
        if scenario_input_key not in st.session_state and isinstance(saved_state, dict) and saved_state.get("scenario_text"):
            st.session_state[scenario_input_key] = saved_state.get("scenario_text")
    except Exception:
        pass

    def _save_planner_state(text: str, template_key: Optional[str], params: Optional[ScenarioParameters] = None) -> None:
        try:
            if not hasattr(db, "get_scenario_assumptions"):
                return
            assumptions = db.get_scenario_assumptions(scenario_id, user_id) or {}
            if not isinstance(assumptions, dict):
                return
            payload: Dict[str, Any] = {
                "scenario_text": str(text or ""),
                "selected_template": template_key,
                "updated_at": datetime.utcnow().isoformat(),
            }
            if params is not None:
                try:
                    payload["parsed_params"] = asdict(params)
                except Exception:
                    payload["parsed_params"] = {}
            assumptions["scenario_planner"] = payload
            db.update_assumptions(scenario_id, user_id, assumptions)
        except Exception:
            return
    
    # Template selection
    st.markdown("#### Quick Templates")
    
    cols = st.columns(len(SCENARIO_TEMPLATES))
    for i, (key, template) in enumerate(SCENARIO_TEMPLATES.items()):
        with cols[i]:
            if st.button(template['name'], key=f'template_{key}', use_container_width=True):
                st.session_state[scenario_input_key] = template['example']
                st.session_state[selected_template_key] = key
                _save_planner_state(template['example'], key, None)
                st.rerun()
    
    # Show selected template description
    if selected_template_key in st.session_state:
        template = SCENARIO_TEMPLATES.get(st.session_state.get(selected_template_key))
        if template:
            st.info(f"**{template['name']}:** {template['description']}")
    
    st.markdown("---")
    st.markdown("#### Describe Your Scenario")
    
    # Text input for scenario
    default_text = st.session_state.get(scenario_input_key, '') or (saved_state.get("scenario_text", "") if isinstance(saved_state, dict) else "")
    scenario_text = st.text_area(
        "Scenario Description",
        value=default_text,
        height=150,
        placeholder="""Example: "Aggressive compound annual revenue growth of 35% with margins remaining constant. 
Overheads do not scale proportionate to revenue but rather at 25% of the growth in revenue. 
To achieve such growth we will need to hold 10% more stock than what revenue growth is set at. 
Terms with suppliers will remain constant.""",
        key=scenario_input_key
    )
    
    # Parse and preview
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔍 Preview Parameters", use_container_width=True, key='preview_scenario'):
            if scenario_text:
                params = parse_scenario_description(scenario_text)
                st.session_state['parsed_params'] = params
                _save_planner_state(scenario_text, st.session_state.get(selected_template_key), params)
    
    with col2:
        if st.button("🚀 Apply Scenario", type='primary', use_container_width=True, key='apply_scenario'):
            if scenario_text:
                params = parse_scenario_description(scenario_text)
                count, changes = apply_scenario_to_config(config, params)
                st.session_state['scenario_applied'] = {
                    'count': count,
                    'changes': changes,
                    'params': params
                }
                _save_planner_state(scenario_text, st.session_state.get(selected_template_key), params)
                st.success(f"✅ Applied scenario to {count} line items!")
                st.rerun()
    
    # Show parsed parameters
    if 'parsed_params' in st.session_state:
        params = st.session_state['parsed_params']
        
        st.markdown("#### 📊 Extracted Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Revenue**")
            st.metric("Growth Rate", f"{params.revenue_growth_rate:.1f}%")
            st.caption(f"Trend: {params.revenue_trend_type}")
        
        with col2:
            st.markdown("**Costs**")
            st.metric("OPEX Growth Ratio", f"{params.opex_growth_ratio:.0%}")
            st.caption(f"OPEX grows at {params.opex_growth_ratio:.0%} of revenue growth")
        
        with col3:
            st.markdown("**Working Capital**")
            st.metric("Stock Premium", f"+{params.stock_growth_premium:.0f}%")
            st.caption(f"Stock grows {params.stock_growth_premium:.0f}% more than revenue")
        
        with st.expander("📋 All Parameters", expanded=False):
            param_data = {
                'Parameter': [
                    'Revenue Growth Rate',
                    'Revenue Trend Type',
                    'Margin Constant',
                    'COGS Growth Ratio',
                    'OPEX Growth Ratio',
                    'OPEX Correlation',
                    'Stock Growth Premium',
                    'MC Enabled',
                    'MC CV',
                    'Scenario Type'
                ],
                'Value': [
                    f"{params.revenue_growth_rate:.1f}%",
                    params.revenue_trend_type,
                    '✓' if params.margin_constant else '✗',
                    f"{params.cogs_growth_ratio:.0%}",
                    f"{params.opex_growth_ratio:.0%}",
                    f"{params.opex_correlation:.0%}",
                    f"+{params.stock_growth_premium:.0f}%",
                    '✓' if params.apply_mc else '✗',
                    f"{params.mc_cv:.0%}",
                    params.scenario_type
                ]
            }
            st.dataframe(param_data, hide_index=True, use_container_width=True)
    
    # Show what was applied
    if 'scenario_applied' in st.session_state:
        applied = st.session_state['scenario_applied']
        
        st.markdown("---")
        st.markdown("#### ✅ Scenario Applied")
        st.success(f"Updated {applied['count']} line items based on '{applied['params'].scenario_name}' scenario")
        
        if applied['changes']:
            with st.expander("📋 Key Changes Made", expanded=True):
                for change in applied['changes'][:10]:  # Show first 10
                    st.write(f"• {change}")
                if len(applied['changes']) > 10:
                    st.caption(f"... and {len(applied['changes']) - 10} more")
        
        st.info("💡 Review the line items table above to fine-tune individual items, then **Save All Configuration**.")
