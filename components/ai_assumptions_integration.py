"""
AI Assumptions Integration Component
====================================
Enhanced UI component for displaying and managing AI assumptions integration
across financial statements, forecasts, and scenario comparisons.

Version: 1.0
Date: December 15, 2025
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


# =============================================================================
# AI ASSUMPTIONS STATUS INDICATORS
# =============================================================================

def render_ai_assumption_badge(assumption_source: str, assumption_value: Any = None) -> None:
    """
    Render a badge indicating the source of an assumption (AI/Manual/Default).
    
    Args:
        assumption_source: Source of assumption ('ai', 'manual', 'default')
        assumption_value: Optional value to display
    """
    if assumption_source == 'ai':
        st.markdown(
            f"""
            <span style="
                display: inline-flex;
                align-items: center;
                padding: 0.25rem 0.5rem;
                background-color: rgba(59, 130, 246, 0.15);
                border: 1px solid #3B82F6;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 500;
                color: #60A5FA;
                margin-left: 0.5rem;
            ">
                🤖 AI
            </span>
            """,
            unsafe_allow_html=True
        )
    elif assumption_source == 'manual':
        st.markdown(
            f"""
            <span style="
                display: inline-flex;
                align-items: center;
                padding: 0.25rem 0.5rem;
                background-color: rgba(212, 165, 55, 0.15);
                border: 1px solid #D4A537;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 500;
                color: #D4A537;
                margin-left: 0.5rem;
            ">
                ✏️ Manual
            </span>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <span style="
                display: inline-flex;
                align-items: center;
                padding: 0.25rem 0.5rem;
                background-color: rgba(113, 113, 122, 0.15);
                border: 1px solid #71717A;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 500;
                color: #A1A1AA;
                margin-left: 0.5rem;
            ">
                ⚙️ Default
            </span>
            """,
            unsafe_allow_html=True
        )


def render_ai_assumptions_summary(
    assumptions_data: Dict[str, Any],
    show_details: bool = False
) -> None:
    """
    Render a summary card showing AI assumptions status.
    
    Args:
        assumptions_data: Dictionary containing assumption sources and values
        show_details: Whether to show detailed breakdown
    """
    if not assumptions_data:
        st.info("No AI assumptions data available.")
        return
    
    # Count assumptions by source
    ai_count = sum(1 for v in assumptions_data.values() if isinstance(v, dict) and v.get('source') == 'ai')
    manual_count = sum(1 for v in assumptions_data.values() if isinstance(v, dict) and v.get('source') == 'manual')
    default_count = sum(1 for v in assumptions_data.values() if isinstance(v, dict) and v.get('source') == 'default')
    total = len(assumptions_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assumptions", total)
    
    with col2:
        st.metric("🤖 AI Derived", ai_count, delta=f"{ai_count/total*100:.0f}%" if total > 0 else "0%")
    
    with col3:
        st.metric("✏️ Manual", manual_count, delta=f"{manual_count/total*100:.0f}%" if total > 0 else "0%")
    
    with col4:
        st.metric("⚙️ Default", default_count, delta=f"{default_count/total*100:.0f}%" if total > 0 else "0%")
    
    if show_details and assumptions_data:
        with st.expander("📊 Detailed Assumptions Breakdown", expanded=False):
            assumptions_df = pd.DataFrame([
                {
                    'Assumption': key,
                    'Value': v.get('value', v) if isinstance(v, dict) else v,
                    'Source': v.get('source', 'unknown') if isinstance(v, dict) else 'unknown',
                    'Distribution': v.get('distribution', 'N/A') if isinstance(v, dict) else 'N/A'
                }
                for key, v in assumptions_data.items()
            ])
            st.dataframe(assumptions_df, use_container_width=True, hide_index=True)


# =============================================================================
# AI ASSUMPTIONS COMPARISON
# =============================================================================

def render_assumptions_comparison(
    assumptions_list: List[Dict[str, Any]],
    scenario_names: List[str]
) -> None:
    """
    Render a comparison table of assumptions across multiple scenarios.
    
    Args:
        assumptions_list: List of assumption dictionaries (one per scenario)
        scenario_names: List of scenario names
    """
    if not assumptions_list or len(assumptions_list) != len(scenario_names):
        st.warning("Invalid assumptions data for comparison.")
        return
    
    # Get all unique assumption keys
    all_keys = set()
    for assumptions in assumptions_list:
        all_keys.update(assumptions.keys())
    
    # Build comparison DataFrame
    comparison_data = []
    for key in sorted(all_keys):
        row = {'Assumption': key}
        for i, (assumptions, name) in enumerate(zip(assumptions_list, scenario_names)):
            value = assumptions.get(key, {})
            if isinstance(value, dict):
                row[f'{name} (Value)'] = value.get('value', 'N/A')
                row[f'{name} (Source)'] = value.get('source', 'unknown')
            else:
                row[f'{name} (Value)'] = value
                row[f'{name} (Source)'] = 'unknown'
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)


# =============================================================================
# AI ASSUMPTIONS DISTRIBUTION VISUALIZATION
# =============================================================================

def render_distribution_info(
    distribution_params: Dict[str, Any],
    assumption_name: str
) -> None:
    """
    Render information about a probability distribution for an assumption.
    
    Args:
        distribution_params: Dictionary containing distribution parameters
        assumption_name: Name of the assumption
    """
    if not distribution_params:
        return
    
    dist_type = distribution_params.get('type', 'unknown')
    params = distribution_params.get('params', {})
    
    st.markdown(f"**{assumption_name} Distribution:**")
    
    if dist_type == 'normal':
        st.caption(f"Normal: μ={params.get('mean', 0):.2f}, σ={params.get('std', 0):.2f}")
    elif dist_type == 'lognormal':
        st.caption(f"Lognormal: μ={params.get('mean', 0):.2f}, σ={params.get('std', 0):.2f}")
    elif dist_type == 'triangular':
        st.caption(f"Triangular: min={params.get('min', 0):.2f}, mode={params.get('mode', 0):.2f}, max={params.get('max', 0):.2f}")
    elif dist_type == 'beta':
        st.caption(f"Beta: α={params.get('alpha', 0):.2f}, β={params.get('beta', 0):.2f}")
    elif dist_type == 'pert':
        st.caption(f"PERT: min={params.get('min', 0):.2f}, mode={params.get('mode', 0):.2f}, max={params.get('max', 0):.2f}")
    else:
        st.caption(f"Distribution type: {dist_type}")


# =============================================================================
# AI ASSUMPTIONS INTEGRATION HELPER
# =============================================================================

def get_assumption_with_source(
    ai_assumptions: Optional[Any],
    assumption_key: str,
    manual_value: Any = None,
    default_value: Any = 0.0
) -> Dict[str, Any]:
    """
    Get an assumption value with its source information.
    
    Args:
        ai_assumptions: AI assumptions object
        assumption_key: Key for the assumption
        manual_value: Manual override value
        default_value: Default value if not found
    
    Returns:
        Dictionary with 'value', 'source', and optionally 'distribution'
    """
    result = {
        'value': default_value,
        'source': 'default'
    }
    
    # Check for manual override first
    if manual_value is not None:
        result['value'] = manual_value
        result['source'] = 'manual'
        return result
    
    # Try to get from AI assumptions
    try:
        if ai_assumptions and hasattr(ai_assumptions, 'assumptions_saved') and ai_assumptions.assumptions_saved:
            if hasattr(ai_assumptions, 'assumptions') and assumption_key in ai_assumptions.assumptions:
                assumption = ai_assumptions.assumptions[assumption_key]
                result['value'] = getattr(assumption, 'value', default_value)
                result['source'] = 'ai'
                
                # Include distribution info if available
                if hasattr(assumption, 'distribution'):
                    result['distribution'] = {
                        'type': getattr(assumption.distribution, 'type', 'unknown'),
                        'params': getattr(assumption.distribution, 'params', {})
                    }
    except Exception:
        pass
    
    return result
