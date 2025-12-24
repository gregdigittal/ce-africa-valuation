"""
Workflow Guidance Component
============================
Contextual help, tips, and prerequisite warnings for workflow stages.

Version: 1.0
"""

import streamlit as st
from typing import List, Dict, Any, Optional

# Import standardized UI components (Sprint 19)
try:
    from components.ui_components import show_warning, show_info
except ImportError:
    # Fallback
    def show_warning(msg, details=None, icon="⚠️"):
        st.warning(f"{icon} **{msg}**" + (f"\n\n{details}" if details else ""))
    def show_info(msg, details=None, icon="ℹ️"):
        st.info(f"{icon} **{msg}**" + (f"\n\n{details}" if details else ""))

def render_contextual_help(
    stage_id: str,
    stage_name: str,
    description: str,
    prerequisites: List[str],
    tips: Optional[List[str]] = None
):
    """
    Render contextual help for a workflow stage.
    
    Args:
        stage_id: Workflow stage ID
        stage_name: Display name of the stage
        description: Stage description
        prerequisites: List of prerequisite stage IDs
        tips: Optional list of tips/guidance
    """
    with st.expander("ℹ️ Contextual Help", expanded=False):
        st.markdown(f"**{stage_name}**")
        if description:
            st.caption(description)
        
        if prerequisites:
            st.markdown("**Prerequisites:**")
            for prereq in prerequisites:
                st.markdown(f"- {prereq.title()}")
        
        if tips:
            st.markdown("**Tips:**")
            for tip in tips:
                st.markdown(f"- {tip}")


def get_stage_tips(stage_id: str) -> List[str]:
    """
    Get tips for a workflow stage.
    
    Args:
        stage_id: Workflow stage ID
        
    Returns:
        List of tip strings
    """
    tips_map = {
        'setup': [
            "Configure WACC and forecast duration in Basics",
            "Import or add customers in the Customers step",
            "Add fleet/machines in the Fleet step",
            "Configure wear profiles for each machine model"
        ],
        'ai_analysis': [
            "Ensure historical financial data is imported",
            "Run analysis to derive probability distributions",
            "Review and adjust assumptions before saving"
        ],
        'assumptions_review': [
            "Review all AI-derived assumptions",
            "Adjust distributions if needed",
            "Save assumptions to proceed to Forecast"
        ],
        'forecast': [
            "Ensure assumptions are saved before running",
            "Review forecast results in the tabs",
            "Export results if needed"
        ],
        'manufacturing': [
            "This is optional but recommended",
            "Configure make vs buy decisions",
            "Set commissioning schedule if manufacturing"
        ],
        'funding': [
            "This is optional",
            "Configure debt/equity if needed",
            "Run IRR analysis"
        ],
    }
    
    return tips_map.get(stage_id, [])


def render_prerequisite_warning(missing: List[str], current_stage: str):
    """
    Render a warning about missing prerequisites with enhanced messaging.
    
    Args:
        missing: List of missing prerequisite stage IDs
        current_stage: Current stage ID
    """
    # Import standardized UI components (Sprint 19)
    try:
        from components.ui_components import show_warning, show_info
    except ImportError:
        # Fallback
        def show_warning(msg, details=None, icon="⚠️"):
            st.warning(f"{icon} **{msg}**" + (f"\n\n{details}" if details else ""))
        def show_info(msg, details=None, icon="ℹ️"):
            st.info(f"{icon} **{msg}**" + (f"\n\n{details}" if details else ""))
    
    if not missing:
        return
    
    stage_names = {
        'setup': 'Setup',
        'ai_analysis': 'AI Analysis',
        'assumptions_review': 'Assumptions Review',
        'forecast': 'Forecast',
        'manufacturing': 'Manufacturing Strategy',
        'funding': 'Funding & Returns'
    }
    
    missing_names = [stage_names.get(m, m.title()) for m in missing]
    
    # Enhanced error message (Sprint 19)
    show_warning(
        "Prerequisites Not Met",
        f"Please complete the following steps before continuing:\n- **{', '.join(missing_names)}**"
    )
    
    # Provide helpful guidance
    if 'setup' in missing:
        show_info(
            "Setup Required",
            "Go to **Setup** in the sidebar to configure scenario basics, import data, and add your fleet."
        )
    elif 'ai_analysis' in missing:
        show_info(
            "AI Analysis Required",
            "Run AI Analysis to derive assumptions from historical data before proceeding."
        )
    elif 'assumptions_review' in missing:
        show_info(
            "Assumptions Review Required",
            "Review and save your assumptions before running the forecast."
        )
    elif 'forecast' in missing:
        show_info(
            "Forecast Required",
            "Run a forecast first to generate baseline results."
        )
    
    if st.button("→ Go to Required Step", type="primary", key=f"goto_{missing[0] if missing else 'setup'}"):
        st.session_state.current_section = missing[0] if missing else 'setup'
        st.rerun()


def render_whats_next_widget(
    current_stage_id: str,
    next_stage: Optional[Dict[str, Any]],
    workflow_progress: Dict[str, Any]
):
    """
    Render a "What's Next?" widget in the sidebar.
    
    Args:
        current_stage_id: Current workflow stage ID
        next_stage: Next stage dict or None
        workflow_progress: Workflow progress information
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### What's Next?")
    
    if not next_stage:
        st.sidebar.success("🎉 All required stages complete!")
        return
    
    stage_name = next_stage.get('name', 'Next Stage')
    stage_description = next_stage.get('description', '')
    
    st.sidebar.markdown(f"**{stage_name}**")
    if stage_description:
        st.sidebar.caption(stage_description)
    
    # Get section for navigation
    section = next_stage.get('section', next_stage.get('id', 'setup'))
    
    if st.sidebar.button(f"→ Go to {stage_name}", key="whats_next_button", use_container_width=True):
        st.session_state.current_section = section
        st.rerun()
