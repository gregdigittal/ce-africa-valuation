"""
Workflow Navigator Component
============================
Visual workflow progress indicator and navigation.

Version: 1.0
"""

import streamlit as st
from typing import Dict, Any, Callable, Optional

def render_workflow_navigator_simple(
    current_stage: str,
    stages_status: Dict[str, Dict[str, Any]],
    on_navigate: Callable[[str, str], None],
    key_prefix: str = "workflow"
):
    """
    Render a simple workflow navigator showing stage progress.
    
    Args:
        current_stage: Current workflow stage ID
        stages_status: Dict of stage_id -> {complete: bool, ...}
        on_navigate: Callback function(stage_id, section) for navigation
        key_prefix: Prefix for Streamlit keys to avoid duplicates
    """
    # Workflow stage mapping
    STAGE_TO_SECTION = {
        'setup': 'setup',
        'ai_analysis': 'ai_assumptions',
        'assumptions_review': 'ai_assumptions',
        'forecast': 'forecast',
        'manufacturing': 'manufacturing',
        'funding': 'funding',
    }
    
    # Stage display names
    STAGE_NAMES = {
        'setup': 'Setup',
        'ai_analysis': 'AI Analysis',
        'assumptions_review': 'Assumptions Review',
        'forecast': 'Forecast',
        'manufacturing': 'Manufacturing',
        'funding': 'Funding',
    }
    
    st.markdown("### Workflow Progress")
    
    # Get workflow stages in order
    workflow_stages = [
        ('setup', 'Setup'),
        ('ai_analysis', 'AI Analysis'),
        ('assumptions_review', 'Assumptions Review'),
        ('forecast', 'Forecast'),
        ('manufacturing', 'Manufacturing Strategy'),
        ('funding', 'Funding & Returns'),
    ]
    
    # Render stages
    cols = st.columns(len(workflow_stages))
    
    for idx, (stage_id, stage_name) in enumerate(workflow_stages):
        with cols[idx]:
            stage_info = stages_status.get(stage_id, {})
            is_complete = stage_info.get('complete', False)
            is_current = current_stage == stage_id
            
            # Determine status icon and color
            if is_complete:
                icon = "✅"
                color = "#22C55E"  # Green
            elif is_current:
                icon = "→"
                color = "#D4A537"  # Gold
            else:
                icon = "○"
                color = "#71717A"  # Gray
            
            # Create button
            section = STAGE_TO_SECTION.get(stage_id, stage_id)
            button_label = f"{icon} {stage_name}"
            
            if st.button(
                button_label,
                key=f"{key_prefix}_stage_{stage_id}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                on_navigate(stage_id, section)
    
    # Show completion status
    completed_count = sum(1 for stage_id, _ in workflow_stages if stages_status.get(stage_id, {}).get('complete', False))
    total_count = len(workflow_stages)
    
    if completed_count == total_count:
        st.success(f"✅ All workflow stages complete ({completed_count}/{total_count})")
    else:
        st.info(f"Progress: {completed_count}/{total_count} stages complete")
