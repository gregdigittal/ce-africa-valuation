"""
Enhanced Navigation Component
============================
Modern navigation components with improved styling and UX.

Version: 1.0
Date: December 15, 2025
"""

import streamlit as st
from typing import List, Dict, Optional, Callable, Any


# =============================================================================
# ENHANCED NAVIGATION COMPONENTS
# =============================================================================

def render_enhanced_sidebar_nav(
    nav_items: List[Dict[str, Any]],
    current_section: Optional[str] = None
) -> None:
    """
    Render an enhanced sidebar navigation with visual indicators.
    
    Args:
        nav_items: List of navigation items with keys: 'id', 'label', 'icon', 'on_click'
        current_section: ID of currently active section
    """
    st.sidebar.markdown("### Navigation")
    
    for item in nav_items:
        item_id = item.get('id', '')
        label = item.get('label', '')
        icon = item.get('icon', '')
        on_click = item.get('on_click', None)
        badge = item.get('badge', None)
        disabled = item.get('disabled', False)
        
        # Determine if this item is active
        is_active = current_section == item_id
        
        # Build label with icon and badge
        display_label = f"{icon} {label}" if icon else label
        if badge:
            display_label += f" {badge}"
        
        # Render button with styling
        button_style = "primary" if is_active else "secondary"
        
        if on_click and callable(on_click):
            if st.sidebar.button(
                display_label,
                key=f"nav_{item_id}",
                use_container_width=True,
                type=button_style,
                disabled=disabled
            ):
                on_click()
        else:
            st.sidebar.markdown(f"**{display_label}**" if is_active else display_label)


def render_breadcrumb_nav(
    items: List[Dict[str, str]],
    separator: str = "›"
) -> None:
    """
    Render a breadcrumb navigation trail.
    
    Args:
        items: List of breadcrumb items with 'label' and optional 'action'
        separator: Separator between items
    """
    breadcrumb_html = '<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; font-size: 0.875rem; color: #A1A1AA;">'
    
    for i, item in enumerate(items):
        label = item.get('label', '')
        action = item.get('action', None)
        
        if action:
            breadcrumb_html += f'<a href="#" onclick="{action}" style="color: #D4A537; text-decoration: none;">{label}</a>'
        else:
            breadcrumb_html += f'<span style="color: #FAFAFA;">{label}</span>'
        
        if i < len(items) - 1:
            breadcrumb_html += f'<span style="color: #71717A;">{separator}</span>'
    
    breadcrumb_html += '</div>'
    st.markdown(breadcrumb_html, unsafe_allow_html=True)


def render_tab_navigation(
    tabs: List[Dict[str, Any]],
    default_tab: Optional[str] = None
) -> str:
    """
    Render enhanced tab navigation with icons and badges.
    
    Args:
        tabs: List of tab definitions with 'id', 'label', 'icon', optional 'badge'
        default_tab: Default selected tab ID
    
    Returns:
        Selected tab ID
    """
    tab_labels = []
    for tab in tabs:
        label = tab.get('label', '')
        icon = tab.get('icon', '')
        badge = tab.get('badge', None)
        
        display_label = f"{icon} {label}" if icon else label
        if badge:
            display_label += f" ({badge})"
        
        tab_labels.append(display_label)
    
    selected_idx = 0
    if default_tab:
        for i, tab in enumerate(tabs):
            if tab.get('id') == default_tab:
                selected_idx = i
                break
    
    selected_label = st.radio(
        "Select view:",
        options=tab_labels,
        horizontal=True,
        index=selected_idx,
        label_visibility="collapsed"
    )
    
    # Find selected tab ID
    selected_idx = tab_labels.index(selected_label)
    return tabs[selected_idx].get('id', '')


def render_progress_indicator(
    steps: List[Dict[str, Any]],
    current_step: Optional[str] = None
) -> None:
    """
    Render a progress indicator showing workflow steps.
    
    Args:
        steps: List of step definitions with 'id', 'label', 'status' ('pending', 'active', 'completed')
        current_step: ID of current step
    """
    st.markdown('<div style="display: flex; align-items: center; gap: 1rem; margin: 1rem 0;">', unsafe_allow_html=True)
    
    for i, step in enumerate(steps):
        step_id = step.get('id', '')
        label = step.get('label', '')
        status = step.get('status', 'pending')
        
        # Determine status
        if current_step == step_id:
            status = 'active'
        elif i < steps.index(next((s for s in steps if s.get('id') == current_step), steps[-1])):
            status = 'completed'
        
        # Render step
        if status == 'completed':
            icon = "✅"
            color = "#22C55E"
        elif status == 'active':
            icon = "→"
            color = "#D4A537"
        else:
            icon = "○"
            color = "#71717A"
        
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
                <div style="
                    width: 2rem;
                    height: 2rem;
                    border-radius: 50%;
                    background-color: {color}20;
                    border: 2px solid {color};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: 600;
                    color: {color};
                ">{icon}</div>
                <span style="margin-top: 0.5rem; font-size: 0.75rem; color: {color}; text-align: center;">{label}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Render connector line (except for last step)
        if i < len(steps) - 1:
            connector_color = "#22C55E" if status == 'completed' else "#71717A"
            st.markdown(
                f'<div style="flex: 1; height: 2px; background-color: {connector_color}; margin: 0 0.5rem;"></div>',
                unsafe_allow_html=True
            )
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_quick_actions(
    actions: List[Dict[str, Any]]
) -> None:
    """
    Render quick action buttons.
    
    Args:
        actions: List of action definitions with 'label', 'icon', 'on_click', optional 'color'
    """
    cols = st.columns(len(actions))
    
    for i, action in enumerate(actions):
        with cols[i]:
            label = action.get('label', '')
            icon = action.get('icon', '')
            on_click = action.get('on_click', None)
            color = action.get('color', 'primary')
            
            display_label = f"{icon} {label}" if icon else label
            
            if on_click and callable(on_click):
                if st.button(
                    display_label,
                    key=f"quick_action_{i}",
                    use_container_width=True,
                    type=color
                ):
                    on_click()
            else:
                st.button(
                    display_label,
                    key=f"quick_action_{i}",
                    use_container_width=True,
                    type=color,
                    disabled=True
                )
