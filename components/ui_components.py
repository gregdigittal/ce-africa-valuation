"""
CE Africa Valuation Platform - UI Components
=============================================
Reusable UI building blocks following Linear design system

Crusher Equipment Africa - Empowering Mining Excellence
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from linear_theme import COLORS, badge, stat_card, section_header, empty_state

# =============================================================================
# LAYOUT COMPONENTS
# =============================================================================

def page_header(title: str, subtitle: str = None, actions: list = None):
    """
    Render a page header with title, optional subtitle, and action buttons.
    
    Args:
        title: Main page title
        subtitle: Optional description text
        actions: List of (label, key, type) tuples for action buttons
    """
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f'''
        <div style="margin-bottom: 1.5rem;">
            <h1 style="
                font-size: 1.875rem;
                font-weight: 700;
                color: {COLORS['text_primary']};
                margin: 0 0 0.25rem 0;
                letter-spacing: -0.02em;
            ">{title}</h1>
            {f'<p style="color: {COLORS["text_tertiary"]}; margin: 0; font-size: 0.9375rem;">{subtitle}</p>' if subtitle else ''}
        </div>
        ''', unsafe_allow_html=True)
    
    if actions:
        with col2:
            action_cols = st.columns(len(actions))
            for i, (label, key, btn_type) in enumerate(actions):
                with action_cols[i]:
                    st.button(label, key=key, type=btn_type if btn_type else "secondary")


def metric_row(metrics: List[Dict[str, Any]], columns: int = 4):
    """
    Render a row of metric cards.
    
    Args:
        metrics: List of dicts with keys: label, value, delta (optional), delta_type (optional), gold (optional)
        columns: Number of columns
    """
    cols = st.columns(columns)
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            delta = metric.get('delta')
            delta_type = metric.get('delta_type', 'positive')
            gold = metric.get('gold', False)
            
            st.markdown(
                stat_card(
                    label=metric['label'],
                    value=metric['value'],
                    delta=delta,
                    delta_type=delta_type,
                    gold=gold
                ),
                unsafe_allow_html=True
            )


def info_card(title: str, content: str, icon: str = None, variant: str = 'default'):
    """
    Render an information card.
    
    Args:
        title: Card title
        content: Card content (can include HTML)
        icon: Optional emoji icon
        variant: 'default', 'gold', 'success', 'warning', 'error'
    """
    border_colors = {
        'default': COLORS['border_subtle'],
        'gold': COLORS['accent_gold'],
        'success': COLORS['success'],
        'warning': COLORS['warning'],
        'error': COLORS['error'],
    }
    border_color = border_colors.get(variant, border_colors['default'])
    
    icon_html = f'<span style="font-size: 1.25rem; margin-right: 0.5rem;">{icon}</span>' if icon else ''
    
    st.markdown(f'''
    <div style="
        background-color: {COLORS['bg_elevated']};
        border: 1px solid {COLORS['border_subtle']};
        border-left: 3px solid {border_color};
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    ">
        <div style="
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
        ">
            {icon_html}
            <span style="
                font-size: 0.9375rem;
                font-weight: 600;
                color: {COLORS['text_primary']};
            ">{title}</span>
        </div>
        <div style="
            font-size: 0.875rem;
            color: {COLORS['text_secondary']};
            line-height: 1.5;
        ">{content}</div>
    </div>
    ''', unsafe_allow_html=True)


def divider(margin: str = '1.5rem'):
    """Render a subtle divider line."""
    st.markdown(f'''
    <hr style="
        border: none;
        border-top: 1px solid {COLORS['border_subtle']};
        margin: {margin} 0;
    " />
    ''', unsafe_allow_html=True)


# =============================================================================
# DATA DISPLAY COMPONENTS
# =============================================================================

def data_table_header(title: str, count: int = None, actions: list = None):
    """
    Render a header for data tables with title, count badge, and actions.
    
    Args:
        title: Table title
        count: Optional count to show as badge
        actions: List of (label, callback) tuples
    """
    col1, col2 = st.columns([3, 1])
    
    with col1:
        count_badge = badge(str(count), 'neutral') if count is not None else ''
        st.markdown(f'''
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
            <span style="
                font-size: 1rem;
                font-weight: 600;
                color: {COLORS['text_primary']};
            ">{title}</span>
            {count_badge}
        </div>
        ''', unsafe_allow_html=True)


def status_badge(status: str, size: str = 'md'):
    """
    Render a status badge with automatic coloring.
    
    Args:
        status: Status text (e.g., 'Active', 'Pending', 'Complete')
        size: 'sm', 'md', or 'lg'
    """
    status_variants = {
        'active': 'success',
        'complete': 'success',
        'completed': 'success',
        'success': 'success',
        'approved': 'success',
        'pending': 'warning',
        'in progress': 'warning',
        'processing': 'warning',
        'review': 'warning',
        'draft': 'neutral',
        'inactive': 'neutral',
        'error': 'error',
        'failed': 'error',
        'rejected': 'error',
        'cancelled': 'error',
    }
    
    variant = status_variants.get(status.lower(), 'neutral')
    
    sizes = {
        'sm': ('0.6875rem', '0.1875rem 0.5rem'),
        'md': ('0.75rem', '0.25rem 0.625rem'),
        'lg': ('0.8125rem', '0.375rem 0.75rem'),
    }
    font_size, padding = sizes.get(size, sizes['md'])
    
    return badge(status, variant)


def progress_indicator(value: float, label: str = None, show_value: bool = True):
    """
    Render a progress bar with optional label.
    
    Args:
        value: Progress value between 0 and 1
        label: Optional label text
        show_value: Whether to show percentage value
    """
    percentage = min(max(value * 100, 0), 100)
    
    # Determine color based on progress
    if percentage >= 100:
        color = COLORS['success']
    elif percentage >= 75:
        color = COLORS['accent_gold']
    elif percentage >= 50:
        color = COLORS['warning']
    else:
        color = COLORS['text_tertiary']
    
    label_html = f'<span style="color: {COLORS["text_secondary"]}; font-size: 0.8125rem;">{label}</span>' if label else ''
    value_html = f'<span style="color: {COLORS["text_primary"]}; font-size: 0.8125rem; font-weight: 500;">{percentage:.0f}%</span>' if show_value else ''
    
    st.markdown(f'''
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.375rem;">
            {label_html}
            {value_html}
        </div>
        <div style="
            height: 6px;
            background-color: {COLORS['bg_active']};
            border-radius: 3px;
            overflow: hidden;
        ">
            <div style="
                height: 100%;
                width: {percentage}%;
                background-color: {color};
                border-radius: 3px;
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


# =============================================================================
# FORM COMPONENTS
# =============================================================================

def form_section(title: str, description: str = None):
    """Start a form section with title and optional description."""
    st.markdown(f'''
    <div style="margin-bottom: 1rem;">
        <div style="
            font-size: 0.9375rem;
            font-weight: 600;
            color: {COLORS['text_primary']};
            margin-bottom: 0.25rem;
        ">{title}</div>
        {f'<div style="font-size: 0.8125rem; color: {COLORS["text_tertiary"]};">{description}</div>' if description else ''}
    </div>
    ''', unsafe_allow_html=True)


def input_group(label: str, help_text: str = None):
    """
    Create a styled input group wrapper.
    Use with st.text_input, st.number_input, etc.
    """
    st.markdown(f'''
    <div style="margin-bottom: 0.375rem;">
        <label style="
            font-size: 0.875rem;
            font-weight: 500;
            color: {COLORS['text_primary']};
        ">{label}</label>
        {f'<span style="font-size: 0.75rem; color: {COLORS["text_tertiary"]}; margin-left: 0.5rem;">{help_text}</span>' if help_text else ''}
    </div>
    ''', unsafe_allow_html=True)


# =============================================================================
# NAVIGATION COMPONENTS
# =============================================================================

def nav_tabs(tabs: List[str], active: str = None) -> str:
    """
    Render navigation tabs and return selected tab.
    
    Args:
        tabs: List of tab names
        active: Currently active tab (optional)
    
    Returns:
        Selected tab name
    """
    if active is None:
        active = tabs[0]
    
    # Use Streamlit's native tabs but style them
    return st.tabs(tabs)


def sidebar_nav_item(label: str, icon: str, active: bool = False, key: str = None):
    """
    Render a sidebar navigation item.
    
    Args:
        label: Item label
        icon: Emoji icon
        active: Whether item is active
        key: Unique key for button
    
    Returns:
        True if clicked
    """
    bg_color = COLORS['accent_gold_muted'] if active else 'transparent'
    text_color = COLORS['accent_gold'] if active else COLORS['text_secondary']
    
    clicked = st.button(
        f"{icon}  {label}",
        key=key or label,
        use_container_width=True,
    )
    
    return clicked


# =============================================================================
# FEEDBACK COMPONENTS
# =============================================================================

# =============================================================================
# STANDARDIZED MESSAGING & LOADING STATES (Sprint 19)
# =============================================================================

def show_loading(message: str = "Loading...", key: str = None):
    """
    Show a standardized loading spinner with message.
    
    Args:
        message: Loading message to display
        key: Optional unique key for the spinner
    """
    return st.spinner(message)


def show_progress(current: int, total: int, message: str = None, key: str = None):
    """
    Show a standardized progress bar.
    
    Args:
        current: Current progress value
        total: Total value
        message: Optional progress message
        key: Optional unique key for the progress bar
    """
    progress = current / total if total > 0 else 0
    if message:
        return st.progress(progress, text=message)
    return st.progress(progress)


def show_success(message: str, details: str = None, icon: str = "✅"):
    """
    Show a standardized success message.
    
    Args:
        message: Main success message
        details: Optional additional details
        icon: Optional icon (default: ✅)
    """
    full_message = f"{icon} **{message}**"
    if details:
        full_message += f"\n\n{details}"
    st.success(full_message)


def show_error(message: str, details: str = None, icon: str = "❌"):
    """
    Show a standardized error message.
    
    Args:
        message: Main error message
        details: Optional additional details or traceback
        icon: Optional icon (default: ❌)
    """
    full_message = f"{icon} **{message}**"
    if details:
        full_message += f"\n\n{details}"
    st.error(full_message)


def show_warning(message: str, details: str = None, icon: str = "⚠️"):
    """
    Show a standardized warning message.
    
    Args:
        message: Main warning message
        details: Optional additional details
        icon: Optional icon (default: ⚠️)
    """
    full_message = f"{icon} **{message}**"
    if details:
        full_message += f"\n\n{details}"
    st.warning(full_message)


def show_info(message: str, details: str = None, icon: str = "ℹ️"):
    """
    Show a standardized info message.
    
    Args:
        message: Main info message
        details: Optional additional details
        icon: Optional icon (default: ℹ️)
    """
    full_message = f"{icon} **{message}**"
    if details:
        full_message += f"\n\n{details}"
    st.info(full_message)


def toast(message: str, variant: str = 'info'):
    """
    Display a toast notification.
    
    Args:
        message: Notification message
        variant: 'info', 'success', 'warning', 'error'
    """
    if variant == 'success':
        st.success(message)
    elif variant == 'warning':
        st.warning(message)
    elif variant == 'error':
        st.error(message)
    else:
        st.info(message)


def confirmation_dialog(title: str, message: str, confirm_label: str = "Confirm", cancel_label: str = "Cancel"):
    """
    Display a confirmation dialog.
    
    Args:
        title: Dialog title
        message: Confirmation message
        confirm_label: Label for confirm button
        cancel_label: Label for cancel button
    
    Returns:
        True if confirmed, False if cancelled, None if no action
    """
    with st.container():
        st.markdown(f'''
        <div style="
            background-color: {COLORS['bg_elevated']};
            border: 1px solid {COLORS['border_default']};
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        ">
            <div style="
                font-size: 1.125rem;
                font-weight: 600;
                color: {COLORS['text_primary']};
                margin-bottom: 0.5rem;
            ">{title}</div>
            <div style="
                font-size: 0.9375rem;
                color: {COLORS['text_secondary']};
                margin-bottom: 1rem;
            ">{message}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            cancel = st.button(cancel_label, key="dialog_cancel")
        with col3:
            confirm = st.button(confirm_label, key="dialog_confirm", type="primary")
        
        if confirm:
            return True
        if cancel:
            return False
        return None


# =============================================================================
# CHART COMPONENTS
# =============================================================================

def get_chart_colors() -> Dict[str, str]:
    """Get standard chart colors matching the theme."""
    return {
        'primary': COLORS['accent_gold'],
        'secondary': COLORS['text_tertiary'],
        'success': COLORS['success'],
        'warning': COLORS['warning'],
        'error': COLORS['error'],
        'info': COLORS['info'],
        'grid': COLORS['border_subtle'],
        'background': COLORS['bg_elevated'],
        'text': COLORS['text_secondary'],
    }


def apply_plotly_theme(fig):
    """
    Apply Linear theme to a Plotly figure.
    
    Args:
        fig: Plotly figure object
    
    Returns:
        Themed figure
    """
    colors = get_chart_colors()
    
    fig.update_layout(
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font=dict(
            family="Inter, sans-serif",
            size=12,
            color=colors['text'],
        ),
        title=dict(
            font=dict(
                size=16,
                color=COLORS['text_primary'],
            ),
            x=0,
            xanchor='left',
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            font=dict(color=colors['text']),
        ),
        xaxis=dict(
            gridcolor=colors['grid'],
            linecolor=colors['grid'],
            tickfont=dict(color=colors['text']),
            title_font=dict(color=colors['text']),
        ),
        yaxis=dict(
            gridcolor=colors['grid'],
            linecolor=colors['grid'],
            tickfont=dict(color=colors['text']),
            title_font=dict(color=colors['text']),
        ),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    
    return fig


def chart_container(title: str = None, height: int = 400):
    """
    Create a styled container for charts.
    
    Args:
        title: Optional chart title
        height: Container height in pixels
    """
    title_html = f'''
    <div style="
        font-size: 0.9375rem;
        font-weight: 600;
        color: {COLORS['text_primary']};
        margin-bottom: 1rem;
    ">{title}</div>
    ''' if title else ''
    
    st.markdown(f'''
    <div style="
        background-color: {COLORS['bg_elevated']};
        border: 1px solid {COLORS['border_subtle']};
        border-radius: 8px;
        padding: 1.25rem;
        min-height: {height}px;
    ">
        {title_html}
    </div>
    ''', unsafe_allow_html=True)


# =============================================================================
# SPECIAL COMPONENTS
# =============================================================================

def scenario_card(name: str, status: str, metrics: Dict[str, str], is_active: bool = False):
    """
    Render a scenario selection card.
    
    Args:
        name: Scenario name
        status: Status text
        metrics: Dict of metric labels and values
        is_active: Whether this scenario is selected
    """
    border_color = COLORS['accent_gold'] if is_active else COLORS['border_subtle']
    bg_color = COLORS['accent_gold_subtle'] if is_active else COLORS['bg_elevated']
    
    metrics_html = ''.join([
        f'''<div style="display: flex; justify-content: space-between; padding: 0.375rem 0; border-bottom: 1px solid {COLORS['border_subtle']};">
            <span style="color: {COLORS['text_tertiary']}; font-size: 0.8125rem;">{k}</span>
            <span style="color: {COLORS['text_primary']}; font-size: 0.8125rem; font-weight: 500;">{v}</span>
        </div>'''
        for k, v in metrics.items()
    ])
    
    st.markdown(f'''
    <div style="
        background-color: {bg_color};
        border: 1px solid {border_color};
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.15s ease;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <span style="font-weight: 600; color: {COLORS['text_primary']};">{name}</span>
            {badge(status, 'success' if status.lower() == 'active' else 'neutral')}
        </div>
        {metrics_html}
    </div>
    ''', unsafe_allow_html=True)


def valuation_summary(enterprise_value: str, equity_value: str, confidence: str = None):
    """
    Render the main valuation summary display.
    
    Args:
        enterprise_value: Formatted EV string
        equity_value: Formatted equity value string
        confidence: Optional confidence interval string
    """
    confidence_html = f'''
    <div style="
        font-size: 0.8125rem;
        color: {COLORS['text_tertiary']};
        margin-top: 0.5rem;
    ">90% Confidence: {confidence}</div>
    ''' if confidence else ''
    
    st.markdown(f'''
    <div style="
        background: linear-gradient(135deg, {COLORS['bg_elevated']} 0%, {COLORS['bg_surface']} 100%);
        border: 1px solid {COLORS['accent_gold']}40;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    ">
        <div style="
            font-size: 0.75rem;
            font-weight: 600;
            color: {COLORS['text_tertiary']};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        ">Enterprise Value</div>
        <div style="
            font-size: 2.5rem;
            font-weight: 700;
            color: {COLORS['accent_gold']};
            letter-spacing: -0.02em;
        ">{enterprise_value}</div>
        {confidence_html}
        <div style="
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid {COLORS['border_subtle']};
        ">
            <div style="
                font-size: 0.75rem;
                font-weight: 600;
                color: {COLORS['text_tertiary']};
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-bottom: 0.25rem;
            ">Equity Value</div>
            <div style="
                font-size: 1.5rem;
                font-weight: 600;
                color: {COLORS['text_primary']};
            ">{equity_value}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def inject_custom_css():
    """Inject custom CSS theme - compatibility function for app_refactored.py"""
    try:
        from linear_theme import apply_ce_africa_branding
        apply_ce_africa_branding()
    except ImportError:
        # Fallback if linear_theme not available
        pass


def timeline_item(date: str, title: str, description: str = None, status: str = 'complete'):
    """
    Render a timeline item for audit logs or history.
    
    Args:
        date: Date/time string
        title: Event title
        description: Optional description
        status: 'complete', 'current', 'upcoming'
    """
    colors = {
        'complete': COLORS['success'],
        'current': COLORS['accent_gold'],
        'upcoming': COLORS['text_tertiary'],
    }
    dot_color = colors.get(status, colors['complete'])
    
    desc_html = f'<div style="color: {COLORS["text_tertiary"]}; font-size: 0.8125rem; margin-top: 0.25rem;">{description}</div>' if description else ''
    
    st.markdown(f'''
    <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
        <div style="
            width: 12px;
            height: 12px;
            background-color: {dot_color};
            border-radius: 50%;
            margin-top: 0.25rem;
            flex-shrink: 0;
        "></div>
        <div style="flex: 1;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="color: {COLORS['text_primary']}; font-weight: 500;">{title}</span>
                <span style="color: {COLORS['text_tertiary']}; font-size: 0.75rem;">{date}</span>
            </div>
            {desc_html}
        </div>
    </div>
    ''', unsafe_allow_html=True)
