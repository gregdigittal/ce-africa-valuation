"""
User & Role Management Component (Sprint 7)
============================================
UI for managing scenario access, sharing, and user roles.

Crusher Equipment Africa - Empowering Mining Excellence
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from db_connector import SupabaseHandler

try:
    from utils.permissions import (
        PermissionChecker, Permissions, 
        ROLE_DISPLAY_NAMES, ROLE_COLORS, ROLE_HIERARCHY,
        render_permission_badge, log_action
    )
    PERMISSIONS_AVAILABLE = True
except ImportError:
    PERMISSIONS_AVAILABLE = False
    # Fallback constants when permissions module not available
    ROLE_COLORS = {
        'owner': '#f59e0b',      # Gold
        'admin': '#8b5cf6',      # Purple
        'analyst': '#3b82f6',    # Blue
        'viewer': '#10b981',     # Green
        'board_member': '#6366f1', # Indigo
        'investor': '#ec4899'    # Pink
    }
    ROLE_DISPLAY_NAMES = {
        'owner': 'Owner',
        'admin': 'Admin',
        'analyst': 'Analyst',
        'viewer': 'Viewer',
        'board_member': 'Board Member',
        'investor': 'Investor'
    }
    ROLE_HIERARCHY = {
        'owner': 100,
        'admin': 80,
        'analyst': 60,
        'viewer': 40,
        'board_member': 30,
        'investor': 20
    }


# =============================================================================
# ROLE SELECTION HELPERS
# =============================================================================

ROLE_DESCRIPTIONS = {
    'owner': 'Full control including delete and role management',
    'admin': 'Broad access, can edit data and manage users',
    'analyst': 'Can edit data, run forecasts, but no admin access',
    'viewer': 'Read-only access to all data and reports',
    'board_member': 'Summary views with valuation focus',
    'investor': 'Limited view - no customer names or detailed costs'
}

ROLE_OPTIONS = ['admin', 'analyst', 'viewer', 'board_member', 'investor']


def get_role_icon(role_name: str) -> str:
    """Get emoji icon for role."""
    icons = {
        'owner': '👑',
        'admin': '🔑',
        'analyst': '📊',
        'viewer': '👁️',
        'board_member': '🏛️',
        'investor': '💼'
    }
    return icons.get(role_name, '👤')


# =============================================================================
# USER MANAGEMENT TAB
# =============================================================================

def render_current_users(db: SupabaseHandler, scenario_id: str, user_id: str):
    """Render list of users with access to this scenario."""
    
    # Get scenario info
    scenarios = db.get_user_scenarios(user_id)
    scenario = next((s for s in scenarios if s.get('id') == scenario_id), None)
    
    is_owner = scenario is not None
    
    # =========================================================================
    # ADD NEW USER SECTION (at top for visibility)
    # =========================================================================
    if is_owner:
        st.markdown("### ➕ Add New User")
        
        with st.expander("Click to add a user to this scenario", expanded=False):
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                        padding: 1rem; border-radius: 8px; margin-bottom: 1rem;
                        border-left: 3px solid #D4A537;">
                <p style="margin: 0; color: #B0B0B0; font-size: 0.9rem;">
                    Add team members or stakeholders to collaborate on this scenario.
                    Each user will have permissions based on their assigned role.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("add_user_form", clear_on_submit=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    new_user_email = st.text_input(
                        "User Email or ID",
                        placeholder="user@company.com or user-uuid",
                        help="Enter the email address or user ID of the person to add"
                    )
                
                with col2:
                    new_role = st.selectbox(
                        "Role",
                        options=ROLE_OPTIONS,
                        format_func=lambda x: f"{get_role_icon(x)} {x.replace('_', ' ').title()}",
                        help="Select the access level for this user"
                    )
                
                # Show role description
                st.info(f"**{new_role.replace('_', ' ').title()}**: {ROLE_DESCRIPTIONS.get(new_role, '')}")
                
                add_submitted = st.form_submit_button("➕ Add User", type="primary", use_container_width=True)
                
                if add_submitted:
                    if not new_user_email:
                        st.error("Please enter a user email or ID")
                    else:
                        # Try to add the user
                        success = False
                        
                        # Check if assign_user_role method exists
                        if hasattr(db, 'assign_user_role'):
                            result = db.assign_user_role(
                                user_id=new_user_email,  # Could be email or UUID
                                scenario_id=scenario_id,
                                role_name=new_role,
                                invited_by=user_id
                            )
                            success = result is not None
                        else:
                            # Direct database insert as fallback
                            try:
                                # Try direct insert to user_roles table
                                data = {
                                    'user_id': new_user_email,
                                    'scenario_id': scenario_id,
                                    'role_name': new_role,
                                    'invited_by': user_id,
                                    'is_active': True
                                }
                                response = db.client.table('user_roles').upsert(
                                    data,
                                    on_conflict='user_id,scenario_id'
                                ).execute()
                                success = response.data is not None
                            except Exception as e:
                                st.error(f"Database error: {e}")
                                success = False
                        
                        if success:
                            st.success(f"✅ Added **{new_user_email}** as **{new_role.replace('_', ' ').title()}**")
                            st.rerun()
                        else:
                            st.error("Failed to add user. Make sure the user_roles table exists (run Sprint 7 migration).")
        
        st.markdown("---")
    
    # =========================================================================
    # CURRENT USERS LIST
    # =========================================================================
    st.markdown("### 👥 Current Users")
    
    # Get users with access
    users = []
    if hasattr(db, 'get_scenario_users'):
        users = db.get_scenario_users(scenario_id)
    else:
        # Fallback: try direct query
        try:
            response = db.client.table('user_roles').select('*').eq('scenario_id', scenario_id).eq('is_active', True).execute()
            users = response.data if response.data else []
        except Exception:
            users = []
    
    # Add owner if not in list
    if is_owner:
        owner_in_list = any(u.get('role_name') == 'owner' for u in users)
        if not owner_in_list:
            users.insert(0, {
                'user_id': user_id,
                'role_name': 'owner',
                'invited_at': scenario.get('created_at'),
                'is_current_user': True
            })
    
    if not users:
        st.info("No users have been granted access to this scenario yet. Use the form above to add users.")
        return
    
    # Display users
    for user in users:
        role = user.get('role_name', 'viewer')
        role_info = user.get('role_definitions', {})
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            user_display = user.get('user_id', 'Unknown')[:8] + '...'
            if user.get('is_current_user') or user.get('user_id') == user_id:
                user_display += " (You)"
            
            st.markdown(f"**{user_display}**")
            invited = user.get('invited_at', '')
            if invited:
                st.caption(f"Added: {str(invited)[:10]}")
        
        with col2:
            color = ROLE_COLORS.get(role, '#64748b')
            icon = get_role_icon(role)
            st.markdown(f"""
            <span style="
                background-color: {color}20;
                color: {color};
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.85rem;
            ">{icon} {role.replace('_', ' ').title()}</span>
            """, unsafe_allow_html=True)
        
        with col3:
            desc = ROLE_DESCRIPTIONS.get(role, '')
            st.caption(desc[:40] + '...' if len(desc) > 40 else desc)
        
        with col4:
            # Can't modify owner or yourself
            if role != 'owner' and user.get('user_id') != user_id and is_owner:
                if st.button("🗑️", key=f"remove_{user.get('id', user.get('user_id'))}"):
                    if hasattr(db, 'remove_user_role'):
                        db.remove_user_role(user.get('user_id'), scenario_id)
                        st.success("Access revoked")
                        st.rerun()
        
        st.markdown("---")


# =============================================================================
# SHARE SCENARIO TAB
# =============================================================================

def render_share_scenario(db: SupabaseHandler, scenario_id: str, user_id: str):
    """Render UI for sharing scenario with others."""
    st.markdown("### 📤 Share Scenario")
    
    st.markdown("""
    <div style="background: rgba(212,165,55,0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #D4A537; margin-bottom: 1rem;">
        <p style="margin: 0; color: #B0B0B0;">
            Share this scenario with team members or external parties. 
            They'll receive an invitation link to access the scenario with the role you assign.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("share_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            email = st.text_input(
                "Email Address",
                placeholder="colleague@company.com",
                help="Enter the email of the person you want to share with"
            )
        
        with col2:
            role = st.selectbox(
                "Role",
                options=ROLE_OPTIONS,
                format_func=lambda x: f"{get_role_icon(x)} {x.replace('_', ' ').title()}",
                help="Select the access level for this user"
            )
        
        # Role description
        st.caption(f"ℹ️ {ROLE_DESCRIPTIONS.get(role, '')}")
        
        col3, col4 = st.columns(2)
        
        with col3:
            expires = st.selectbox(
                "Link Expires",
                options=[7, 14, 30, 90, None],
                format_func=lambda x: f"{x} days" if x else "Never",
                index=2
            )
        
        with col4:
            message = st.text_area(
                "Message (optional)",
                placeholder="Hi, I'm sharing our valuation scenario with you...",
                height=80
            )
        
        submitted = st.form_submit_button("📤 Send Invitation", type="primary")
        
        if submitted:
            if not email:
                st.error("Please enter an email address")
            elif '@' not in email:
                st.error("Please enter a valid email address")
            else:
                if hasattr(db, 'create_scenario_share'):
                    result = db.create_scenario_share(
                        scenario_id=scenario_id,
                        shared_by=user_id,
                        shared_with_email=email,
                        role_name=role,
                        message=message,
                        expires_days=expires
                    )
                    
                    if result:
                        st.success(f"✅ Invitation sent to {email}")
                        
                        # Show share link
                        share_token = result.get('share_token', '')
                        if share_token:
                            share_url = f"https://yourapp.com/accept-share/{share_token}"
                            st.code(share_url)
                            st.caption("Share this link with the recipient")
                        
                        # Log action
                        log_action(db, user_id, 'share', 'scenario', 
                                  scenario_id=scenario_id,
                                  summary=f"Shared with {email} as {role}")
                    else:
                        st.error("Failed to create share invitation")
                else:
                    st.warning("Sharing functionality not available. Run the Sprint 7 migration.")
    
    # Show existing shares
    st.markdown("---")
    st.markdown("### 📋 Pending Invitations")
    
    if hasattr(db, 'get_scenario_shares'):
        shares = db.get_scenario_shares(scenario_id)
        pending = [s for s in shares if s.get('status') == 'pending']
        
        if not pending:
            st.info("No pending invitations")
        else:
            for share in pending:
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**{share.get('shared_with_email', 'Unknown')}**")
                    st.caption(f"Sent: {str(share.get('shared_at', ''))[:10]}")
                
                with col2:
                    role = share.get('role_name', 'viewer')
                    st.markdown(f"{get_role_icon(role)} {role.replace('_', ' ').title()}")
                
                with col3:
                    expires = share.get('expires_at')
                    if expires:
                        st.caption(f"Expires: {str(expires)[:10]}")
                    else:
                        st.caption("No expiry")
                
                with col4:
                    if st.button("❌", key=f"revoke_{share.get('id')}"):
                        if hasattr(db, 'revoke_scenario_share'):
                            db.revoke_scenario_share(share.get('id'))
                            st.success("Invitation revoked")
                            st.rerun()
                
                st.markdown("---")
    else:
        st.info("Sharing tracking not available. Run the Sprint 7 migration.")


# =============================================================================
# ROLE DEFINITIONS TAB
# =============================================================================

def render_role_definitions(db: SupabaseHandler):
    """Render role definitions and permissions matrix."""
    st.markdown("### 📜 Role Definitions")
    
    st.markdown("""
    <div style="background: rgba(212,165,55,0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #D4A537; margin-bottom: 1rem;">
        <p style="margin: 0; color: #B0B0B0;">
            These are the default permission levels for each role. 
            Owners can override specific permissions for individual users.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if hasattr(db, 'get_role_definitions'):
        roles = db.get_role_definitions()
    else:
        # Use defaults
        roles = [
            {'role_name': 'owner', 'role_description': 'Full administrative access'},
            {'role_name': 'admin', 'role_description': 'Broad access with limited admin'},
            {'role_name': 'analyst', 'role_description': 'Data editing and forecast capabilities'},
            {'role_name': 'viewer', 'role_description': 'Read-only access'},
            {'role_name': 'board_member', 'role_description': 'Summary and valuation views'},
            {'role_name': 'investor', 'role_description': 'Restricted external access'},
        ]
    
    # Display as cards
    cols = st.columns(3)
    
    for i, role in enumerate(roles):
        role_name = role.get('role_name', '')
        
        with cols[i % 3]:
            color = ROLE_COLORS.get(role_name, '#64748b')
            icon = get_role_icon(role_name)
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color}10, {color}05);
                border: 1px solid {color}40;
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1rem;
                height: 180px;
            ">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{role_name.replace('_', ' ').title()}</h4>
                <p style="color: #B0B0B0; font-size: 0.85rem; margin: 0;">
                    {role.get('role_description', ROLE_DESCRIPTIONS.get(role_name, ''))}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Permissions matrix
    with st.expander("📊 View Permissions Matrix", expanded=False):
        if roles and 'can_edit_scenario' in roles[0]:
            # Build matrix from actual data
            permission_cols = [k for k in roles[0].keys() if k.startswith('can_')]
            
            matrix_data = []
            for role in roles:
                row = {'Role': role.get('role_name', '').replace('_', ' ').title()}
                for perm in permission_cols:
                    row[perm.replace('can_', '').replace('_', ' ').title()] = '✅' if role.get(perm) else '❌'
                matrix_data.append(row)
            
            df = pd.DataFrame(matrix_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            # Show simplified matrix
            st.markdown("""
            | Permission | Owner | Admin | Analyst | Viewer | Investor |
            |------------|-------|-------|---------|--------|----------|
            | Edit Data | ✅ | ✅ | ✅ | ❌ | ❌ |
            | Run Forecast | ✅ | ✅ | ✅ | ❌ | ❌ |
            | View Revenue | ✅ | ✅ | ✅ | ✅ | ✅ |
            | View Margins | ✅ | ✅ | ✅ | ✅ | ❌ |
            | View Costs | ✅ | ✅ | ✅ | ✅ | ❌ |
            | View Customer Names | ✅ | ✅ | ✅ | ✅ | ❌ |
            | Manage Users | ✅ | ✅ | ❌ | ❌ | ❌ |
            | Share Scenario | ✅ | ✅ | ❌ | ❌ | ❌ |
            | Export Data | ✅ | ✅ | ✅ | ❌ | ❌ |
            """)


# =============================================================================
# AUDIT LOG TAB
# =============================================================================

def render_audit_log(db: SupabaseHandler, scenario_id: str, user_id: str):
    """Render audit log for scenario."""
    st.markdown("### 📝 Audit Log")
    
    st.markdown("""
    <div style="background: rgba(212,165,55,0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #D4A537; margin-bottom: 1rem;">
        <p style="margin: 0; color: #B0B0B0;">
            Track all changes made to this scenario for compliance and debugging.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        action_filter = st.selectbox(
            "Action",
            options=['All', 'create', 'update', 'delete', 'view', 'export', 'share'],
            index=0
        )
    
    with col2:
        entity_filter = st.selectbox(
            "Entity Type",
            options=['All', 'scenario', 'assumptions', 'customer', 'forecast', 'snapshot', 'share'],
            index=0
        )
    
    with col3:
        limit = st.number_input("Show last", value=50, min_value=10, max_value=500, step=10)
    
    # Get audit log
    if hasattr(db, 'get_audit_log'):
        audit_entries = db.get_audit_log(
            scenario_id=scenario_id,
            action=action_filter if action_filter != 'All' else None,
            entity_type=entity_filter if entity_filter != 'All' else None,
            limit=limit
        )
    else:
        audit_entries = []
        st.info("Audit logging not available. Run the Sprint 7 migration.")
    
    if not audit_entries:
        st.info("No audit entries found for the selected filters.")
        return
    
    # Display entries
    for entry in audit_entries:
        action = entry.get('action', '')
        entity = entry.get('entity_type', '')
        timestamp = entry.get('created_at', '')
        summary = entry.get('changes_summary', '')
        user = entry.get('user_id', '')[:8] + '...'
        
        # Action icon
        action_icons = {
            'create': '➕',
            'update': '✏️',
            'delete': '🗑️',
            'view': '👁️',
            'export': '📥',
            'share': '📤'
        }
        icon = action_icons.get(action, '📋')
        
        # Action color
        action_colors = {
            'create': '#10b981',
            'update': '#3b82f6',
            'delete': '#ef4444',
            'view': '#8b5cf6',
            'export': '#f59e0b',
            'share': '#D4A537'
        }
        color = action_colors.get(action, '#64748b')
        
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: flex-start;
            padding: 0.75rem;
            border-left: 3px solid {color};
            background: {color}10;
            border-radius: 0 8px 8px 0;
            margin-bottom: 0.5rem;
        ">
            <div style="font-size: 1.2rem; margin-right: 0.75rem;">{icon}</div>
            <div style="flex: 1;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: {color}; font-weight: 600;">
                        {action.upper()} {entity}
                    </span>
                    <span style="color: #64748b; font-size: 0.8rem;">
                        {str(timestamp)[:19]}
                    </span>
                </div>
                <div style="color: #B0B0B0; font-size: 0.9rem; margin-top: 0.25rem;">
                    {summary or f"User {user} performed {action} on {entity}"}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Export option
    if st.button("📥 Export Audit Log"):
        df = pd.DataFrame(audit_entries)
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"audit_log_{scenario_id[:8]}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )


# =============================================================================
# INVESTOR VIEW TAB
# =============================================================================

def render_investor_view_manager(db: SupabaseHandler, scenario_id: str, user_id: str):
    """Render investor view creation and management."""
    st.markdown("### 💼 Investor Views")
    
    st.markdown("""
    <div style="background: rgba(212,165,55,0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #D4A537; margin-bottom: 1rem;">
        <p style="margin: 0; color: #B0B0B0;">
            Create sanitized views of your scenario for investor due diligence. 
            Sensitive information like customer names and detailed costs are automatically masked.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create new investor view
    with st.expander("➕ Create Investor View", expanded=False):
        with st.form("investor_view_form"):
            view_name = st.text_input(
                "View Name",
                placeholder="Q4 2024 Due Diligence Pack"
            )
            
            st.markdown("**Sanitization Settings**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                hide_customers = st.checkbox("Hide customer names", value=True)
                hide_sites = st.checkbox("Hide site names", value=True)
                hide_costs = st.checkbox("Hide detailed costs", value=True)
            
            with col2:
                hide_margins = st.checkbox("Hide margin details", value=True)
                aggregate_small = st.checkbox("Aggregate small customers", value=True)
                threshold = st.slider("Aggregation threshold", 1, 10, 5, help="Customers below this % are aggregated")
            
            valid_days = st.number_input("Valid for (days)", value=30, min_value=1, max_value=365)
            
            submitted = st.form_submit_button("Create Investor View", type="primary")
            
            if submitted:
                if not view_name:
                    st.error("Please enter a view name")
                else:
                    settings = {
                        'hide_customer_names': hide_customers,
                        'hide_site_names': hide_sites,
                        'hide_detailed_costs': hide_costs,
                        'hide_margins': hide_margins,
                        'aggregate_small_customers': aggregate_small,
                        'aggregation_threshold': threshold / 100
                    }
                    
                    if hasattr(db, 'create_investor_view'):
                        result = db.create_investor_view(
                            scenario_id=scenario_id,
                            created_by=user_id,
                            view_name=view_name,
                            settings=settings
                        )
                        
                        if result:
                            st.success(f"✅ Investor view '{view_name}' created!")
                            st.rerun()
                        else:
                            st.error("Failed to create investor view")
                    else:
                        st.warning("Investor view functionality not available. Run the Sprint 7 migration.")
    
    # List existing views
    st.markdown("---")
    st.markdown("**Existing Investor Views**")
    
    if hasattr(db, 'get_investor_views'):
        views = db.get_investor_views(scenario_id)
    else:
        views = []
    
    if not views:
        st.info("No investor views created yet.")
    else:
        for view in views:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{view.get('view_name', 'Unnamed')}**")
                created = view.get('created_at', '')
                st.caption(f"Created: {str(created)[:10]} • Views: {view.get('view_count', 0)}")
            
            with col2:
                if view.get('is_active'):
                    st.markdown("🟢 Active")
                else:
                    st.markdown("🔴 Inactive")
            
            with col3:
                if st.button("📋 Copy Link", key=f"copy_{view.get('id')}"):
                    view_url = f"https://yourapp.com/investor-view/{view.get('id')}"
                    st.code(view_url)
            
            st.markdown("---")


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_user_management(
    db: SupabaseHandler,
    scenario_id: str,
    user_id: str
):
    """
    Main render function for User & Role Management.
    """
    # Branded header
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
        <div style="flex: 0 0 auto; margin-right: 1rem;">
            <span style="font-size: 2rem;">👥</span>
        </div>
        <div style="flex: 1;">
            <h1 style="color: #D4A537; margin: 0; font-size: 1.75rem; font-weight: 700;">Access Control</h1>
            <p style="color: #B0B0B0; margin: 0; font-size: 0.875rem;">Crusher Equipment Africa - Manage Scenario Access & Sharing</p>
        </div>
        <div style="flex: 0 0 auto;">
            <div style="display: flex; align-items: center;">
                <div style="width: 40px; height: 2px; background: linear-gradient(90deg, transparent, #D4A537);"></div>
                <span style="color: #D4A537; padding: 0 0.5rem;">🔒</span>
                <div style="width: 40px; height: 2px; background: linear-gradient(90deg, #D4A537, transparent);"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if user has permission to manage users
    can_manage = True  # Default to owner behavior
    if PERMISSIONS_AVAILABLE:
        from utils.permissions import PermissionChecker, Permissions
        checker = PermissionChecker(db, user_id, scenario_id)
        can_manage = checker.can(Permissions.MANAGE_USERS) or checker.is_owner
    
    if not can_manage:
        st.warning("🔒 You don't have permission to manage users for this scenario.")
        
        # Show read-only view of their own role
        st.markdown("### Your Access")
        if PERMISSIONS_AVAILABLE:
            render_permission_badge(checker)
        return
    
    # Main tabs
    tab_users, tab_share, tab_roles, tab_audit, tab_investor = st.tabs([
        "👥 Users",
        "📤 Share",
        "📜 Roles",
        "📝 Audit Log",
        "💼 Investor Views"
    ])
    
    with tab_users:
        render_current_users(db, scenario_id, user_id)
    
    with tab_share:
        render_share_scenario(db, scenario_id, user_id)
    
    with tab_roles:
        render_role_definitions(db)
    
    with tab_audit:
        render_audit_log(db, scenario_id, user_id)
    
    with tab_investor:
        render_investor_view_manager(db, scenario_id, user_id)
