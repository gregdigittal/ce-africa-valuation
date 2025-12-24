"""
Supabase Database Utilities
Handles connection and RLS-aware queries
"""
import streamlit as st
from supabase import create_client, Client
from typing import Optional

def get_supabase_client() -> Client:
    """Get Supabase client from Streamlit secrets"""
    try:
        url = st.secrets["supabase"]["url"]
        # Prefer anon key, otherwise service_role_key or fallback 'key'
        supa = st.secrets["supabase"]
        if "anon_key" in supa:
            key = supa["anon_key"]
        elif "service_role_key" in supa:
            key = supa["service_role_key"]
        elif "key" in supa:
            key = supa["key"]
        else:
            raise KeyError("No Supabase key found (anon_key/service_role_key/key)")
        return create_client(url, key)
    except KeyError as e:
        st.error(f"Missing Supabase configuration in secrets: {e}")
        # Return None so callers can fail gracefully
        return None
    except Exception as e:
        st.error(f"Error connecting to Supabase: {e}")
        return None

def get_user_id() -> str:
    """
    Returns the authenticated user ID.
    For Phase 1 Dev (pre-login screen), we return a fixed Test UUID.
    """
    # 1. Check session state first
    if "user_id" in st.session_state and st.session_state.user_id:
        return st.session_state.user_id
    
    # 2. Fallback check for dev secrets (optional)
    try:
        dev_user_id = st.secrets.get("dev", {}).get("user_id")
        if dev_user_id:
            st.session_state.user_id = dev_user_id
            return dev_user_id
    except:
        pass
    
    # 3. Fallback: try to read first profile from DB (if client available)
    client = get_supabase_client()
    if client:
        try:
            resp = client.table("profiles").select("id").limit(1).execute()
            if resp.data and len(resp.data) > 0 and resp.data[0].get("id"):
                db_user_id = resp.data[0]["id"]
                st.session_state.user_id = db_user_id
                return db_user_id
        except Exception:
            pass
    
    # 4. DEV MODE: Return the Nil UUID
    # This ensures we never send 'None' to a database column requiring UUID
    NIL_UUID = "00000000-0000-0000-0000-000000000000"
    st.session_state.user_id = NIL_UUID
    
    return NIL_UUID

def set_user_id(user_id: str):
    """Set user ID in session state"""
    st.session_state.user_id = user_id