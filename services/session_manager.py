"""
Session State Manager
=====================
Sprint 21: Centralized session state management.

Provides unified interface for session state operations with validation,
persistence, and migration support.
"""

import streamlit as st
from typing import Any, Optional, Dict, List
from datetime import datetime
import json


class SessionManager:
    """
    Centralized session state manager.
    
    Provides validation, persistence, and migration support for session state.
    """
    
    # Session state keys (centralized constants)
    SCENARIO_ID = 'scenario_id'
    USER_ID = 'user_id'
    DB_HANDLER = 'db_handler'
    FORECAST_RESULTS = 'forecast_results'
    AI_ASSUMPTIONS = 'ai_assumptions_set'
    MANUFACTURING_SCENARIO = 'vi_scenario'
    CURRENT_SECTION = 'current_section'
    WORKFLOW_PROGRESS = 'workflow_progress'
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get value from session state.
        
        Args:
            key: Session state key
            default: Default value if key doesn't exist
        
        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value: Any) -> None:
        """
        Set value in session state.
        
        Args:
            key: Session state key
            value: Value to set
        """
        st.session_state[key] = value
    
    @staticmethod
    def delete(key: str) -> None:
        """
        Delete key from session state.
        
        Args:
            key: Session state key
        """
        if key in st.session_state:
            del st.session_state[key]
    
    @staticmethod
    def exists(key: str) -> bool:
        """
        Check if key exists in session state.
        
        Args:
            key: Session state key
        
        Returns:
            True if key exists
        """
        return key in st.session_state
    
    @staticmethod
    def get_scenario_id() -> Optional[str]:
        """Get current scenario ID."""
        return SessionManager.get(SessionManager.SCENARIO_ID)
    
    @staticmethod
    def set_scenario_id(scenario_id: str) -> None:
        """Set current scenario ID."""
        SessionManager.set(SessionManager.SCENARIO_ID, scenario_id)
    
    @staticmethod
    def get_user_id() -> Optional[str]:
        """Get current user ID."""
        return SessionManager.get(SessionManager.USER_ID)
    
    @staticmethod
    def set_user_id(user_id: str) -> None:
        """Set current user ID."""
        SessionManager.set(SessionManager.USER_ID, user_id)
    
    @staticmethod
    def get_forecast_results(scenario_id: Optional[str] = None) -> Optional[Dict]:
        """
        Get forecast results from session state.
        
        Args:
            scenario_id: Optional scenario ID (uses current if not provided)
        
        Returns:
            Forecast results dictionary or None
        """
        if scenario_id:
            key = f'forecast_results_{scenario_id}'
        else:
            key = SessionManager.FORECAST_RESULTS
        
        return SessionManager.get(key)
    
    @staticmethod
    def set_forecast_results(
        results: Dict,
        scenario_id: Optional[str] = None
    ) -> None:
        """
        Set forecast results in session state.
        
        Args:
            results: Forecast results dictionary
            scenario_id: Optional scenario ID (uses current if not provided)
        """
        if scenario_id:
            key = f'forecast_results_{scenario_id}'
        else:
            key = SessionManager.FORECAST_RESULTS
        
        SessionManager.set(key, results)
    
    @staticmethod
    def clear_forecast_results(scenario_id: Optional[str] = None) -> None:
        """
        Clear forecast results from session state.
        
        Args:
            scenario_id: Optional scenario ID (uses current if not provided)
        """
        if scenario_id:
            key = f'forecast_results_{scenario_id}'
        else:
            key = SessionManager.FORECAST_RESULTS
        
        SessionManager.delete(key)
    
    @staticmethod
    def get_workflow_progress(scenario_id: Optional[str] = None) -> Dict[str, bool]:
        """
        Get workflow progress from session state.
        
        Args:
            scenario_id: Optional scenario ID
        
        Returns:
            Dictionary mapping stage_id to completion status
        """
        if scenario_id:
            key = f'workflow_progress_{scenario_id}'
        else:
            key = SessionManager.WORKFLOW_PROGRESS
        
        return SessionManager.get(key, {})
    
    @staticmethod
    def set_workflow_progress(
        progress: Dict[str, bool],
        scenario_id: Optional[str] = None
    ) -> None:
        """
        Set workflow progress in session state.
        
        Args:
            progress: Dictionary mapping stage_id to completion status
            scenario_id: Optional scenario ID
        """
        if scenario_id:
            key = f'workflow_progress_{scenario_id}'
        else:
            key = SessionManager.WORKFLOW_PROGRESS
        
        SessionManager.set(key, progress)
    
    @staticmethod
    def mark_workflow_stage(
        stage_id: str,
        completed: bool,
        scenario_id: Optional[str] = None
    ) -> None:
        """
        Mark a workflow stage as complete or incomplete.
        
        Args:
            stage_id: Stage ID
            completed: Completion status
            scenario_id: Optional scenario ID
        """
        progress = SessionManager.get_workflow_progress(scenario_id)
        progress[stage_id] = completed
        SessionManager.set_workflow_progress(progress, scenario_id)
    
    @staticmethod
    def clear_cache(prefix: Optional[str] = None) -> None:
        """
        Clear cached data from session state.
        
        Args:
            prefix: Optional prefix to filter keys (e.g., 'forecast_data_')
        """
        keys_to_delete = []
        
        if prefix:
            for key in st.session_state.keys():
                if key.startswith(prefix):
                    keys_to_delete.append(key)
        else:
            # Clear common cache keys
            cache_prefixes = [
                'forecast_data_',
                'ai_assumptions_',
                'workflow_progress_',
            ]
            for key in st.session_state.keys():
                for prefix in cache_prefixes:
                    if key.startswith(prefix):
                        keys_to_delete.append(key)
                        break
        
        for key in keys_to_delete:
            SessionManager.delete(key)
    
    @staticmethod
    def validate_state() -> tuple[bool, List[str]]:
        """
        Validate session state integrity.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required keys
        if not SessionManager.exists(SessionManager.SCENARIO_ID):
            issues.append("Missing scenario_id")
        
        if not SessionManager.exists(SessionManager.USER_ID):
            issues.append("Missing user_id")
        
        # Validate data types
        scenario_id = SessionManager.get_scenario_id()
        if scenario_id and not isinstance(scenario_id, str):
            issues.append("scenario_id must be a string")
        
        user_id = SessionManager.get_user_id()
        if user_id and not isinstance(user_id, str):
            issues.append("user_id must be a string")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def migrate_state(from_version: str, to_version: str) -> bool:
        """
        Migrate session state between versions.
        
        Args:
            from_version: Source version
            to_version: Target version
        
        Returns:
            True if migration successful
        """
        # Placeholder for future migration logic
        # This would handle breaking changes in session state structure
        try:
            # Example: Migrate old key names to new ones
            if from_version < "2.0" and to_version >= "2.0":
                # Migration logic here
                pass
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_state_summary() -> Dict[str, Any]:
        """
        Get summary of current session state.
        
        Returns:
            Dictionary with state summary
        """
        return {
            'scenario_id': SessionManager.get_scenario_id(),
            'user_id': SessionManager.get_user_id(),
            'current_section': SessionManager.get(SessionManager.CURRENT_SECTION),
            'has_forecast_results': SessionManager.get_forecast_results() is not None,
            'has_ai_assumptions': SessionManager.exists(SessionManager.AI_ASSUMPTIONS),
            'has_manufacturing': SessionManager.exists(SessionManager.MANUFACTURING_SCENARIO),
            'workflow_progress': SessionManager.get_workflow_progress(),
            'total_keys': len(st.session_state)
        }
