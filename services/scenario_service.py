"""
Scenario Service
================
Business logic for scenario management.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime


class ScenarioService:
    """
    Service for scenario business logic.
    
    Separates business rules from database operations.
    """
    
    def __init__(self, db_handler):
        """
        Initialize scenario service.
        
        Args:
            db_handler: Database handler instance (SupabaseHandler)
        """
        self.db = db_handler
    
    def get_user_scenarios(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all scenarios for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            List of scenario dictionaries
        """
        return self.db.get_user_scenarios(user_id)
    
    def create_scenario(
        self,
        user_id: str,
        name: str,
        status: str = "draft"
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new scenario with business logic validation.
        
        Args:
            user_id: User ID
            name: Scenario name
            status: Initial status (default: "draft")
        
        Returns:
            Created scenario dictionary or None if failed
        """
        # Business logic: Validate name
        if not name or not name.strip():
            return None
        
        # Business logic: Validate status
        valid_statuses = ["draft", "active", "archived"]
        if status not in valid_statuses:
            status = "draft"
        
        return self.db.create_scenario(user_id, name, status)
    
    def update_scenario(
        self,
        scenario_id: str,
        user_id: str,
        name: Optional[str] = None,
        status: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update scenario with business logic validation.
        
        Args:
            scenario_id: Scenario ID
            user_id: User ID
            name: New name (optional)
            status: New status (optional)
        
        Returns:
            Updated scenario dictionary or None if failed
        """
        # Business logic: Validate name if provided
        if name is not None and (not name or not name.strip()):
            return None
        
        # Business logic: Validate status if provided
        if status is not None:
            valid_statuses = ["draft", "active", "archived"]
            if status not in valid_statuses:
                status = None  # Don't update invalid status
        
        return self.db.update_scenario(scenario_id, user_id, name, status)
    
    def delete_scenario(self, scenario_id: str, user_id: str) -> bool:
        """
        Delete scenario with business logic checks.
        
        Args:
            scenario_id: Scenario ID
            user_id: User ID
        
        Returns:
            True if deleted, False otherwise
        """
        # Business logic: Could add checks here (e.g., prevent deletion of active scenarios)
        return self.db.delete_scenario(scenario_id, user_id)
    
    def get_scenario(self, scenario_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific scenario.
        
        Args:
            scenario_id: Scenario ID
            user_id: User ID
        
        Returns:
            Scenario dictionary or None if not found
        """
        scenarios = self.get_user_scenarios(user_id)
        for scenario in scenarios:
            if scenario.get('id') == scenario_id:
                return scenario
        return None
