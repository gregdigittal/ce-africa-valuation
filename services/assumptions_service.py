"""
Assumptions Service
===================
Business logic for assumptions management.
"""

from typing import Dict, Any, Optional
from components.ai_assumptions_engine import (
    load_assumptions_from_db,
    save_assumptions_to_db,
    AssumptionsSet
)


class AssumptionsService:
    """
    Service for assumptions business logic.
    
    Handles AI assumptions, manual assumptions, and merging logic.
    """
    
    def __init__(self, db_handler):
        """
        Initialize assumptions service.
        
        Args:
            db_handler: Database handler instance (SupabaseHandler)
        """
        self.db = db_handler
    
    def get_assumptions(
        self,
        scenario_id: str,
        user_id: str,
        include_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Get assumptions for a scenario.
        
        Args:
            scenario_id: Scenario ID
            user_id: User ID
            include_ai: Whether to include AI assumptions
        
        Returns:
            Dictionary with manual and optionally AI assumptions
        """
        # Get manual assumptions
        manual_assumptions = self.db.get_scenario_assumptions(scenario_id, user_id)
        
        result = {
            'manual': manual_assumptions,
            'ai': None
        }
        
        if include_ai:
            # Get AI assumptions
            try:
                ai_assumptions = load_assumptions_from_db(self.db, scenario_id, user_id)
                result['ai'] = ai_assumptions
            except Exception:
                pass  # AI assumptions not available
        
        return result
    
    def save_assumptions(
        self,
        scenario_id: str,
        user_id: str,
        assumptions: Dict[str, Any],
        assumption_type: str = "manual"
    ) -> bool:
        """
        Save assumptions with business logic validation.
        
        Args:
            scenario_id: Scenario ID
            user_id: User ID
            assumptions: Assumptions dictionary
            assumption_type: Type of assumptions ("manual" or "ai")
        
        Returns:
            True if saved successfully, False otherwise
        """
        if assumption_type == "ai":
            # Save AI assumptions
            if isinstance(assumptions, AssumptionsSet):
                return save_assumptions_to_db(self.db, assumptions, user_id)
            else:
                # Convert dict to AssumptionsSet if needed
                try:
                    assumptions_set = AssumptionsSet.from_dict(assumptions)
                    return save_assumptions_to_db(self.db, assumptions_set, user_id)
                except Exception:
                    return False
        else:
            # Save manual assumptions
            return self.db.update_assumptions(scenario_id, user_id, assumptions)
    
    def merge_assumptions(
        self,
        manual: Dict[str, Any],
        ai: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Merge manual and AI assumptions with business logic.
        
        Args:
            manual: Manual assumptions dictionary
            ai: Optional AI assumptions (AssumptionsSet)
        
        Returns:
            Merged assumptions dictionary
        """
        merged = manual.copy()
        
        if ai:
            # Extract AI assumptions if AssumptionsSet
            if hasattr(ai, 'assumptions'):
                # AI assumptions take precedence for specific fields
                # Business logic: AI assumptions override manual for derived values
                for key, assumption in ai.assumptions.items():
                    if hasattr(assumption, 'final_static_value'):
                        merged[key] = assumption.final_static_value
        
        return merged
