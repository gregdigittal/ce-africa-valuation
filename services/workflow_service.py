"""
Workflow Service
================
Business logic for workflow state management.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class WorkflowService:
    """
    Service for workflow business logic.
    
    Handles workflow stage completion, prerequisites, and state management.
    """
    
    def __init__(self, db_handler):
        """
        Initialize workflow service.
        
        Args:
            db_handler: Database handler instance (SupabaseHandler)
        """
        self.db = db_handler
    
    def mark_stage_complete(
        self,
        scenario_id: str,
        stage_id: str,
        user_id: str
    ) -> bool:
        """
        Mark a workflow stage as complete.
        
        Args:
            scenario_id: Scenario ID
            stage_id: Stage ID
            user_id: User ID
        
        Returns:
            True if marked successfully
        """
        try:
            if hasattr(self.db, 'client'):
                data = {
                    'scenario_id': scenario_id,
                    'stage': stage_id,
                    'completed': True,
                    'completed_at': datetime.now().isoformat(),
                    'user_id': user_id
                }
                
                try:
                    self.db.client.table('workflow_progress').upsert(
                        data, on_conflict='scenario_id,stage'
                    ).execute()
                except:
                    try:
                        self.db.client.table('workflow_progress').insert(data).execute()
                    except:
                        return False
                
                return True
        except Exception:
            return False
    
    def is_stage_complete(
        self,
        scenario_id: str,
        stage_id: str,
        user_id: str
    ) -> bool:
        """
        Check if a workflow stage is complete.
        
        Args:
            scenario_id: Scenario ID
            stage_id: Stage ID
            user_id: User ID
        
        Returns:
            True if stage is complete
        """
        try:
            if hasattr(self.db, 'client'):
                result = self.db.client.table('workflow_progress').select('completed').eq(
                    'scenario_id', scenario_id
                ).eq('stage', stage_id).execute()
                
                if result.data and len(result.data) > 0:
                    return result.data[0].get('completed', False)
        except Exception:
            pass
        
        return False
    
    def get_workflow_progress(
        self,
        scenario_id: str,
        user_id: str
    ) -> Dict[str, bool]:
        """
        Get workflow progress for a scenario.
        
        Args:
            scenario_id: Scenario ID
            user_id: User ID
        
        Returns:
            Dictionary mapping stage_id to completion status
        """
        progress = {}
        
        try:
            if hasattr(self.db, 'client'):
                result = self.db.client.table('workflow_progress').select('stage,completed').eq(
                    'scenario_id', scenario_id
                ).eq('user_id', user_id).execute()
                
                if result.data:
                    for row in result.data:
                        progress[row['stage']] = row.get('completed', False)
        except Exception:
            pass
        
        return progress
    
    def check_prerequisites(
        self,
        scenario_id: str,
        stage_id: str,
        prerequisites: List[str],
        user_id: str
    ) -> tuple[bool, List[str]]:
        """
        Check if prerequisites for a stage are met.
        
        Args:
            scenario_id: Scenario ID
            stage_id: Stage ID
            prerequisites: List of prerequisite stage IDs
            user_id: User ID
        
        Returns:
            (all_met, missing_prerequisites)
        """
        missing = []
        
        for prereq_id in prerequisites:
            if not self.is_stage_complete(scenario_id, prereq_id, user_id):
                missing.append(prereq_id)
        
        return len(missing) == 0, missing
