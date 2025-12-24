"""
Forecast Service
================
Business logic for forecast operations.
"""

from typing import Dict, Any, Optional, Callable
from forecast_engine import ForecastEngine
from components.forecast_section import load_forecast_data


class ForecastService:
    """
    Service for forecast business logic.
    
    Coordinates between data loading, forecast engine, and results.
    """
    
    def __init__(self, db_handler):
        """
        Initialize forecast service.
        
        Args:
            db_handler: Database handler instance (SupabaseHandler)
        """
        self.db = db_handler
        self.engine = ForecastEngine()
    
    def run_forecast(
        self,
        scenario_id: str,
        user_id: str,
        manufacturing_scenario: Optional[Any] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run forecast with business logic coordination.
        
        Args:
            scenario_id: Scenario ID
            user_id: User ID
            manufacturing_scenario: Optional manufacturing strategy
            progress_callback: Optional progress callback
        
        Returns:
            Forecast results dictionary
        """
        # Load forecast data (handles caching)
        if progress_callback:
            progress_callback(0.1, "Loading data...")
        
        data = load_forecast_data(self.db, scenario_id, user_id)
        
        # Run forecast using engine
        results = self.engine.run_forecast(
            data,
            manufacturing_scenario,
            progress_callback
        )
        
        # Add data_source to results
        results['data_source'] = data.get('data_source', 'unknown')
        
        return results
    
    def validate_forecast_prerequisites(
        self,
        scenario_id: str,
        user_id: str
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that prerequisites for running forecast are met.
        
        Args:
            scenario_id: Scenario ID
            user_id: User ID
        
        Returns:
            (is_valid, error_message)
        """
        data = load_forecast_data(self.db, scenario_id, user_id)
        
        if not data.get('assumptions'):
            return False, "No assumptions configured. Please complete setup first."
        
        if not data.get('machines'):
            return False, "No machines found in fleet. Please import fleet data first."
        
        return True, None
