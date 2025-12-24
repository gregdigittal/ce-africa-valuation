"""
Integration Tests
=================
Sprint 22: End-to-end and integration tests.
"""

import pytest
from unittest.mock import Mock, patch
from forecast_engine import ForecastEngine
from services.forecast_service import ForecastService
from services.scenario_service import ScenarioService
from tests.conftest import sample_forecast_data


@pytest.mark.integration
class TestForecastIntegration:
    """Integration tests for forecast workflow."""
    
    def test_forecast_service_with_engine(self, mock_db_handler, sample_forecast_data):
        """Test ForecastService integration with ForecastEngine."""
        with patch('services.forecast_service.load_forecast_data', return_value=sample_forecast_data):
            service = ForecastService(mock_db_handler)
            results = service.run_forecast('scenario_1', 'user_1')
            
            assert results is not None
            assert results['success'] is True
            assert 'revenue' in results
            assert 'costs' in results
            assert 'profit' in results
    
    def test_forecast_validation_workflow(self, mock_db_handler):
        """Test forecast validation workflow."""
        with patch('services.forecast_service.load_forecast_data', return_value=sample_forecast_data):
            service = ForecastService(mock_db_handler)
            
            # Valid prerequisites
            is_valid, error = service.validate_forecast_prerequisites('scenario_1', 'user_1')
            assert is_valid is True
            
            # Invalid prerequisites (missing assumptions)
            invalid_data = {'assumptions': None, 'machines': [{'id': 'test'}]}
            with patch('services.forecast_service.load_forecast_data', return_value=invalid_data):
                is_valid, error = service.validate_forecast_prerequisites('scenario_1', 'user_1')
                assert is_valid is False


@pytest.mark.integration
class TestServiceIntegration:
    """Integration tests for service layer."""
    
    def test_scenario_workflow(self, mock_db_handler):
        """Test complete scenario workflow."""
        service = ScenarioService(mock_db_handler)
        
        # Mock database responses
        mock_db_handler.create_scenario.return_value = {'id': 'new_scenario', 'name': 'New Scenario'}
        mock_db_handler.get_user_scenarios.return_value = [
            {'id': 'new_scenario', 'name': 'New Scenario'}
        ]
        
        # Create scenario
        scenario = service.create_scenario('user_1', 'New Scenario')
        assert scenario is not None
        
        # Get scenario
        retrieved = service.get_scenario('new_scenario', 'user_1')
        assert retrieved is not None
        assert retrieved['name'] == 'New Scenario'
