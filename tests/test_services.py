"""
Unit Tests for Services
========================
Sprint 22: Tests for service layer classes.
"""

import pytest
from unittest.mock import Mock, MagicMock
from services.scenario_service import ScenarioService
from services.forecast_service import ForecastService
from services.assumptions_service import AssumptionsService
from services.workflow_service import WorkflowService
from services.session_manager import SessionManager


class TestScenarioService:
    """Test suite for ScenarioService."""
    
    def test_service_initialization(self, mock_db_handler):
        """Test service initialization."""
        service = ScenarioService(mock_db_handler)
        assert service is not None
        assert service.db == mock_db_handler
    
    def test_create_scenario_valid(self, mock_db_handler):
        """Test creating a valid scenario."""
        mock_db_handler.create_scenario.return_value = {'id': 'test_id', 'name': 'Test Scenario'}
        service = ScenarioService(mock_db_handler)
        
        result = service.create_scenario('user_1', 'Test Scenario', 'draft')
        
        assert result is not None
        assert result['id'] == 'test_id'
        mock_db_handler.create_scenario.assert_called_once()
    
    def test_create_scenario_invalid_name(self, mock_db_handler):
        """Test creating scenario with invalid name."""
        service = ScenarioService(mock_db_handler)
        
        result = service.create_scenario('user_1', '', 'draft')
        
        assert result is None
        mock_db_handler.create_scenario.assert_not_called()
    
    def test_create_scenario_invalid_status(self, mock_db_handler):
        """Test creating scenario with invalid status."""
        mock_db_handler.create_scenario.return_value = {'id': 'test_id', 'name': 'Test'}
        service = ScenarioService(mock_db_handler)
        
        result = service.create_scenario('user_1', 'Test', 'invalid_status')
        
        # Should default to 'draft'
        assert result is not None
        mock_db_handler.create_scenario.assert_called_once_with('user_1', 'Test', 'draft')
    
    def test_update_scenario(self, mock_db_handler):
        """Test updating scenario."""
        mock_db_handler.update_scenario.return_value = {'id': 'test_id', 'name': 'Updated'}
        service = ScenarioService(mock_db_handler)
        
        result = service.update_scenario('test_id', 'user_1', name='Updated Name')
        
        assert result is not None
        mock_db_handler.update_scenario.assert_called_once()
    
    def test_delete_scenario(self, mock_db_handler):
        """Test deleting scenario."""
        mock_db_handler.delete_scenario.return_value = True
        service = ScenarioService(mock_db_handler)
        
        result = service.delete_scenario('test_id', 'user_1')
        
        assert result is True
        mock_db_handler.delete_scenario.assert_called_once_with('test_id', 'user_1')


class TestForecastService:
    """Test suite for ForecastService."""
    
    def test_service_initialization(self, mock_db_handler):
        """Test service initialization."""
        service = ForecastService(mock_db_handler)
        assert service is not None
        assert service.db == mock_db_handler
        assert service.engine is not None
    
    def test_validate_prerequisites_valid(self, mock_db_handler, sample_forecast_data):
        """Test prerequisite validation with valid data."""
        # Mock load_forecast_data
        from unittest.mock import patch
        with patch('services.forecast_service.load_forecast_data', return_value=sample_forecast_data):
            service = ForecastService(mock_db_handler)
            is_valid, error = service.validate_forecast_prerequisites('scenario_1', 'user_1')
            
            assert is_valid is True
            assert error is None
    
    def test_validate_prerequisites_missing_assumptions(self, mock_db_handler):
        """Test prerequisite validation with missing assumptions."""
        from unittest.mock import patch
        data = {'assumptions': None, 'machines': [{'id': 'test'}]}
        
        with patch('services.forecast_service.load_forecast_data', return_value=data):
            service = ForecastService(mock_db_handler)
            is_valid, error = service.validate_forecast_prerequisites('scenario_1', 'user_1')
            
            assert is_valid is False
            assert error is not None
            assert 'assumptions' in error.lower()


class TestWorkflowService:
    """Test suite for WorkflowService."""
    
    def test_service_initialization(self, mock_db_handler):
        """Test service initialization."""
        service = WorkflowService(mock_db_handler)
        assert service is not None
        assert service.db == mock_db_handler
    
    def test_mark_stage_complete(self, mock_db_handler):
        """Test marking stage as complete."""
        mock_db_handler.client.table.return_value.upsert.return_value.execute.return_value = True
        service = WorkflowService(mock_db_handler)
        
        result = service.mark_stage_complete('scenario_1', 'setup', 'user_1')
        
        assert result is True
    
    def test_is_stage_complete(self, mock_db_handler):
        """Test checking if stage is complete."""
        mock_result = Mock()
        mock_result.data = [{'completed': True}]
        mock_db_handler.client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_result
        
        service = WorkflowService(mock_db_handler)
        result = service.is_stage_complete('scenario_1', 'setup', 'user_1')
        
        assert result is True
    
    def test_check_prerequisites(self, mock_db_handler):
        """Test prerequisite checking."""
        # Mock is_stage_complete to return True for all
        service = WorkflowService(mock_db_handler)
        service.is_stage_complete = Mock(return_value=True)
        
        all_met, missing = service.check_prerequisites(
            'scenario_1', 'forecast', ['setup', 'ai_analysis'], 'user_1'
        )
        
        assert all_met is True
        assert len(missing) == 0


class TestSessionManager:
    """Test suite for SessionManager."""
    
    def test_get_set_basic(self):
        """Test basic get/set operations."""
        import streamlit as st
        
        # Note: This test requires Streamlit context
        # In practice, these would be integration tests
        SessionManager.set('test_key', 'test_value')
        value = SessionManager.get('test_key')
        
        assert value == 'test_value'
    
    def test_delete_key(self):
        """Test deleting a key."""
        SessionManager.set('test_key', 'test_value')
        SessionManager.delete('test_key')
        
        assert not SessionManager.exists('test_key')
    
    def test_exists_check(self):
        """Test existence check."""
        SessionManager.set('test_key', 'test_value')
        
        assert SessionManager.exists('test_key')
        assert not SessionManager.exists('nonexistent_key')
