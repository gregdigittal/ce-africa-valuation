"""
Pytest Configuration and Fixtures
==================================
Sprint 22: Shared test fixtures and configuration.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any
from unittest.mock import Mock, MagicMock
from datetime import datetime


@pytest.fixture
def mock_db_handler():
    """Create a mock database handler."""
    db = Mock()
    db.client = Mock()
    return db


@pytest.fixture
def sample_assumptions():
    """Sample assumptions dictionary for testing."""
    return {
        'forecast_duration_months': 60,
        'inflation_rate': 5,
        'gross_margin_liner': 38,
        'gross_margin_refurb': 32,
        'margin_consumable_pct': 0.38,
        'margin_refurb_pct': 0.32,
    }


@pytest.fixture
def sample_machines():
    """Sample machine data for testing."""
    return [
        {
            'id': 'machine_1',
            'machine_id': 'M001',
            'customer_name': 'Test Customer',
            'site_name': 'Test Site',
            'machine_model': 'Model A',
            'commission_date': '2024-01-01',
            'status': 'Active',
            'wear_profiles_v2': {
                'liner_life_months': 6,
                'avg_consumable_revenue': 50000,
                'refurb_interval_months': 36,
                'avg_refurb_revenue': 150000,
                'gross_margin_liner': 0.38,
                'gross_margin_refurb': 0.32
            }
        },
        {
            'id': 'machine_2',
            'machine_id': 'M002',
            'customer_name': 'Test Customer',
            'site_name': 'Test Site',
            'machine_model': 'Model B',
            'commission_date': '2024-02-01',
            'status': 'Active',
            'wear_profiles_v2': {
                'liner_life_months': 8,
                'avg_consumable_revenue': 60000,
                'refurb_interval_months': 48,
                'avg_refurb_revenue': 180000,
                'gross_margin_liner': 0.40,
                'gross_margin_refurb': 0.35
            }
        }
    ]


@pytest.fixture
def sample_prospects():
    """Sample prospect data for testing."""
    return [
        {
            'id': 'prospect_1',
            'scenario_id': 'test_scenario',
            'expected_close_date': '2024-06-01',
            'confidence_pct': 75,
            'annual_liner_value': 1000000,
            'refurb_value': 500000
        }
    ]


@pytest.fixture
def sample_expenses():
    """Sample expense assumptions for testing."""
    return [
        {
            'id': 'expense_1',
            'scenario_id': 'test_scenario',
            'is_active': True,
            'function_type': 'fixed',
            'fixed_monthly': 100000,
            'escalation_rate': 0.05
        },
        {
            'id': 'expense_2',
            'scenario_id': 'test_scenario',
            'is_active': True,
            'function_type': 'variable',
            'fixed_monthly': 50000,
            'variable_rate': 0.10,
            'escalation_rate': 0.03
        }
    ]


@pytest.fixture
def sample_forecast_data(sample_assumptions, sample_machines, sample_prospects, sample_expenses):
    """Complete sample forecast data structure."""
    return {
        'assumptions': sample_assumptions,
        'ai_assumptions': None,
        'machines': sample_machines,
        'prospects': sample_prospects,
        'expenses': sample_expenses,
        'data_source': 'machine_instances'
    }
