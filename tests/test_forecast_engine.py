"""
Unit Tests for Forecast Engine
===============================
Sprint 22: Tests for the extracted forecast engine.
"""

import pytest
import numpy as np
from forecast_engine import ForecastEngine
from tests.conftest import sample_forecast_data


class TestForecastEngine:
    """Test suite for ForecastEngine."""
    
    def test_engine_initialization(self):
        """Test that engine initializes correctly."""
        engine = ForecastEngine()
        assert engine is not None
    
    def test_run_forecast_basic(self, sample_forecast_data):
        """Test basic forecast execution."""
        engine = ForecastEngine()
        results = engine.run_forecast(sample_forecast_data)
        
        assert results is not None
        assert 'success' in results
        assert 'timeline' in results
        assert 'revenue' in results
        assert 'costs' in results
        assert 'profit' in results
    
    def test_run_forecast_success(self, sample_forecast_data):
        """Test successful forecast execution."""
        engine = ForecastEngine()
        results = engine.run_forecast(sample_forecast_data)
        
        assert results['success'] is True
        assert results['error'] is None
        assert len(results['timeline']) > 0
    
    def test_run_forecast_missing_assumptions(self):
        """Test forecast with missing assumptions."""
        engine = ForecastEngine()
        data = {
            'assumptions': None,
            'machines': [{'id': 'test'}],
            'prospects': [],
            'expenses': []
        }
        
        results = engine.run_forecast(data)
        
        assert results['success'] is False
        assert results['error'] is not None
        assert 'assumptions' in results['error'].lower()
    
    def test_run_forecast_missing_machines(self, sample_assumptions):
        """Test forecast with missing machines."""
        engine = ForecastEngine()
        data = {
            'assumptions': sample_assumptions,
            'machines': [],
            'prospects': [],
            'expenses': []
        }
        
        results = engine.run_forecast(data)
        
        assert results['success'] is False
        assert results['error'] is not None
        assert 'machines' in results['error'].lower()
    
    def test_revenue_calculation(self, sample_forecast_data):
        """Test revenue calculation."""
        engine = ForecastEngine()
        results = engine.run_forecast(sample_forecast_data)
        
        assert results['success'] is True
        revenue = results['revenue']
        
        assert 'consumables' in revenue
        assert 'refurb' in revenue
        assert 'pipeline' in revenue
        assert 'total' in revenue
        
        # Check that revenue arrays have correct length
        timeline_length = len(results['timeline'])
        assert len(revenue['consumables']) == timeline_length
        assert len(revenue['refurb']) == timeline_length
        assert len(revenue['pipeline']) == timeline_length
        assert len(revenue['total']) == timeline_length
    
    def test_cogs_calculation(self, sample_forecast_data):
        """Test COGS calculation."""
        engine = ForecastEngine()
        results = engine.run_forecast(sample_forecast_data)
        
        assert results['success'] is True
        costs = results['costs']
        
        assert 'cogs' in costs
        assert 'opex' in costs
        assert 'total' in costs
        
        # Check that cost arrays have correct length
        timeline_length = len(results['timeline'])
        assert len(costs['cogs']) == timeline_length
        assert len(costs['opex']) == timeline_length
    
    def test_profit_calculation(self, sample_forecast_data):
        """Test profit calculation."""
        engine = ForecastEngine()
        results = engine.run_forecast(sample_forecast_data)
        
        assert results['success'] is True
        profit = results['profit']
        
        assert 'gross' in profit
        assert 'ebit' in profit
        
        # Check that profit arrays have correct length
        timeline_length = len(results['timeline'])
        assert len(profit['gross']) == timeline_length
        assert len(profit['ebit']) == timeline_length
    
    def test_summary_statistics(self, sample_forecast_data):
        """Test summary statistics calculation."""
        engine = ForecastEngine()
        results = engine.run_forecast(sample_forecast_data)
        
        assert results['success'] is True
        summary = results['summary']
        
        assert 'total_revenue' in summary
        assert 'total_cogs' in summary
        assert 'total_opex' in summary
        assert 'total_gross_profit' in summary
        assert 'total_ebit' in summary
        assert 'avg_gross_margin' in summary
        assert 'avg_ebit_margin' in summary
        
        # Check that margins are reasonable (0-1 range)
        assert 0 <= summary['avg_gross_margin'] <= 1
        assert 0 <= summary['avg_ebit_margin'] <= 1
    
    def test_timeline_generation(self, sample_forecast_data):
        """Test timeline generation."""
        engine = ForecastEngine()
        results = engine.run_forecast(sample_forecast_data)
        
        assert results['success'] is True
        assert len(results['timeline']) > 0
        assert len(results['timeline_dates']) > 0
        assert len(results['timeline']) == len(results['timeline_dates'])
        
        # Check timeline format (YYYY-MM)
        for month in results['timeline']:
            assert len(month) == 7  # YYYY-MM format
            assert month[4] == '-'  # Dash separator
    
    def test_progress_callback(self, sample_forecast_data):
        """Test progress callback functionality."""
        progress_calls = []
        
        def progress_callback(progress, message):
            progress_calls.append((progress, message))
        
        engine = ForecastEngine()
        results = engine.run_forecast(sample_forecast_data, progress_callback=progress_callback)
        
        assert results['success'] is True
        assert len(progress_calls) > 0
        
        # Check that progress increases
        progress_values = [p[0] for p in progress_calls]
        assert progress_values[0] >= 0
        assert progress_values[-1] <= 1.0
