"""
Assumption Storage Migration Utility
=====================================
Consolidates assumption storage from multiple locations into unified_line_item_config.

Technical Debt Resolution: December 20, 2025

Current Storage Locations:
1. forecast_configs - Element-level trend configs from legacy Trend Forecast tab
2. ai_assumptions - AI-generated distribution parameters
3. unified_line_item_config - New unified line-item configuration

Target: All should be consolidated into unified_line_item_config
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json


def analyze_assumption_storage(db, scenario_id: str, user_id: str) -> Dict[str, Any]:
    """
    Analyze current assumption storage and identify what needs migration.
    
    Returns:
        Dict with storage analysis
    """
    analysis = {
        'has_forecast_configs': False,
        'has_ai_assumptions': False,
        'has_unified_config': False,
        'forecast_configs_count': 0,
        'ai_assumptions_count': 0,
        'unified_items_count': 0,
        'needs_migration': False,
        'conflicts': []
    }
    
    try:
        assumptions = db.get_scenario_assumptions(scenario_id, user_id)
        if not assumptions:
            return analysis
        
        # Check forecast_configs
        if 'forecast_configs' in assumptions and assumptions['forecast_configs']:
            analysis['has_forecast_configs'] = True
            analysis['forecast_configs_count'] = len(assumptions['forecast_configs'])
        
        # Check ai_assumptions
        if 'ai_assumptions' in assumptions and assumptions['ai_assumptions']:
            ai_data = assumptions['ai_assumptions']
            if isinstance(ai_data, dict) and 'assumptions' in ai_data:
                analysis['has_ai_assumptions'] = True
                analysis['ai_assumptions_count'] = len(ai_data.get('assumptions', {}))
        
        # Check unified_line_item_config
        if 'unified_line_item_config' in assumptions and assumptions['unified_line_item_config']:
            unified = assumptions['unified_line_item_config']
            if isinstance(unified, dict) and 'line_items' in unified:
                analysis['has_unified_config'] = True
                analysis['unified_items_count'] = len(unified.get('line_items', {}))
        
        # Determine if migration is needed
        analysis['needs_migration'] = (
            (analysis['has_forecast_configs'] or analysis['has_ai_assumptions'])
            and not analysis['has_unified_config']
        )
        
        # Check for potential conflicts
        if analysis['has_unified_config'] and (analysis['has_forecast_configs'] or analysis['has_ai_assumptions']):
            analysis['conflicts'].append("Multiple storage locations have data - unified_config should be the source of truth")
        
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis


def migrate_to_unified_config(db, scenario_id: str, user_id: str, dry_run: bool = True) -> Dict[str, Any]:
    """
    Migrate assumptions from legacy storage to unified_line_item_config.
    
    Args:
        db: Database connector
        scenario_id: Scenario ID
        user_id: User ID
        dry_run: If True, only analyze without making changes
        
    Returns:
        Migration result
    """
    result = {
        'success': False,
        'dry_run': dry_run,
        'items_migrated': 0,
        'items_created': [],
        'errors': [],
        'warnings': []
    }
    
    try:
        assumptions = db.get_scenario_assumptions(scenario_id, user_id)
        if not assumptions:
            result['errors'].append("No assumptions found for scenario")
            return result
        
        # Get or create unified config structure
        unified_config = assumptions.get('unified_line_item_config', {})
        if not unified_config:
            unified_config = {
                'scenario_id': scenario_id,
                'line_items': {},
                'forecast_method': assumptions.get('forecast_method', 'pipeline'),
                'last_updated': '',
                'version': '1.0'
            }
        
        line_items = unified_config.get('line_items', {})
        
        # =======================================================================
        # Migrate from forecast_configs (legacy trend config)
        # =======================================================================
        forecast_configs = assumptions.get('forecast_configs', {})
        for element_key, config in forecast_configs.items():
            if not config:
                continue
            
            # Create line item if doesn't exist
            if element_key not in line_items:
                line_items[element_key] = {
                    'line_item_name': element_key.replace('_', ' ').title(),
                    'category': _infer_category(element_key),
                    'statement_type': 'income_statement',
                    'trend_type': 'linear',
                    'trend_growth_rate': 0.0,
                    'distribution_type': 'normal',
                    'use_distribution': False,
                    'distribution_cv': 0.15,
                    'historical_mean': 0.0,
                    'historical_std': 0.0,
                    'historical_trend': 'stable',
                    'is_configurable': True
                }
                result['items_created'].append(element_key)
            
            # Update with trend config
            method = config.get('method', 'TREND')
            if method == 'TREND':
                trend_type = config.get('trend_function_type', 'linear')
                trend_params = config.get('trend_parameters', {})
                
                line_items[element_key]['trend_type'] = trend_type.lower()
                
                # Extract growth rate from trend parameters
                if 'slope' in trend_params:
                    line_items[element_key]['trend_growth_rate'] = trend_params['slope'] * 100
                elif 'growth_rate' in trend_params:
                    line_items[element_key]['trend_growth_rate'] = trend_params['growth_rate'] * 100
            
            result['items_migrated'] += 1
        
        # =======================================================================
        # Migrate from ai_assumptions (distribution parameters)
        # =======================================================================
        ai_data = assumptions.get('ai_assumptions', {})
        if isinstance(ai_data, dict) and 'assumptions' in ai_data:
            for key, assumption in ai_data['assumptions'].items():
                if not assumption:
                    continue
                
                # Create line item if doesn't exist
                if key not in line_items:
                    line_items[key] = {
                        'line_item_name': assumption.get('display_name', key.replace('_', ' ').title()),
                        'category': assumption.get('category', 'Other'),
                        'statement_type': 'income_statement',
                        'trend_type': 'linear',
                        'trend_growth_rate': 0.0,
                        'distribution_type': 'normal',
                        'use_distribution': False,
                        'distribution_cv': 0.15,
                        'historical_mean': 0.0,
                        'historical_std': 0.0,
                        'historical_trend': 'stable',
                        'is_configurable': True
                    }
                    result['items_created'].append(key)
                
                # Update with distribution config
                dist_params = assumption.get('distribution_params', {})
                if dist_params:
                    line_items[key]['distribution_type'] = dist_params.get('distribution_type', 'normal')
                    
                    mean = dist_params.get('mean', 0)
                    std = dist_params.get('std', 0)
                    if mean != 0 and std != 0:
                        line_items[key]['distribution_cv'] = std / abs(mean)
                    
                    line_items[key]['historical_mean'] = mean
                    line_items[key]['historical_std'] = std
                    line_items[key]['use_distribution'] = assumption.get('use_distribution', False)
                
                result['items_migrated'] += 1
        
        # =======================================================================
        # Save if not dry run
        # =======================================================================
        if not dry_run:
            unified_config['line_items'] = line_items
            unified_config['last_updated'] = datetime.now().isoformat()
            unified_config['migrated_from_legacy'] = True
            
            assumptions['unified_line_item_config'] = unified_config
            
            # Optionally archive legacy data (don't delete, just mark as migrated)
            if 'forecast_configs' in assumptions:
                assumptions['_legacy_forecast_configs'] = assumptions.pop('forecast_configs')
            if 'ai_assumptions' in assumptions:
                # Keep ai_assumptions as it may have other data
                assumptions['_legacy_ai_assumptions'] = assumptions['ai_assumptions']
            
            success = db.update_assumptions(scenario_id, user_id, assumptions)
            result['success'] = success
            
            if success:
                # Clear caches
                cache_keys = [f'assumptions_{scenario_id}', f'ai_assumptions_{scenario_id}', f'unified_config_{scenario_id}']
                for key in cache_keys:
                    if key in st.session_state:
                        del st.session_state[key]
        else:
            result['success'] = True  # Dry run successful
        
    except Exception as e:
        result['errors'].append(str(e))
    
    return result


def _infer_category(key: str) -> str:
    """Infer category from key name."""
    key_lower = key.lower()
    
    if 'revenue' in key_lower or 'sales' in key_lower:
        return 'Revenue'
    elif 'cogs' in key_lower or 'cost of' in key_lower or 'direct' in key_lower:
        return 'COGS'
    elif 'personnel' in key_lower or 'salary' in key_lower or 'wage' in key_lower:
        return 'Operating Expenses'
    elif 'opex' in key_lower or 'expense' in key_lower or 'overhead' in key_lower:
        return 'Operating Expenses'
    else:
        return 'Other'


def render_migration_ui(db, scenario_id: str, user_id: str):
    """Render the migration utility UI."""
    st.markdown("### 🔄 Assumption Storage Migration")
    st.caption("Consolidate legacy assumption storage into unified line-item configuration")
    
    # Analyze current state
    analysis = analyze_assumption_storage(db, scenario_id, user_id)
    
    # Display current state
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Forecast Configs (Legacy)",
            analysis['forecast_configs_count'],
            delta="Migrate" if analysis['has_forecast_configs'] else None,
            delta_color="off"
        )
    
    with col2:
        st.metric(
            "AI Assumptions (Legacy)",
            analysis['ai_assumptions_count'],
            delta="Migrate" if analysis['has_ai_assumptions'] else None,
            delta_color="off"
        )
    
    with col3:
        st.metric(
            "Unified Config Items",
            analysis['unified_items_count'],
            delta="Active" if analysis['has_unified_config'] else None,
            delta_color="normal" if analysis['has_unified_config'] else "off"
        )
    
    # Status
    if analysis.get('error'):
        st.error(f"Error analyzing storage: {analysis['error']}")
        return
    
    if analysis['needs_migration']:
        st.warning("⚠️ Legacy data found that should be migrated to unified configuration.")
    elif analysis['conflicts']:
        st.info("ℹ️ Multiple storage locations have data. Unified config is active.")
        for conflict in analysis['conflicts']:
            st.caption(conflict)
    else:
        st.success("✅ Assumption storage is clean. Using unified configuration.")
    
    st.markdown("---")
    
    # Migration actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Dry Run Migration", use_container_width=True):
            with st.spinner("Analyzing..."):
                result = migrate_to_unified_config(db, scenario_id, user_id, dry_run=True)
            
            if result['success']:
                st.success(f"✅ Dry run complete: {result['items_migrated']} items would be migrated")
                if result['items_created']:
                    with st.expander("Items to create"):
                        for item in result['items_created']:
                            st.write(f"- {item}")
            else:
                st.error(f"Dry run failed: {result['errors']}")
    
    with col2:
        if st.button("🚀 Run Migration", type="primary", use_container_width=True):
            with st.spinner("Migrating..."):
                result = migrate_to_unified_config(db, scenario_id, user_id, dry_run=False)
            
            if result['success']:
                st.success(f"✅ Migration complete: {result['items_migrated']} items migrated")
                st.info("Legacy data has been archived. Unified config is now active.")
                st.rerun()
            else:
                st.error(f"Migration failed: {result['errors']}")
