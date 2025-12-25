"""
Unified Assumptions Configuration
=================================
Single source of truth for line-item level configuration of trends and distributions.

Architecture:
1. Configure at LINE ITEM level only (Personnel, Facilities, Revenue items, etc.)
2. Aggregates (Total OPEX, Total Revenue) are CALCULATED from line items
3. Calculated elements (GP, EBIT, Tax) are READ-ONLY and derived
4. Batch editing via st.data_editor for efficiency

Date: December 20, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LineItemConfig:
    """Configuration for a single line item."""
    line_item_name: str
    category: str  # 'Revenue', 'COGS', 'Operating Expenses', 'Working Capital', 'Fixed Assets'
    statement_type: str  # 'income_statement', 'balance_sheet', 'cash_flow'
    
    # NEW: Sub-group for finer filtering (e.g., 'Personnel', 'Admin', 'Facilities')
    sub_group: str = ''
    
    # Trend configuration
    trend_type: str = 'linear'  # linear, exponential, logarithmic, flat, manual
    trend_growth_rate: float = 0.0  # Annual growth rate (%)
    
    # NEW: Manual trend overrides (for adjusting poor historical data)
    use_manual_trend: bool = False
    manual_intercept: float = 0.0  # Starting value for manual trend
    manual_slope: float = 0.0  # Monthly change for manual trend
    
    # Distribution configuration (for Monte Carlo)
    distribution_type: str = 'normal'  # normal, lognormal, triangular, uniform
    use_distribution: bool = False
    distribution_cv: float = 0.15  # Coefficient of variation (std/mean)
    
    # NEW: Correlation with revenue (for batch correlation setting)
    correlate_with_revenue: bool = False
    revenue_correlation: float = 0.0  # -1 to 1
    
    # NEW: Balance sheet / Working capital specific
    is_working_capital: bool = False
    days_metric: float = 0.0  # Stock days, receivable days, payable days
    days_metric_type: str = ''  # 'stock_days', 'receivable_days', 'payable_days', 'asset_life'
    
    # Historical stats (for reference)
    historical_mean: float = 0.0
    historical_std: float = 0.0
    historical_trend: str = 'stable'
    
    # Metadata
    is_configurable: bool = True  # False for calculated items
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LineItemConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class UnifiedAssumptionsConfig:
    """Complete configuration for all line items."""
    scenario_id: str
    line_items: Dict[str, LineItemConfig] = field(default_factory=dict)
    
    # Forecast method toggle
    # - pipeline: installed base (fleet) + prospects
    # - trend: line-item trends drive the full forecast
    # - hybrid: trend-driven baseline + pipeline layered on top
    forecast_method: str = 'pipeline'  # 'pipeline' | 'hybrid' | 'trend'
    
    # Metadata
    last_updated: str = ''
    version: str = '1.0'
    
    def to_dict(self) -> dict:
        return {
            'scenario_id': self.scenario_id,
            'line_items': {k: v.to_dict() for k, v in self.line_items.items()},
            'forecast_method': self.forecast_method,
            'last_updated': self.last_updated,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UnifiedAssumptionsConfig':
        config = cls(scenario_id=data.get('scenario_id', ''))
        config.forecast_method = data.get('forecast_method', 'pipeline')
        config.last_updated = data.get('last_updated', '')
        config.version = data.get('version', '1.0')
        
        line_items_data = data.get('line_items', {})
        for key, item_data in line_items_data.items():
            config.line_items[key] = LineItemConfig.from_dict(item_data)
        
        return config


# =============================================================================
# CALCULATED ELEMENTS (Read-Only)
# =============================================================================

CALCULATED_ELEMENTS = {
    'gross_profit': 'Revenue - COGS',
    'total_gross_profit': 'Revenue - COGS',
    'ebit': 'Gross Profit - OPEX',
    'operating_profit': 'Gross Profit - OPEX',
    'ebitda': 'EBIT + Depreciation',
    'ebt': 'EBIT - Interest',
    'profit_before_tax': 'EBIT - Interest',
    'net_income': 'EBT - Tax',
    'net_profit': 'EBT - Tax',
}

CALCULATED_DISPLAY_PATTERNS = [
    'gross profit', 'ebit', 'ebitda', 'operating profit',
    'net income', 'net profit', 'profit before tax', 'profit after tax',
]


def is_calculated_element(name: str) -> bool:
    """Check if a line item is calculated (not directly configurable)."""
    if not name:
        return False
    name_lower = name.lower()
    
    # Check direct matches
    if name_lower.replace(' ', '_') in CALCULATED_ELEMENTS:
        return True
    
    # Check patterns
    for pattern in CALCULATED_DISPLAY_PATTERNS:
        if pattern in name_lower:
            return True
    
    return False


# =============================================================================
# LOAD / SAVE FUNCTIONS
# =============================================================================

def load_unified_config(db, scenario_id: str, user_id: str) -> UnifiedAssumptionsConfig:
    """Load unified configuration from database."""
    config = UnifiedAssumptionsConfig(scenario_id=scenario_id)
    
    try:
        assumptions = db.get_scenario_assumptions(scenario_id, user_id)
        if assumptions and 'unified_line_item_config' in assumptions:
            config = UnifiedAssumptionsConfig.from_dict(assumptions['unified_line_item_config'])
            config.scenario_id = scenario_id
    except Exception as e:
        st.warning(f"Could not load saved configuration: {e}")
    
    return config


def save_unified_config(db, scenario_id: str, user_id: str, config: UnifiedAssumptionsConfig) -> bool:
    """Save unified configuration to database."""
    try:
        config.last_updated = datetime.now().isoformat()
        
        # Load existing assumptions
        assumptions = db.get_scenario_assumptions(scenario_id, user_id) or {}
        
        # Update with unified config
        assumptions['unified_line_item_config'] = config.to_dict()
        
        # Also set the forecast method toggle at top level for easy access
        assumptions['forecast_method'] = config.forecast_method
        assumptions['use_trend_forecast'] = (config.forecast_method in ('trend', 'hybrid'))
        
        # Mark as saved in the assumptions
        assumptions['assumptions_saved'] = True
        assumptions['last_saved'] = config.last_updated
        
        # Save
        success = db.update_assumptions(scenario_id, user_id, assumptions)
        
        if success:
            # Update the assumptions_set in session state to reflect saved status
            if 'ai_assumptions_set' in st.session_state:
                st.session_state.ai_assumptions_set.assumptions_saved = True
            
            # Clear caches
            cache_keys = [f'assumptions_{scenario_id}', f'ai_assumptions_{scenario_id}']
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]
        
        return success
    except Exception as e:
        st.error(f"Error saving configuration: {e}")
        return False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _infer_sub_group(category: str, line_item_name: str) -> str:
    """Infer sub-group from category and line item name."""
    name_lower = line_item_name.lower()
    cat_lower = category.lower()
    
    # Personnel-related
    if any(kw in name_lower for kw in ['salary', 'salaries', 'wage', 'personnel', 'staff', 'employee', 'bonus', 'benefits']):
        return 'Personnel'
    
    # Admin-related
    if any(kw in name_lower for kw in ['admin', 'office', 'legal', 'audit', 'professional', 'consulting']):
        return 'Admin'
    
    # Facilities-related
    if any(kw in name_lower for kw in ['rent', 'lease', 'property', 'utilities', 'electric', 'water', 'maintenance', 'facility', 'building']):
        return 'Facilities'
    
    # Sales & Marketing
    if any(kw in name_lower for kw in ['sales', 'marketing', 'advertising', 'promotion', 'commission']):
        return 'Sales & Marketing'
    
    # IT & Technology
    if any(kw in name_lower for kw in ['software', 'hardware', 'it ', 'technology', 'computer', 'license', 'subscription']):
        return 'IT & Technology'
    
    # Travel & Entertainment
    if any(kw in name_lower for kw in ['travel', 'entertainment', 'meal', 'accommodation', 'transport']):
        return 'Travel & Entertainment'
    
    # Depreciation & Amortization
    if any(kw in name_lower for kw in ['depreciation', 'amortization', 'amortisation']):
        return 'Depreciation'
    
    # Insurance
    if 'insurance' in name_lower:
        return 'Insurance'
    
    # Revenue types
    if 'revenue' in cat_lower or 'sales' in cat_lower or 'income' in cat_lower:
        if 'service' in name_lower:
            return 'Service Revenue'
        elif 'product' in name_lower or 'goods' in name_lower:
            return 'Product Revenue'
        elif 'parts' in name_lower or 'consumable' in name_lower:
            return 'Parts Revenue'
        else:
            return 'Other Revenue'
    
    # COGS types
    if 'cogs' in cat_lower or 'cost of' in cat_lower:
        if 'material' in name_lower or 'raw' in name_lower:
            return 'Materials'
        elif 'labour' in name_lower or 'labor' in name_lower:
            return 'Direct Labour'
        else:
            return 'Direct Costs'
    
    # Default based on category
    if 'operating' in cat_lower or 'opex' in cat_lower or 'expense' in cat_lower:
        return 'Other Operating'
    
    return 'Other'


# =============================================================================
# INITIALIZE FROM HISTORICAL DATA
# =============================================================================

def initialize_line_items_from_historical(db, scenario_id: str, user_id: str) -> Dict[str, LineItemConfig]:
    """
    Initialize line item configurations from historical data.
    
    Loads line items from:
    1. historical_income_statement_line_items
    2. historical_balance_sheet_line_items  
    3. historical_cashflow_line_items
    """
    line_items = {}
    
    # Table mapping
    tables = {
        'income_statement': 'historical_income_statement_line_items',
        'balance_sheet': 'historical_balance_sheet_line_items',
        'cash_flow': 'historical_cashflow_line_items'
    }
    
    for statement_type, table_name in tables.items():
        try:
            if hasattr(db, 'client'):
                response = db.client.table(table_name).select(
                    'line_item_name, category, amount'
                ).eq('scenario_id', scenario_id).execute()
                
                if response.data:
                    df = pd.DataFrame(response.data)
                    
                    # Group by line item and calculate stats
                    for line_item_name, group in df.groupby('line_item_name'):
                        if is_calculated_element(line_item_name):
                            continue
                        
                        key = f"{statement_type}_{line_item_name}".replace(' ', '_').lower()
                        category = group['category'].iloc[0] if 'category' in group.columns else 'Other'
                        
                        amounts = group['amount'].dropna().values
                        hist_mean = float(np.mean(amounts)) if len(amounts) > 0 else 0.0
                        hist_std = float(np.std(amounts)) if len(amounts) > 1 else 0.0
                        
                        # Determine trend from data
                        if len(amounts) >= 2:
                            growth = (amounts[-1] - amounts[0]) / max(abs(amounts[0]), 1)
                            if growth > 0.1:
                                hist_trend = 'growing'
                            elif growth < -0.1:
                                hist_trend = 'declining'
                            else:
                                hist_trend = 'stable'
                        else:
                            hist_trend = 'stable'
                        
                        # Infer sub-group from category
                        sub_group = _infer_sub_group(str(category), str(line_item_name))
                        
                        # Calculate a reasonable default slope (based on historical trend)
                        if len(amounts) >= 2:
                            # Monthly slope based on historical data
                            n_periods = len(amounts)
                            default_slope = (amounts[-1] - amounts[0]) / max(n_periods, 1) / 12  # Annualized to monthly
                        else:
                            default_slope = 0.0
                        
                        line_items[key] = LineItemConfig(
                            line_item_name=str(line_item_name),
                            category=str(category),
                            sub_group=sub_group,
                            statement_type=statement_type,
                            historical_mean=hist_mean,
                            historical_std=hist_std,
                            historical_trend=hist_trend,
                            trend_type='linear',
                            distribution_type='normal',
                            distribution_cv=0.15 if hist_std > 0 else 0.0,
                            use_distribution=False,
                            is_configurable=True,
                            # Set manual intercept to historical mean by default
                            manual_intercept=hist_mean,
                            manual_slope=default_slope,
                            use_manual_trend=False
                        )
        except Exception as e:
            pass  # Continue with other tables
    
    return line_items


# =============================================================================
# RENDER UI
# =============================================================================

def render_unified_config_ui(db, scenario_id: str, user_id: str):
    """
    Render the unified line-item configuration UI.
    
    Features:
    - Forecast method toggle (Pipeline vs Trend)
    - Batch editing of all line items in a table
    - Single save button
    """
    st.markdown("## 📊 Unified Assumptions Configuration")
    
    # Load or initialize config
    config_key = f'unified_config_{scenario_id}'
    if config_key not in st.session_state:
        st.session_state[config_key] = load_unified_config(db, scenario_id, user_id)
    
    config: UnifiedAssumptionsConfig = st.session_state[config_key]
    
    # ==========================================================================
    # FORECAST METHOD TOGGLE
    # ==========================================================================
    st.markdown("### 🔀 Forecast Method")
    
    col1, col2 = st.columns(2)
    with col1:
        method = st.radio(
            "Select Forecast Method",
            options=['pipeline', 'hybrid', 'trend'],
            format_func=lambda x: (
                '📈 Pipeline-Based (Fleet + Prospects)' if x == 'pipeline'
                else ('🧩 Hybrid (Trend Baseline + Pipeline Overlay)' if x == 'hybrid'
                      else '📊 Trend-Based (Line-Item Trends)')
            ),
            index=0 if config.forecast_method == 'pipeline' else (1 if config.forecast_method == 'hybrid' else 2),
            key=f'forecast_method_{scenario_id}',
            help="""
**Pipeline-Based:** Uses installed fleet revenue + prospect pipeline.
Best when you have good fleet data and sales pipeline.

**Hybrid:** Uses Trend/Line-Item trends to build the baseline forecast, then adds Prospect pipeline on top.
Best when you want historical trend shape **and** explicit pipeline opportunities.

**Trend-Based:** Uses line-item trends to project future values.
Best when you want the forecast fully driven by configured trends.
            """
        )
        config.forecast_method = method
    
    with col2:
        if method == 'pipeline':
            st.info("""
**Pipeline-Based Forecast:**
- Revenue = Fleet Revenue + Prospect Pipeline
- COGS = Margin-based calculation
- OPEX = From expense assumptions
            """)
        elif method == 'hybrid':
            st.info("""
**Hybrid Forecast:**
- Baseline = Trend/Line-item forecast
- Plus = Prospect pipeline layered on top
- COGS = Margin-based on total revenue (with pipeline impact)
            """)
        else:
            st.info("""
**Trend-Based Forecast:**
- Each line item follows its configured trend
- Aggregates calculated from line items
- Monte Carlo uses configured distributions
            """)
    
    st.markdown("---")
    
    # ==========================================================================
    # SCENARIO PLANNER (Natural Language)
    # ==========================================================================
    with st.expander("🎯 Scenario Planner (Natural Language)", expanded=False):
        try:
            from components.scenario_planner import render_scenario_planner
            render_scenario_planner(db, scenario_id, user_id, config)
        except ImportError as e:
            st.error(f"Could not load scenario planner: {e}")
        except Exception as e:
            st.error(f"Error in scenario planner: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # ==========================================================================
    # LINE ITEM CONFIGURATION TABLE
    # ==========================================================================
    st.markdown("### 📋 Configure Line Items")
    st.caption("Edit trends and distributions for each line item. Changes are saved when you click 'Save All'.")
    
    # Initialize line items if empty
    if not config.line_items:
        with st.spinner("Loading line items from historical data..."):
            config.line_items = initialize_line_items_from_historical(db, scenario_id, user_id)
        
        if not config.line_items:
            st.warning("No historical line items found. Please import historical data first.")
            st.info("Go to **Setup → Historics** to import Income Statement, Balance Sheet, and Cash Flow data.")
            return
    
    # Convert to DataFrame for editing
    items_data = []
    for key, item in config.line_items.items():
        if not item.is_configurable:
            continue
        items_data.append({
            'key': key,
            'Line Item': item.line_item_name,
            'Category': item.category,
            'Sub-Group': getattr(item, 'sub_group', '') or '',
            'Type': item.statement_type.replace('_', ' ').title(),
            'Hist Mean': item.historical_mean,
            'Trend Type': item.trend_type,
            'Growth %': item.trend_growth_rate,
            'Manual': getattr(item, 'use_manual_trend', False),
            'Intercept': getattr(item, 'manual_intercept', 0.0),
            'Slope': getattr(item, 'manual_slope', 0.0),
            'Use MC': item.use_distribution,
            'Distribution': item.distribution_type,
            'CV %': item.distribution_cv * 100,
            'Rev Corr': getattr(item, 'correlate_with_revenue', False),
            'Corr %': getattr(item, 'revenue_correlation', 0.0) * 100,
        })
    
    if not items_data:
        st.warning("No configurable line items found.")
        return
    
    df = pd.DataFrame(items_data)
    
    # Filters: Category AND Sub-Group
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        categories = ['All'] + sorted(df['Category'].unique().tolist())
        selected_cat = st.selectbox("Filter by Category", categories, key='line_item_cat_filter')
    
    with col_f2:
        # Get unique sub-groups for selected category
        if selected_cat != 'All':
            sub_groups_available = df[df['Category'] == selected_cat]['Sub-Group'].unique().tolist()
        else:
            sub_groups_available = df['Sub-Group'].unique().tolist()
        sub_groups = ['All'] + sorted([s for s in sub_groups_available if s])
        selected_sub = st.selectbox("Filter by Sub-Group", sub_groups, key='line_item_sub_filter')
    
    with col_f3:
        statement_types = ['All'] + sorted(df['Type'].unique().tolist())
        selected_type = st.selectbox("Filter by Statement", statement_types, key='line_item_type_filter')
    
    # Apply filters
    df_filtered = df.copy()
    if selected_cat != 'All':
        df_filtered = df_filtered[df_filtered['Category'] == selected_cat]
    if selected_sub != 'All':
        df_filtered = df_filtered[df_filtered['Sub-Group'] == selected_sub]
    if selected_type != 'All':
        df_filtered = df_filtered[df_filtered['Type'] == selected_type]
    
    # Show count
    st.caption(f"Showing {len(df_filtered)} of {len(df)} line items")
    
    # Editable table
    edited_df = st.data_editor(
        df_filtered,
        column_config={
            'key': None,  # Hide key column
            'Line Item': st.column_config.TextColumn('Line Item', disabled=True, width='medium'),
            'Category': st.column_config.TextColumn('Category', disabled=True, width='small'),
            'Sub-Group': st.column_config.TextColumn('Sub-Group', width='small'),
            'Type': st.column_config.TextColumn('Statement', disabled=True, width='small'),
            'Hist Mean': st.column_config.NumberColumn('Hist Mean', disabled=True, format='R %.0f'),
            'Trend Type': st.column_config.SelectboxColumn(
                'Trend',
                options=['flat', 'linear', 'exponential', 'logarithmic', 'manual'],
                width='small'
            ),
            'Growth %': st.column_config.NumberColumn(
                'Growth %/yr',
                min_value=-50.0,
                max_value=100.0,
                step=1.0,
                format='%.1f%%'
            ),
            'Manual': st.column_config.CheckboxColumn('Manual', width='small', help='Use manual intercept/slope'),
            'Intercept': st.column_config.NumberColumn('Intercept', format='R %.0f', help='Starting value for manual trend'),
            'Slope': st.column_config.NumberColumn('Slope/mo', format='R %.0f', help='Monthly change for manual trend'),
            'Use MC': st.column_config.CheckboxColumn('MC', width='small'),
            'Distribution': st.column_config.SelectboxColumn(
                'Distribution',
                options=['normal', 'lognormal', 'triangular', 'uniform'],
                width='small'
            ),
            'CV %': st.column_config.NumberColumn(
                'CV %',
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                format='%.0f%%',
                help='Coefficient of Variation (Std Dev / Mean)'
            ),
            'Rev Corr': st.column_config.CheckboxColumn('Rev Corr', width='small', help='Correlate with Revenue'),
            'Corr %': st.column_config.NumberColumn(
                'Corr %',
                min_value=-100.0,
                max_value=100.0,
                step=5.0,
                format='%.0f%%',
                help='Correlation with Revenue (-100% to 100%)'
            ),
        },
        hide_index=True,
        use_container_width=True,
        key=f'line_items_editor_{scenario_id}'
    )
    
    # ==========================================================================
    # BATCH ACTIONS FOR FILTERED ITEMS
    # ==========================================================================
    st.markdown("---")
    
    # Get the keys of filtered items
    filtered_keys = edited_df['key'].tolist() if not edited_df.empty else []
    filter_desc = f"**{len(filtered_keys)} items**"
    if selected_cat != 'All':
        filter_desc += f" in {selected_cat}"
    if selected_sub != 'All':
        filter_desc += f" / {selected_sub}"
    
    st.markdown(f"#### 🔧 Batch Actions for {filter_desc}")
    
    # Row 1: Basic batch actions - MC controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Enable MC for All", use_container_width=True, key='batch_mc_on'):
            for key in filtered_keys:
                if key in config.line_items:
                    config.line_items[key].use_distribution = True
            st.success(f"Enabled MC for {len(filtered_keys)} items")
            st.rerun()
    
    with col2:
        if st.button("❌ Disable MC for All", use_container_width=True, key='batch_mc_off'):
            for key in filtered_keys:
                if key in config.line_items:
                    config.line_items[key].use_distribution = False
            st.success(f"Disabled MC for {len(filtered_keys)} items")
            st.rerun()
    
    # Row 2: Trend and Distribution with Apply buttons (NOT auto-apply)
    col3, col4, col5, col6 = st.columns([2, 1, 2, 1])
    
    with col3:
        batch_trend = st.selectbox(
            "Trend Type", 
            options=['flat', 'linear', 'exponential', 'manual'], 
            key='batch_trend_select',
            index=1  # Default to linear
        )
    
    with col4:
        if st.button("Apply", key='apply_batch_trend', use_container_width=True):
            for key in filtered_keys:
                if key in config.line_items:
                    config.line_items[key].trend_type = batch_trend
                    if batch_trend == 'manual':
                        config.line_items[key].use_manual_trend = True
            st.success(f"Set trend to '{batch_trend}' for {len(filtered_keys)} items")
            st.rerun()
    
    with col5:
        batch_dist = st.selectbox(
            "Distribution", 
            options=['normal', 'lognormal', 'triangular', 'uniform'], 
            key='batch_dist_select',
            index=0  # Default to normal
        )
    
    with col6:
        if st.button("Apply", key='apply_batch_dist', use_container_width=True):
            for key in filtered_keys:
                if key in config.line_items:
                    config.line_items[key].distribution_type = batch_dist
            st.success(f"Set distribution to '{batch_dist}' for {len(filtered_keys)} items")
            st.rerun()
    
    # ==========================================================================
    # MANUAL TREND BATCH CONFIGURATION
    # ==========================================================================
    st.markdown("---")
    st.markdown("#### 📐 Manual Trend Batch Configuration")
    st.caption(f"Apply manual intercept & slope to {filter_desc}")
    
    # Calculate average historical mean for filtered items
    filtered_hist_means = [config.line_items[k].historical_mean for k in filtered_keys if k in config.line_items]
    avg_hist_mean = np.mean(filtered_hist_means) if filtered_hist_means else 0.0
    
    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    
    with col_m1:
        st.markdown("**Intercept (Starting Value)**")
        intercept_mode = st.radio(
            "Intercept Mode",
            options=['hist_mean', 'custom'],
            format_func=lambda x: 'Use Historical Mean' if x == 'hist_mean' else 'Custom Value',
            horizontal=True,
            key='intercept_mode'
        )
        
        if intercept_mode == 'hist_mean':
            batch_intercept = avg_hist_mean
            st.info(f"Avg Historical Mean: **R {avg_hist_mean:,.0f}**")
        else:
            batch_intercept = st.number_input(
                "Custom Intercept (R)",
                value=avg_hist_mean,
                step=10000.0,
                format="%.0f",
                key='batch_intercept_custom'
            )
    
    with col_m2:
        st.markdown("**Slope (Monthly Change)**")
        slope_mode = st.radio(
            "Slope Mode",
            options=['absolute', 'percentage'],
            format_func=lambda x: 'Absolute (R/month)' if x == 'absolute' else 'Percentage (%/month)',
            horizontal=True,
            key='slope_mode'
        )
        
        if slope_mode == 'absolute':
            batch_slope = st.number_input(
                "Slope (R per month)",
                value=0.0,
                step=1000.0,
                format="%.0f",
                key='batch_slope_abs',
                help="Positive = growth, Negative = decline"
            )
        else:
            slope_pct = st.number_input(
                "Slope (% per month)",
                value=0.0,
                min_value=-10.0,
                max_value=20.0,
                step=0.5,
                format="%.1f",
                key='batch_slope_pct',
                help="Positive = growth, Negative = decline"
            )
            # Convert percentage to absolute based on intercept
            batch_slope = batch_intercept * (slope_pct / 100)
            st.caption(f"= R {batch_slope:,.0f} per month")
    
    with col_m3:
        st.markdown("**Preview (12 months)**")
        
        # Generate preview data
        preview_months = 12
        preview_t = np.arange(0, preview_months)
        preview_values = batch_intercept + batch_slope * preview_t
        
        # Calculate end value and total change
        end_value = preview_values[-1] if len(preview_values) > 0 else batch_intercept
        total_change = end_value - batch_intercept
        pct_change = (total_change / batch_intercept * 100) if batch_intercept != 0 else 0
        
        # Create mini chart using plotly
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, preview_months + 1)),
                y=preview_values.tolist(),
                mode='lines',
                line=dict(color='#2196F3' if batch_slope >= 0 else '#F44336', width=2),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.1)' if batch_slope >= 0 else 'rgba(244, 67, 54, 0.1)'
            ))
            
            fig.update_layout(
                height=150,
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis=dict(title='Month', showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#eee', tickformat=',.0f'),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            # Fallback: text-based preview
            st.code(f"Start: R {batch_intercept:,.0f}\nEnd:   R {end_value:,.0f}")
        
        # Show change summary
        if total_change >= 0:
            st.success(f"📈 +R {total_change:,.0f} ({pct_change:+.1f}%) over 12mo")
        else:
            st.error(f"📉 R {total_change:,.0f} ({pct_change:+.1f}%) over 12mo")
    
    # Apply button for manual trend
    col_apply1, col_apply2 = st.columns([3, 1])
    
    with col_apply2:
        if st.button("📐 Apply Manual Trend", type='primary', use_container_width=True, key='apply_manual_batch'):
            count = 0
            for key in filtered_keys:
                if key in config.line_items:
                    item = config.line_items[key]
                    item.use_manual_trend = True
                    item.trend_type = 'manual'
                    
                    # Set intercept (use item's hist mean if mode is hist_mean)
                    if intercept_mode == 'hist_mean':
                        item.manual_intercept = item.historical_mean
                    else:
                        item.manual_intercept = batch_intercept
                    
                    # Set slope (recalculate if percentage mode, based on item's intercept)
                    if slope_mode == 'percentage':
                        item.manual_slope = item.manual_intercept * (slope_pct / 100)
                    else:
                        item.manual_slope = batch_slope
                    
                    count += 1
            
            st.success(f"✅ Applied manual trend to {count} items")
            st.rerun()
    
    with col_apply1:
        st.caption(f"This will set Manual trend for all {len(filtered_keys)} filtered items with the configured intercept and slope.")
    
    # ==========================================================================
    # GROUPED CORRELATION (Batch set correlations by category/sub-group)
    # ==========================================================================
    st.markdown("---")
    st.markdown("#### 🔗 Grouped Correlations with Revenue")
    st.caption("Set correlation with revenue for entire categories or sub-groups at once")
    
    col_gc1, col_gc2, col_gc3, col_gc4 = st.columns([2, 2, 1, 1])
    
    with col_gc1:
        group_corr_type = st.selectbox(
            "Apply to",
            options=['Category', 'Sub-Group'],
            key='group_corr_type'
        )
    
    with col_gc2:
        if group_corr_type == 'Category':
            available_groups = sorted(set(item.category for item in config.line_items.values()))
        else:
            available_groups = sorted(set(getattr(item, 'sub_group', '') or 'Ungrouped' for item in config.line_items.values()))
        
        selected_group = st.selectbox(
            f"Select {group_corr_type}",
            options=available_groups,
            key='group_corr_selection'
        )
    
    with col_gc3:
        group_corr_value = st.slider(
            "Correlation",
            min_value=-1.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key='group_corr_value'
        )
    
    with col_gc4:
        if st.button("Apply", key='apply_group_corr', use_container_width=True):
            count = 0
            for key, item in config.line_items.items():
                if group_corr_type == 'Category':
                    if item.category == selected_group:
                        item.correlate_with_revenue = True
                        item.revenue_correlation = group_corr_value
                        count += 1
                else:
                    if (getattr(item, 'sub_group', '') or 'Ungrouped') == selected_group:
                        item.correlate_with_revenue = True
                        item.revenue_correlation = group_corr_value
                        count += 1
            st.success(f"Applied {group_corr_value:.0%} correlation to {count} items in {selected_group}")
            st.rerun()
    
    # ==========================================================================
    # AI-SUGGESTED CORRELATIONS FOR MINING SERVICES
    # ==========================================================================
    st.markdown("---")
    st.markdown("#### 🤖 AI-Suggested Correlations (Mining Services)")
    st.caption("Apply industry-typical correlations for mining services / installed base businesses")
    
    # Define typical correlations for mining services businesses
    MINING_SERVICES_CORRELATIONS = {
        # Sub-group based correlations
        'sub_group': {
            'Personnel': 0.80,          # Grows with business, but with some lag
            'Admin': 0.50,              # Semi-fixed, some growth with scale
            'Facilities': 0.35,         # Mostly fixed, step changes
            'Sales & Marketing': 0.75,  # Tied to sales activity
            'IT & Technology': 0.45,    # Semi-fixed with scale
            'Travel & Entertainment': 0.75,  # Tied to activity
            'Depreciation': 0.55,       # Tied to asset base
            'Insurance': 0.25,          # Mostly fixed
            'Service Revenue': 1.0,     # Is revenue
            'Product Revenue': 1.0,     # Is revenue
            'Parts Revenue': 1.0,       # Is revenue
            'Other Revenue': 1.0,       # Is revenue
            'Materials': 0.92,          # Direct cost, very high correlation
            'Direct Labour': 0.88,      # Direct cost, high correlation
            'Direct Costs': 0.90,       # Direct cost, very high correlation
            'Other Operating': 0.60,    # Mixed
        },
        # Category-based fallbacks
        'category': {
            'Revenue': 1.0,             # Revenue correlates perfectly with itself
            'COGS': 0.92,               # Direct costs highly correlated
            'Cost of Sales': 0.92,
            'Operating Expenses': 0.65, # Mixed fixed and variable
            'OPEX': 0.65,
            'Overhead': 0.60,
            'Working Capital': 0.85,    # Tied to activity
            'Fixed Assets': 0.35,       # Mostly stable, step changes
        }
    }
    
    col_ai1, col_ai2 = st.columns([3, 1])
    
    with col_ai1:
        with st.expander("📋 View AI Correlation Logic", expanded=False):
            st.markdown("""
**Mining Services / Installed Base Business Correlations:**

| Category/Sub-Group | Correlation | Rationale |
|-------------------|-------------|-----------|
| **Direct Costs (COGS)** | 90-95% | Highly variable with volume |
| **Personnel** | 80% | Grows with business, some lag |
| **Sales & Marketing** | 75% | Tied to sales activity |
| **Travel** | 75% | Tied to field activity |
| **Admin** | 50% | Semi-fixed |
| **IT** | 45% | Semi-fixed with scale |
| **Facilities/Rent** | 35% | Mostly fixed, step changes |
| **Insurance** | 25% | Largely fixed |
| **Working Capital** | 85% | Tied to activity levels |
| **Fixed Assets** | 35% | Stable, occasional capex |
            """)
    
    with col_ai2:
        if st.button("🤖 Apply AI Correlations", type='primary', use_container_width=True, key='apply_ai_corr'):
            count = 0
            applied_details = []
            
            for key, item in config.line_items.items():
                sub_group = getattr(item, 'sub_group', '') or ''
                category = item.category or ''
                
                # Try sub-group first, then category
                correlation = None
                match_type = None
                
                # Check sub-group
                for sg_key, sg_corr in MINING_SERVICES_CORRELATIONS['sub_group'].items():
                    if sg_key.lower() in sub_group.lower():
                        correlation = sg_corr
                        match_type = f"Sub-group: {sg_key}"
                        break
                
                # Fallback to category
                if correlation is None:
                    for cat_key, cat_corr in MINING_SERVICES_CORRELATIONS['category'].items():
                        if cat_key.lower() in category.lower():
                            correlation = cat_corr
                            match_type = f"Category: {cat_key}"
                            break
                
                # Apply if found
                if correlation is not None:
                    item.correlate_with_revenue = True
                    item.revenue_correlation = correlation
                    count += 1
                    applied_details.append({
                        'item': item.line_item_name[:30],
                        'corr': correlation,
                        'match': match_type
                    })
            
            st.success(f"✅ Applied AI correlations to {count} line items")
            st.session_state['ai_corr_applied'] = applied_details
            st.rerun()
    
    # Show what was applied
    if 'ai_corr_applied' in st.session_state and st.session_state['ai_corr_applied']:
        with st.expander(f"📊 AI Correlations Applied ({len(st.session_state['ai_corr_applied'])} items)", expanded=False):
            applied_df = pd.DataFrame(st.session_state['ai_corr_applied'])
            applied_df.columns = ['Line Item', 'Correlation', 'Matched On']
            applied_df['Correlation'] = applied_df['Correlation'].apply(lambda x: f"{x:.0%}")
            st.dataframe(applied_df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Quick manual presets (kept for fine-tuning)
    st.caption("**Quick Manual Presets:**")
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        if st.button("📊 All OPEX → 0.7", use_container_width=True, key='preset_overhead'):
            for key, item in config.line_items.items():
                cat_lower = item.category.lower()
                if 'opex' in cat_lower or 'operating' in cat_lower or 'overhead' in cat_lower or 'expense' in cat_lower:
                    item.correlate_with_revenue = True
                    item.revenue_correlation = 0.7
            st.rerun()
    
    with col_p2:
        if st.button("📈 All COGS → 0.95", use_container_width=True, key='preset_cogs'):
            for key, item in config.line_items.items():
                cat_lower = item.category.lower()
                if 'cogs' in cat_lower or 'cost of' in cat_lower or 'direct' in cat_lower:
                    item.correlate_with_revenue = True
                    item.revenue_correlation = 0.95
            st.rerun()
    
    with col_p3:
        if st.button("🏢 Fixed Assets → 0.3", use_container_width=True, key='preset_assets'):
            for key, item in config.line_items.items():
                cat_lower = item.category.lower()
                if 'fixed' in cat_lower or 'asset' in cat_lower or 'depreciation' in cat_lower:
                    item.correlate_with_revenue = True
                    item.revenue_correlation = 0.3
            st.rerun()
    
    # ==========================================================================
    # WORKING CAPITAL & BALANCE SHEET ITEMS
    # ==========================================================================
    st.markdown("---")
    
    with st.expander("📊 Working Capital & Balance Sheet Configuration", expanded=False):
        st.markdown("Configure decision-driven balance sheet items")
        
        # Check for working capital items
        wc_items = {k: v for k, v in config.line_items.items() 
                   if v.statement_type == 'balance_sheet' or getattr(v, 'is_working_capital', False)}
        
        if not wc_items:
            st.info("No balance sheet items found. These will be auto-detected from imported data.")
            
            # Add standard working capital items
            st.markdown("**Add Standard Working Capital Items:**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("➕ Add Stock/Inventory", use_container_width=True):
                    config.line_items['inventory'] = LineItemConfig(
                        line_item_name='Inventory / Stock',
                        category='Working Capital',
                        sub_group='Current Assets',
                        statement_type='balance_sheet',
                        is_working_capital=True,
                        days_metric=45.0,
                        days_metric_type='stock_days',
                        correlate_with_revenue=True,
                        revenue_correlation=0.9
                    )
                    st.rerun()
                
                if st.button("➕ Add Receivables", use_container_width=True):
                    config.line_items['receivables'] = LineItemConfig(
                        line_item_name='Trade Receivables',
                        category='Working Capital',
                        sub_group='Current Assets',
                        statement_type='balance_sheet',
                        is_working_capital=True,
                        days_metric=45.0,
                        days_metric_type='receivable_days',
                        correlate_with_revenue=True,
                        revenue_correlation=0.95
                    )
                    st.rerun()
            
            with col2:
                if st.button("➕ Add Payables", use_container_width=True):
                    config.line_items['payables'] = LineItemConfig(
                        line_item_name='Trade Payables',
                        category='Working Capital',
                        sub_group='Current Liabilities',
                        statement_type='balance_sheet',
                        is_working_capital=True,
                        days_metric=30.0,
                        days_metric_type='payable_days',
                        correlate_with_revenue=True,
                        revenue_correlation=0.9
                    )
                    st.rerun()
                
                if st.button("➕ Add Fixed Assets", use_container_width=True):
                    config.line_items['fixed_assets'] = LineItemConfig(
                        line_item_name='Property, Plant & Equipment',
                        category='Fixed Assets',
                        sub_group='Non-Current Assets',
                        statement_type='balance_sheet',
                        is_working_capital=False,
                        days_metric=10.0,  # Years
                        days_metric_type='asset_life',
                        trend_type='flat',  # Constant by default
                        correlate_with_revenue=False
                    )
                    st.rerun()
        else:
            # Show working capital configuration
            wc_data = []
            for key, item in wc_items.items():
                wc_data.append({
                    'key': key,
                    'Item': item.line_item_name,
                    'Category': item.category,
                    'Days/Years': getattr(item, 'days_metric', 0.0),
                    'Metric Type': getattr(item, 'days_metric_type', ''),
                    'Corr w/Rev': getattr(item, 'correlate_with_revenue', False),
                    'Correlation': getattr(item, 'revenue_correlation', 0.0) * 100,
                    'Use MC': item.use_distribution,
                    'CV %': item.distribution_cv * 100,
                })
            
            wc_df = pd.DataFrame(wc_data)
            
            edited_wc_df = st.data_editor(
                wc_df,
                column_config={
                    'key': None,
                    'Item': st.column_config.TextColumn('Item', disabled=True),
                    'Category': st.column_config.TextColumn('Category', disabled=True),
                    'Days/Years': st.column_config.NumberColumn('Days/Years', min_value=0, max_value=365),
                    'Metric Type': st.column_config.SelectboxColumn(
                        'Metric',
                        options=['stock_days', 'receivable_days', 'payable_days', 'asset_life'],
                    ),
                    'Corr w/Rev': st.column_config.CheckboxColumn('Corr w/Rev'),
                    'Correlation': st.column_config.NumberColumn('Corr %', min_value=-100, max_value=100, format='%.0f%%'),
                    'Use MC': st.column_config.CheckboxColumn('MC'),
                    'CV %': st.column_config.NumberColumn('CV %', min_value=0, max_value=100, format='%.0f%%'),
                },
                hide_index=True,
                use_container_width=True,
                key='wc_editor'
            )
            
            # Apply working capital edits
            for _, row in edited_wc_df.iterrows():
                key = row['key']
                if key in config.line_items:
                    config.line_items[key].days_metric = float(row['Days/Years'])
                    config.line_items[key].days_metric_type = row['Metric Type']
                    config.line_items[key].correlate_with_revenue = bool(row['Corr w/Rev'])
                    config.line_items[key].revenue_correlation = float(row['Correlation']) / 100
                    config.line_items[key].use_distribution = bool(row['Use MC'])
                    config.line_items[key].distribution_cv = float(row['CV %']) / 100
            
            st.caption("**Stock Days:** Inventory value = (Daily COGS × Stock Days)")
            st.caption("**Receivable Days:** Receivables = (Daily Revenue × Receivable Days)")
            st.caption("**Payable Days:** Payables = (Daily COGS × Payable Days)")
    
    # ==========================================================================
    # CORRELATIONS (Phase 3) - Individual pairs
    # ==========================================================================
    st.markdown("---")
    
    with st.expander("🔗 Advanced: Individual Correlation Pairs", expanded=False):
        try:
            from components.correlation_config import render_correlation_config_ui
            
            # Convert line items to simple dict for correlation UI
            line_items_for_corr = {
                k: {'name': v.line_item_name, 'category': v.category}
                for k, v in config.line_items.items()
            }
            
            render_correlation_config_ui(db, scenario_id, user_id, line_items_for_corr)
        except ImportError as e:
            st.info("Correlation configuration module not available.")
        except Exception as e:
            st.error(f"Error loading correlation config: {e}")
    
    # ==========================================================================
    # SAVE BUTTON
    # ==========================================================================
    st.markdown("---")
    
    # Apply edits from table back to config
    for _, row in edited_df.iterrows():
        key = row['key']
        if key in config.line_items:
            # Basic trend settings
            config.line_items[key].trend_type = row['Trend Type']
            config.line_items[key].trend_growth_rate = float(row['Growth %'])
            
            # Sub-group
            config.line_items[key].sub_group = row.get('Sub-Group', '')
            
            # Manual trend overrides
            config.line_items[key].use_manual_trend = bool(row.get('Manual', False))
            config.line_items[key].manual_intercept = float(row.get('Intercept', 0.0))
            config.line_items[key].manual_slope = float(row.get('Slope', 0.0))
            
            # Distribution settings
            config.line_items[key].use_distribution = bool(row['Use MC'])
            config.line_items[key].distribution_type = row['Distribution']
            config.line_items[key].distribution_cv = float(row['CV %']) / 100.0
            
            # Revenue correlation
            config.line_items[key].correlate_with_revenue = bool(row.get('Rev Corr', False))
            config.line_items[key].revenue_correlation = float(row.get('Corr %', 0.0)) / 100.0
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("💾 Save All Configuration", type='primary', use_container_width=True):
            if save_unified_config(db, scenario_id, user_id, config):
                st.success("✅ Configuration saved successfully!")
                st.session_state[config_key] = config
            else:
                st.error("Failed to save configuration")
    
    with col1:
        st.caption(f"Last saved: {config.last_updated or 'Never'}")
    
    # ==========================================================================
    # AGGREGATES PREVIEW (Read-Only)
    # ==========================================================================
    with st.expander("📊 Aggregates Preview (Calculated from Line Items)", expanded=False):
        st.markdown("These totals are **calculated** from the configured line items above.")
        
        # Calculate aggregates
        agg_data = {}
        for key, item in config.line_items.items():
            cat = item.category
            if cat not in agg_data:
                agg_data[cat] = {'count': 0, 'total_mean': 0}
            agg_data[cat]['count'] += 1
            agg_data[cat]['total_mean'] += item.historical_mean
        
        agg_df = pd.DataFrame([
            {'Category': cat, 'Line Items': data['count'], 'Total (Historical)': data['total_mean']}
            for cat, data in agg_data.items()
        ])
        
        if not agg_df.empty:
            st.dataframe(
                agg_df,
                column_config={
                    'Total (Historical)': st.column_config.NumberColumn(format='R %.0f')
                },
                hide_index=True,
                use_container_width=True
            )
        
        st.caption("When running forecast, line items are summed to produce Revenue, COGS, OPEX totals.")


def get_forecast_method(db, scenario_id: str, user_id: str) -> str:
    """Get the configured forecast method for a scenario."""
    try:
        assumptions = db.get_scenario_assumptions(scenario_id, user_id)
        return assumptions.get('forecast_method', 'pipeline')
    except:
        return 'pipeline'
