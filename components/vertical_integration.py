"""
Vertical Integration & Manufacturing Strategy Component - Enhanced v3.2
======================================================================
Date: December 14, 2025
Version: 3.2 - Sprint 16+ (Overhead & Part-Level Integration)

NEW IN V3.2 (Sprint 16+ - Cost Structure Enhancement):
1. ✅ Separate Variable vs Fixed Overhead in income statement
2. ✅ Part-level analysis now syncs to scenario.make_costs and scenario.buy_costs
3. ✅ Choice of overhead calculation: % of direct costs OR fixed per unit
4. ✅ Detailed cost breakdown display (raw materials, labor, variable overhead, QC)
5. ✅ Working Capital separated from Commissioning costs

NEW IN V3.1 (Sprint 16+ - Persistence):
1. ✅ Save/Load manufacturing strategy to database
2. ✅ Auto-load saved strategy when returning to scenario
3. ✅ Save Strategy button with visual feedback
4. ✅ Reset button to clear configuration
5. ✅ Serialization methods (to_dict/from_dict) on IntegrationScenario

NEW IN V3.0 (Sprint 16):
1. ✅ Commissioning Workflow - Start month, duration, dynamic cost table
2. ✅ Working Capital Calculation - Based on raw material days + manufacturing time
3. ✅ AI-Derived Defaults - Pull annual demand and avg price from historical analysis
4. ✅ Manufacturing Impact Timing - Only kicks in after commissioning complete
5. ✅ Forecast Integration API - get_manufacturing_impact_for_forecast()
6. ✅ CAPEX validation - Commissioning costs must equal CAPEX (excl. Working Capital)

ENHANCEMENTS FROM V2.0 (Sprint 13):
1. Historical Data Integration - Auto-populate from historical liner sales
2. Two Analysis Modes Toggle:
   - AVERAGE MODE: Use historical averages + blanket % manufactured
   - PART-LEVEL MODE: Select individual liners with specific costs
3. Import/Export Template - Bulk part configuration via CSV
4. Part Selection with filtering and search
5. Weighted average calculations for mixed scenarios

FEATURES (from v1.0):
- Manufacturing Strategy Toggle (Make/Buy/Hybrid)
- Manufacturing Cost Modeling (raw materials, labor, overhead)
- Capacity Constraint Planning
- CAPEX Requirements Analysis
- Break-even Analysis
- NPV Comparison (Make vs Buy)
- Scenario Comparison View

INSTALLATION:
1. Save as components/vertical_integration.py
2. Import in app_refactored.py
3. Add navigation option
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from dateutil.relativedelta import relativedelta
import io

# =============================================================================
# COLOR CONSTANTS
# =============================================================================
GOLD = "#D4A537"
GOLD_LIGHT = "rgba(212, 165, 55, 0.1)"
GREEN = "#10b981"
RED = "#ef4444"
BLUE = "#3b82f6"
PURPLE = "#8b5cf6"
TEXT_MUTED = "#888888"
TEXT_WHITE = "#FFFFFF"
BORDER_COLOR = "#404040"
DARK_BG = "#1E1E1E"

CHART_COLORS = {
    'primary': '#3b82f6',
    'secondary': '#8b5cf6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'make': '#10b981',  # Green for Make
    'buy': '#3b82f6',   # Blue for Buy
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LinerPartData:
    """Individual liner part data with pricing and cost info."""
    part_number: str = ""
    description: str = ""
    category: str = "Liner"
    historical_avg_selling_price: float = 0.0
    historical_avg_cogs: float = 0.0
    historical_units_sold: int = 0
    historical_revenue: float = 0.0
    gross_margin_pct: float = 0.0
    # Manufacturing decision
    manufacture_pct: float = 0.0  # 0-100% of this part to manufacture
    mfg_cost_pct_of_buy: float = 75.0  # Expected mfg cost as % of buy price


@dataclass
class HistoricalSummary:
    """Summary of historical liner data."""
    total_revenue: float = 0.0
    total_cogs: float = 0.0
    total_units: int = 0
    avg_selling_price: float = 0.0
    avg_cogs_per_unit: float = 0.0
    avg_gross_margin_pct: float = 0.0
    unique_parts: int = 0
    period_months: int = 0


@dataclass
class ManufacturingCosts:
    """Cost structure for in-house manufacturing."""
    # Direct costs per unit
    raw_material_per_unit: float = 0.0        # R per unit - raw materials
    direct_labor_per_unit: float = 0.0        # R per unit - direct labor
    
    # Variable overhead (choose one method)
    variable_overhead_per_unit: float = 0.0   # R per unit - if using fixed amount
    manufacturing_overhead_pct: float = 0.15  # % of direct costs - if using percentage
    use_overhead_pct: bool = True             # True = use %, False = use per unit
    
    # Other variable costs
    quality_control_pct: float = 0.05         # % of production cost
    wastage_pct: float = 0.03                 # % material wastage
    
    @property
    def direct_costs_per_unit(self) -> float:
        """Direct costs = Raw materials (with wastage) + Direct labor."""
        return self.raw_material_per_unit * (1 + self.wastage_pct) + self.direct_labor_per_unit
    
    @property
    def variable_overhead_amount(self) -> float:
        """Variable overhead per unit (calculated based on method chosen)."""
        if self.use_overhead_pct:
            return self.direct_costs_per_unit * self.manufacturing_overhead_pct
        else:
            return self.variable_overhead_per_unit
    
    @property
    def qc_cost_per_unit(self) -> float:
        """Quality control cost per unit."""
        subtotal = self.direct_costs_per_unit + self.variable_overhead_amount
        return subtotal * self.quality_control_pct
    
    @property
    def total_variable_cost_per_unit(self) -> float:
        """Calculate total variable cost per unit (all-in)."""
        return self.direct_costs_per_unit + self.variable_overhead_amount + self.qc_cost_per_unit
    
    @property 
    def cost_breakdown(self) -> Dict[str, float]:
        """Return detailed cost breakdown per unit."""
        return {
            'raw_material': self.raw_material_per_unit * (1 + self.wastage_pct),
            'direct_labor': self.direct_labor_per_unit,
            'variable_overhead': self.variable_overhead_amount,
            'quality_control': self.qc_cost_per_unit,
            'total': self.total_variable_cost_per_unit
        }


@dataclass
class ManufacturingCapacity:
    """Capacity constraints for manufacturing."""
    monthly_capacity_units: int = 1000       # Max units per month
    ramp_up_months: int = 6                  # Months to reach full capacity
    initial_capacity_pct: float = 0.3        # Starting capacity %
    capacity_step_pct: float = 0.15          # Monthly capacity increase
    
    def get_capacity_for_month(self, month_index: int) -> int:
        """Get available capacity for a given month."""
        if month_index < self.ramp_up_months:
            capacity_pct = min(
                self.initial_capacity_pct + (month_index * self.capacity_step_pct),
                1.0
            )
        else:
            capacity_pct = 1.0
        return int(self.monthly_capacity_units * capacity_pct)


@dataclass
class ManufacturingCapex:
    """CAPEX requirements for manufacturing setup."""
    equipment_cost: float = 0.0              # R - Manufacturing equipment
    facility_cost: float = 0.0               # R - Building/modifications
    tooling_cost: float = 0.0                # R - Molds, dies, tooling
    installation_cost: float = 0.0           # R - Installation & commissioning
    working_capital: float = 0.0             # R - Initial inventory & WC (NOT part of commissioning)
    
    # Depreciation
    equipment_life_years: int = 10
    facility_life_years: int = 20
    tooling_life_years: int = 5
    depreciation_years: int = 10  # Default for overall calculation
    
    @property
    def total_capex(self) -> float:
        """Total CAPEX including working capital (for investment analysis)."""
        return (self.equipment_cost + self.facility_cost + 
                self.tooling_cost + self.installation_cost + self.working_capital)
    
    @property
    def total_capex_for_commissioning(self) -> float:
        """CAPEX excluding working capital (for commissioning cost validation).
        
        Working capital is a separate investment that flows into forecast 
        working capital requirements, not into commissioning costs.
        """
        return (self.equipment_cost + self.facility_cost + 
                self.tooling_cost + self.installation_cost)
    
    @property
    def total_ppe(self) -> float:
        """Total Property, Plant & Equipment (for balance sheet)."""
        return self.equipment_cost + self.facility_cost + self.tooling_cost
    
    @property
    def monthly_depreciation(self) -> float:
        equip_dep = self.equipment_cost / (self.equipment_life_years * 12) if self.equipment_life_years > 0 else 0
        facility_dep = self.facility_cost / (self.facility_life_years * 12) if self.facility_life_years > 0 else 0
        tooling_dep = self.tooling_cost / (self.tooling_life_years * 12) if self.tooling_life_years > 0 else 0
        return equip_dep + facility_dep + tooling_dep


# =============================================================================
# NEW IN V3.0: COMMISSIONING & WORKING CAPITAL
# =============================================================================

@dataclass
class CommissioningSchedule:
    """
    NEW IN v3.0: Commissioning schedule for manufacturing setup.
    
    Defines when manufacturing starts, how long commissioning takes,
    and the costs incurred during each month of commissioning.
    """
    start_month: int = 6  # Month index from forecast start (0-based)
    duration_months: int = 4  # How many months to commission
    monthly_costs: List[float] = field(default_factory=list)  # Cost per month during commissioning
    
    @property
    def completion_month(self) -> int:
        """Month when commissioning is complete and manufacturing can begin."""
        return self.start_month + self.duration_months
    
    @property
    def total_commissioning_cost(self) -> float:
        """Total cost of commissioning across all months."""
        return sum(self.monthly_costs) if self.monthly_costs else 0.0
    
    def get_cost_at_month(self, month: int) -> float:
        """Get commissioning cost for a specific month."""
        if month < self.start_month:
            return 0.0
        relative_month = month - self.start_month
        if relative_month < len(self.monthly_costs):
            return self.monthly_costs[relative_month]
        return 0.0
    
    def is_manufacturing_active(self, month: int) -> bool:
        """Check if manufacturing should be active (commissioning complete)."""
        return month >= self.completion_month


@dataclass
class WorkingCapitalParams:
    """
    NEW IN v3.0: Working capital calculation parameters.
    
    WC = (Monthly MFG COGS × Mfg %) × (RM Days + MFG Days) / 30
    Labour is expensed when incurred (make-to-order model).
    """
    raw_material_days: int = 30  # Days of raw material inventory
    manufacturing_days: int = 14  # Days to manufacture (cycle time)
    calculated_wc: float = 0.0   # Last calculated WC value
    manual_override: bool = False  # Whether to use manual value
    manual_wc_value: float = 0.0   # Manual override value
    
    def calculate_working_capital(self, monthly_mfg_cogs: float, mfg_pct: float = 1.0) -> float:
        """
        Calculate working capital requirement.
        
        Args:
            monthly_mfg_cogs: Monthly COGS for manufactured portion
            mfg_pct: Percentage being manufactured (0-1)
        
        Returns:
            Working capital requirement in Rands
        """
        total_days = self.raw_material_days + self.manufacturing_days
        wc_requirement = (monthly_mfg_cogs * mfg_pct) * total_days / 30
        return wc_requirement


@dataclass
class BuyCosts:
    """Cost structure for buying/outsourcing."""
    purchase_price_per_unit: float = 0.0     # R per unit from supplier
    freight_per_unit: float = 0.0            # R per unit freight
    import_duty_pct: float = 0.0             # % import duty
    quality_inspection_pct: float = 0.02     # % inspection costs
    
    @property
    def total_landed_cost_per_unit(self) -> float:
        """Calculate total landed cost per unit."""
        base = self.purchase_price_per_unit + self.freight_per_unit
        duty = base * self.import_duty_pct
        inspection = (base + duty) * self.quality_inspection_pct
        return base + duty + inspection


@dataclass 
class IntegrationScenario:
    """Complete vertical integration scenario."""
    name: str = "Default Scenario"
    strategy: str = "buy"  # 'make', 'buy', 'hybrid'
    hybrid_make_pct: float = 0.5  # % to manufacture if hybrid
    
    # Analysis mode: 'average' or 'part_level'
    analysis_mode: str = "average"
    
    # Product info
    product_name: str = "Crusher Liners"
    annual_demand_units: int = 10000
    selling_price_per_unit: float = 0.0
    
    # Cost structures
    make_costs: ManufacturingCosts = field(default_factory=ManufacturingCosts)
    buy_costs: BuyCosts = field(default_factory=BuyCosts)
    
    # Manufacturing setup
    capacity: ManufacturingCapacity = field(default_factory=ManufacturingCapacity)
    capex: ManufacturingCapex = field(default_factory=ManufacturingCapex)
    
    # NEW IN v3.0: Commissioning schedule
    commissioning: CommissioningSchedule = field(default_factory=CommissioningSchedule)
    
    # NEW IN v3.0: Working capital parameters
    wc_params: WorkingCapitalParams = field(default_factory=WorkingCapitalParams)
    
    # Fixed costs for manufacturing
    manufacturing_fixed_costs_monthly: float = 0.0  # R - Salaries, rent, utilities
    
    # Historical data
    historical_summary: HistoricalSummary = field(default_factory=HistoricalSummary)
    
    # Part-level data (for part_level analysis mode)
    parts_data: List[LinerPartData] = field(default_factory=list)
    
    # NEW IN v3.0: AI-derived defaults
    ai_defaults_loaded: bool = False
    ai_annual_demand: Optional[int] = None
    ai_avg_selling_price: Optional[float] = None
    ai_avg_cogs: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize scenario to dictionary for database storage."""
        return {
            'strategy': self.strategy,
            'hybrid_make_pct': self.hybrid_make_pct,
            'analysis_mode': self.analysis_mode,
            'product_name': self.product_name,
            'annual_demand_units': self.annual_demand_units,
            'selling_price_per_unit': self.selling_price_per_unit,
            'manufacturing_fixed_costs_monthly': self.manufacturing_fixed_costs_monthly,
            
            # Nested structures - using actual attribute names
            'make_costs': {
                'raw_material_per_unit': self.make_costs.raw_material_per_unit,
                'direct_labor_per_unit': self.make_costs.direct_labor_per_unit,
                'variable_overhead_per_unit': self.make_costs.variable_overhead_per_unit,
                'manufacturing_overhead_pct': self.make_costs.manufacturing_overhead_pct,
                'use_overhead_pct': self.make_costs.use_overhead_pct,
                'quality_control_pct': self.make_costs.quality_control_pct,
                'wastage_pct': self.make_costs.wastage_pct,
            },
            'buy_costs': {
                'purchase_price_per_unit': self.buy_costs.purchase_price_per_unit,
                'freight_per_unit': self.buy_costs.freight_per_unit,
                'import_duty_pct': self.buy_costs.import_duty_pct,
                'quality_inspection_pct': self.buy_costs.quality_inspection_pct,
            },
            'capacity': {
                'monthly_capacity_units': self.capacity.monthly_capacity_units,
                'ramp_up_months': self.capacity.ramp_up_months,
                'initial_capacity_pct': self.capacity.initial_capacity_pct,
                'capacity_step_pct': self.capacity.capacity_step_pct,
            },
            'capex': {
                'equipment_cost': self.capex.equipment_cost,
                'facility_cost': self.capex.facility_cost,
                'tooling_cost': self.capex.tooling_cost,
                'installation_cost': self.capex.installation_cost,
                'working_capital': self.capex.working_capital,
                'equipment_life_years': self.capex.equipment_life_years,
                'facility_life_years': self.capex.facility_life_years,
                'tooling_life_years': self.capex.tooling_life_years,
            },
            'commissioning': {
                'start_month': self.commissioning.start_month,
                'duration_months': self.commissioning.duration_months,
                'monthly_costs': self.commissioning.monthly_costs,
            },
            'wc_params': {
                'raw_material_days': self.wc_params.raw_material_days,
                'manufacturing_days': self.wc_params.manufacturing_days,
                'calculated_wc': self.wc_params.calculated_wc,
                'manual_override': self.wc_params.manual_override,
                'manual_wc_value': self.wc_params.manual_wc_value,
            },
            
            # AI defaults
            'ai_defaults_loaded': self.ai_defaults_loaded,
            'ai_annual_demand': self.ai_annual_demand,
            'ai_avg_selling_price': self.ai_avg_selling_price,
            'ai_avg_cogs': self.ai_avg_cogs,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntegrationScenario':
        """Deserialize scenario from dictionary."""
        if not data:
            return cls()
        
        scenario = cls()
        
        # Simple fields
        scenario.strategy = data.get('strategy', 'buy')
        scenario.hybrid_make_pct = data.get('hybrid_make_pct', 0.5)
        scenario.analysis_mode = data.get('analysis_mode', 'average')
        scenario.product_name = data.get('product_name', 'Crusher Liners')
        scenario.annual_demand_units = data.get('annual_demand_units', 10000)
        scenario.selling_price_per_unit = data.get('selling_price_per_unit', 0.0)
        scenario.manufacturing_fixed_costs_monthly = data.get('manufacturing_fixed_costs_monthly', 0.0)
        
        # Nested structures - using actual attribute names
        if 'make_costs' in data:
            mc = data['make_costs']
            scenario.make_costs = ManufacturingCosts(
                raw_material_per_unit=mc.get('raw_material_per_unit', 0),
                direct_labor_per_unit=mc.get('direct_labor_per_unit', 0),
                variable_overhead_per_unit=mc.get('variable_overhead_per_unit', 0),
                manufacturing_overhead_pct=mc.get('manufacturing_overhead_pct', 0.15),
                use_overhead_pct=mc.get('use_overhead_pct', True),
                quality_control_pct=mc.get('quality_control_pct', 0.05),
                wastage_pct=mc.get('wastage_pct', 0.03),
            )
        
        if 'buy_costs' in data:
            bc = data['buy_costs']
            scenario.buy_costs = BuyCosts(
                purchase_price_per_unit=bc.get('purchase_price_per_unit', 0),
                freight_per_unit=bc.get('freight_per_unit', 0),
                import_duty_pct=bc.get('import_duty_pct', 0),
                quality_inspection_pct=bc.get('quality_inspection_pct', 0.02),
            )
        
        if 'capacity' in data:
            cap = data['capacity']
            scenario.capacity = ManufacturingCapacity(
                monthly_capacity_units=cap.get('monthly_capacity_units', 1000),
                ramp_up_months=cap.get('ramp_up_months', 6),
                initial_capacity_pct=cap.get('initial_capacity_pct', 0.3),
                capacity_step_pct=cap.get('capacity_step_pct', 0.15),
            )
        
        if 'capex' in data:
            cx = data['capex']
            scenario.capex = ManufacturingCapex(
                equipment_cost=cx.get('equipment_cost', 0),
                facility_cost=cx.get('facility_cost', 0),
                tooling_cost=cx.get('tooling_cost', 0),
                installation_cost=cx.get('installation_cost', 0),
                working_capital=cx.get('working_capital', 0),
                equipment_life_years=cx.get('equipment_life_years', 10),
                facility_life_years=cx.get('facility_life_years', 20),
                tooling_life_years=cx.get('tooling_life_years', 5),
            )
        
        if 'commissioning' in data:
            comm = data['commissioning']
            scenario.commissioning = CommissioningSchedule(
                start_month=comm.get('start_month', 6),
                duration_months=comm.get('duration_months', 4),
                monthly_costs=comm.get('monthly_costs', []),
            )
        
        if 'wc_params' in data:
            wc = data['wc_params']
            scenario.wc_params = WorkingCapitalParams(
                raw_material_days=wc.get('raw_material_days', 30),
                manufacturing_days=wc.get('manufacturing_days', 15),
                calculated_wc=wc.get('calculated_wc', 0),
                manual_override=wc.get('manual_override', False),
                manual_wc_value=wc.get('manual_wc_value', 0),
            )
        
        # AI defaults
        scenario.ai_defaults_loaded = data.get('ai_defaults_loaded', False)
        scenario.ai_annual_demand = data.get('ai_annual_demand')
        scenario.ai_avg_selling_price = data.get('ai_avg_selling_price')
        scenario.ai_avg_cogs = data.get('ai_avg_cogs')
        
        return scenario


# =============================================================================
# DATABASE SAVE/LOAD FUNCTIONS
# =============================================================================

def save_manufacturing_strategy(db, scenario_id: str, user_id: str, scenario: 'IntegrationScenario') -> bool:
    """Save manufacturing strategy to database (within assumptions table)."""
    try:
        # Get existing assumptions
        existing = db.get_scenario_assumptions(scenario_id, user_id) or {}
        
        # Add/update manufacturing strategy
        existing['manufacturing_strategy'] = scenario.to_dict()
        existing['manufacturing_strategy_saved'] = True
        existing['manufacturing_strategy_updated'] = datetime.now().isoformat()
        
        # Save back to database
        return db.update_assumptions(scenario_id, user_id, existing)
    except Exception as e:
        st.error(f"Error saving manufacturing strategy: {e}")
        return False


def load_manufacturing_strategy(db, scenario_id: str, user_id: str) -> Optional['IntegrationScenario']:
    """Load manufacturing strategy from database."""
    try:
        assumptions = db.get_scenario_assumptions(scenario_id, user_id)
        if assumptions and 'manufacturing_strategy' in assumptions:
            return IntegrationScenario.from_dict(assumptions['manufacturing_strategy'])
        return None
    except Exception as e:
        st.error(f"Error loading manufacturing strategy: {e}")
        return None

def section_header(title: str, subtitle: str = None):
    """Render a section header."""
    sub = f'<p style="color: {TEXT_MUTED}; margin: 0; font-size: 0.85rem;">{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div style="margin: 1.5rem 0 1rem 0;">
        <h3 style="color: {GOLD}; margin: 0; font-size: 1.2rem;">{title}</h3>
        {sub}
    </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = None, help_text: str = None):
    """Render a metric card."""
    delta_html = f'<div style="color: {TEXT_MUTED}; font-size: 0.75rem;">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div style="
        background: {GOLD_LIGHT};
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    ">
        <div style="font-size: 1.5rem; font-weight: bold; color: {GOLD};">{value}</div>
        <div style="color: {TEXT_MUTED}; font-size: 0.85rem;">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def format_currency(value: float, decimals: int = 0) -> str:
    """Format value as South African Rand currency."""
    if pd.isna(value) or value is None:
        return "-"
    if abs(value) >= 1_000_000_000:
        return f"R {value/1_000_000_000:,.{decimals}f}B"
    elif abs(value) >= 1_000_000:
        return f"R {value/1_000_000:,.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"R {value/1_000:,.{decimals}f}K"
    else:
        return f"R {value:,.{decimals}f}"


def alert_box(message: str, alert_type: str = "info"):
    """Render a styled alert box."""
    colors = {
        "info": (BLUE, "rgba(59, 130, 246, 0.1)"),
        "warning": ("#f59e0b", "rgba(245, 158, 11, 0.1)"),
        "error": (RED, "rgba(239, 68, 68, 0.1)"),
        "success": (GREEN, "rgba(16, 185, 129, 0.1)")
    }
    color, bg = colors.get(alert_type, colors["info"])
    st.markdown(f"""
    <div style="
        background: {bg};
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    ">
        <span style="color: {color};">{message}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# HISTORICAL DATA LOADING
# =============================================================================

def load_historical_liner_data(db, scenario_id: str) -> Tuple[HistoricalSummary, List[LinerPartData]]:
    """
    Load historical liner sales data from database.
    Returns summary statistics and detailed part-level data.
    """
    summary = HistoricalSummary()
    parts = []
    
    try:
        # Try to load from historic_financials for high-level summary
        hist_df = pd.DataFrame()
        
        # Method 1: Use db.get_historic_financials() if available
        if hasattr(db, 'get_historic_financials'):
            data = db.get_historic_financials(scenario_id)
            if data:
                hist_df = pd.DataFrame(data)
        
        # Method 2: Use db.get_historical_financials()
        if hist_df.empty and hasattr(db, 'get_historical_financials'):
            data = db.get_historical_financials(scenario_id)
            if data:
                hist_df = pd.DataFrame(data)
        
        # Method 3: Direct client access
        if hist_df.empty and hasattr(db, 'client'):
            try:
                response = db.client.table('historic_financials').select('*').eq(
                    'scenario_id', scenario_id
                ).execute()
                if response.data:
                    hist_df = pd.DataFrame(response.data)
            except:
                pass
        
        if not hist_df.empty:
            # Calculate summary from historical financials
            if 'total_revenue' in hist_df.columns or 'revenue' in hist_df.columns:
                rev_col = 'total_revenue' if 'total_revenue' in hist_df.columns else 'revenue'
                summary.total_revenue = hist_df[rev_col].sum()
            
            if 'total_cogs' in hist_df.columns or 'cogs' in hist_df.columns:
                cogs_col = 'total_cogs' if 'total_cogs' in hist_df.columns else 'cogs'
                summary.total_cogs = hist_df[cogs_col].sum()
            
            summary.period_months = len(hist_df)
        
        # Try to load detailed parts data from installed_base
        parts_df = pd.DataFrame()
        
        if hasattr(db, 'get_installed_base'):
            data = db.get_installed_base(scenario_id)
            if data:
                parts_df = pd.DataFrame(data)
        elif hasattr(db, 'client'):
            try:
                response = db.client.table('installed_base').select('*').eq(
                    'scenario_id', scenario_id
                ).execute()
                if response.data:
                    parts_df = pd.DataFrame(response.data)
            except:
                pass
        
        if not parts_df.empty:
            # Aggregate by part/liner type if available
            if 'liner_type' in parts_df.columns or 'part_number' in parts_df.columns:
                group_col = 'liner_type' if 'liner_type' in parts_df.columns else 'part_number'
                
                # Get annual liner value as proxy for revenue
                value_col = 'annual_liner_value' if 'annual_liner_value' in parts_df.columns else None
                
                if value_col:
                    # Group by part
                    for name, group in parts_df.groupby(group_col):
                        part = LinerPartData(
                            part_number=str(name),
                            description=str(name),
                            historical_revenue=group[value_col].sum(),
                            historical_units_sold=len(group)
                        )
                        if part.historical_units_sold > 0:
                            part.historical_avg_selling_price = part.historical_revenue / part.historical_units_sold
                        parts.append(part)
        
        # Calculate summary statistics
        if parts:
            summary.unique_parts = len(parts)
            summary.total_units = sum(p.historical_units_sold for p in parts)
            if summary.total_units > 0:
                summary.avg_selling_price = summary.total_revenue / summary.total_units if summary.total_revenue > 0 else sum(p.historical_avg_selling_price * p.historical_units_sold for p in parts) / summary.total_units
        
        # Estimate COGS if not directly available
        if summary.total_cogs == 0 and summary.total_revenue > 0:
            # Assume typical 60% COGS for liners
            summary.total_cogs = summary.total_revenue * 0.60
        
        if summary.total_units > 0:
            summary.avg_cogs_per_unit = summary.total_cogs / summary.total_units
        
        if summary.total_revenue > 0:
            summary.avg_gross_margin_pct = (summary.total_revenue - summary.total_cogs) / summary.total_revenue * 100
        
        # Update parts with COGS estimates
        for part in parts:
            if part.historical_revenue > 0:
                # Estimate COGS based on overall margin
                part.historical_avg_cogs = part.historical_avg_selling_price * (summary.total_cogs / summary.total_revenue) if summary.total_revenue > 0 else part.historical_avg_selling_price * 0.6
                part.gross_margin_pct = (part.historical_avg_selling_price - part.historical_avg_cogs) / part.historical_avg_selling_price * 100 if part.historical_avg_selling_price > 0 else 0
                
    except Exception as e:
        st.warning(f"Could not load historical data: {str(e)}")
    
    return summary, parts


# =============================================================================
# NEW IN V3.0: AI DEFAULTS LOADING
# =============================================================================

def load_ai_defaults(db, scenario_id: str) -> Dict[str, Any]:
    """
    NEW IN v3.0: Load AI-derived defaults from AI Assumptions Engine.
    
    Returns dict with:
    - annual_demand: Estimated annual unit demand
    - avg_selling_price: Average selling price per unit
    - avg_cogs: Average COGS per unit
    - loaded: Whether data was successfully loaded
    """
    defaults = {
        'annual_demand': None,
        'avg_selling_price': None,
        'avg_cogs': None,
        'loaded': False
    }
    
    try:
        # Try to import AI assumptions engine
        from components.ai_assumptions_engine import get_saved_assumptions
        
        ai_assumptions = get_saved_assumptions(db, scenario_id)
        if ai_assumptions and hasattr(ai_assumptions, 'assumptions_saved') and ai_assumptions.assumptions_saved:
            assumptions = ai_assumptions.assumptions if hasattr(ai_assumptions, 'assumptions') else {}
            
            # Extract relevant values
            if 'total_revenue' in assumptions:
                total_rev = assumptions['total_revenue']
                if hasattr(total_rev, 'historical_mean'):
                    defaults['avg_selling_price'] = total_rev.historical_mean / 12
            
            if 'gross_margin_pct' in assumptions:
                margin = assumptions['gross_margin_pct']
                if hasattr(margin, 'historical_mean') and defaults['avg_selling_price']:
                    margin_pct = margin.historical_mean / 100 if margin.historical_mean > 1 else margin.historical_mean
                    defaults['avg_cogs'] = defaults['avg_selling_price'] * (1 - margin_pct)
            
            defaults['loaded'] = True
    except ImportError:
        pass
    except Exception as e:
        pass  # Silently fail, will use fallback
    
    # Fallback: Load from historical financials directly
    if not defaults['loaded'] and hasattr(db, 'client'):
        try:
            response = db.client.table('historic_financials').select(
                'revenue, cogs'
            ).eq('scenario_id', scenario_id).execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                total_revenue = df['revenue'].sum() if 'revenue' in df.columns else 0
                total_cogs = df['cogs'].sum() if 'cogs' in df.columns else 0
                months = len(df)
                
                if months > 0 and total_revenue > 0:
                    avg_monthly_rev = total_revenue / months
                    avg_monthly_cogs = total_cogs / months
                    
                    defaults['avg_selling_price'] = avg_monthly_rev
                    defaults['avg_cogs'] = avg_monthly_cogs
                    # Rough estimate of annual demand
                    defaults['annual_demand'] = int(total_revenue / max(avg_monthly_rev / 100, 1) * 12 / months) if avg_monthly_rev > 0 else 10000
                    defaults['loaded'] = True
        except Exception:
            pass
    
    return defaults


def generate_sample_parts_data() -> List[LinerPartData]:
    """Generate sample parts data for demonstration."""
    sample_parts = [
        ("MN-CL-001", "Cone Liner - Standard", "Cone", 45000, 27000, 120),
        ("MN-CL-002", "Cone Liner - Heavy Duty", "Cone", 68000, 40800, 85),
        ("MN-ML-001", "Mantle Liner - Standard", "Mantle", 52000, 31200, 95),
        ("MN-ML-002", "Mantle Liner - Heavy Duty", "Mantle", 78000, 46800, 65),
        ("MN-BL-001", "Bowl Liner - Standard", "Bowl", 38000, 22800, 150),
        ("MN-BL-002", "Bowl Liner - Heavy Duty", "Bowl", 55000, 33000, 110),
        ("MN-JC-001", "Jaw Crusher Plate - Fixed", "Jaw", 42000, 25200, 180),
        ("MN-JC-002", "Jaw Crusher Plate - Movable", "Jaw", 48000, 28800, 175),
        ("MN-IC-001", "Impact Crusher Bar", "Impact", 22000, 13200, 250),
        ("MN-IC-002", "Impact Crusher Apron", "Impact", 35000, 21000, 90),
        ("MN-WP-001", "Wear Plate - Standard", "Wear Plate", 15000, 9000, 320),
        ("MN-WP-002", "Wear Plate - AR400", "Wear Plate", 25000, 15000, 180),
        ("MN-CH-001", "Chute Liner - Standard", "Chute", 18000, 10800, 200),
        ("MN-CH-002", "Chute Liner - Ceramic Backed", "Chute", 45000, 27000, 75),
        ("MN-SC-001", "Screen Cloth - Polyurethane", "Screen", 8500, 5100, 400),
    ]
    
    parts = []
    for pn, desc, cat, price, cogs, units in sample_parts:
        part = LinerPartData(
            part_number=pn,
            description=desc,
            category=cat,
            historical_avg_selling_price=price,
            historical_avg_cogs=cogs,
            historical_units_sold=units,
            historical_revenue=price * units,
            gross_margin_pct=(price - cogs) / price * 100 if price > 0 else 0,
            manufacture_pct=0,
            mfg_cost_pct_of_buy=75
        )
        parts.append(part)
    
    return parts


# =============================================================================
# IMPORT/EXPORT FUNCTIONS
# =============================================================================

def generate_export_template(parts: List[LinerPartData]) -> pd.DataFrame:
    """Generate a CSV template for part-level manufacturing decisions."""
    data = []
    for part in parts:
        data.append({
            'Part Number': part.part_number,
            'Description': part.description,
            'Category': part.category,
            'Avg Selling Price (R)': part.historical_avg_selling_price,
            'Avg COGS (R)': part.historical_avg_cogs,
            'Historical Units': part.historical_units_sold,
            'Gross Margin %': round(part.gross_margin_pct, 1),
            'Manufacture %': part.manufacture_pct,
            'Mfg Cost as % of Buy': part.mfg_cost_pct_of_buy
        })
    
    return pd.DataFrame(data)


def parse_import_template(uploaded_file) -> List[LinerPartData]:
    """Parse uploaded CSV template and return updated parts list."""
    parts = []
    
    try:
        df = pd.read_csv(uploaded_file)
        
        required_cols = ['Part Number', 'Manufacture %']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return parts
        
        for _, row in df.iterrows():
            part = LinerPartData(
                part_number=str(row.get('Part Number', '')),
                description=str(row.get('Description', '')),
                category=str(row.get('Category', 'Other')),
                historical_avg_selling_price=float(row.get('Avg Selling Price (R)', 0)),
                historical_avg_cogs=float(row.get('Avg COGS (R)', 0)),
                historical_units_sold=int(row.get('Historical Units', 0)),
                gross_margin_pct=float(row.get('Gross Margin %', 0)),
                manufacture_pct=float(row.get('Manufacture %', 0)),
                mfg_cost_pct_of_buy=float(row.get('Mfg Cost as % of Buy', 75))
            )
            part.historical_revenue = part.historical_avg_selling_price * part.historical_units_sold
            parts.append(part)
            
    except Exception as e:
        st.error(f"Error parsing import file: {str(e)}")
    
    return parts


# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def calculate_weighted_costs(parts: List[LinerPartData]) -> Dict[str, float]:
    """
    Calculate weighted average costs based on part-level manufacturing decisions.
    """
    total_units = sum(p.historical_units_sold for p in parts)
    if total_units == 0:
        return {
            'weighted_buy_cost': 0,
            'weighted_make_cost': 0,
            'weighted_blended_cost': 0,
            'total_make_units': 0,
            'total_buy_units': 0,
            'overall_make_pct': 0
        }
    
    weighted_buy_cost = 0
    weighted_make_cost = 0
    weighted_blended_cost = 0
    total_make_units = 0
    total_buy_units = 0
    
    for part in parts:
        units = part.historical_units_sold
        make_pct = part.manufacture_pct / 100
        buy_pct = 1 - make_pct
        
        # Buy cost = historical COGS (what we currently pay)
        buy_cost_per_unit = part.historical_avg_cogs
        
        # Make cost = % of buy cost based on user's estimate
        make_cost_per_unit = buy_cost_per_unit * (part.mfg_cost_pct_of_buy / 100)
        
        # Calculate weighted contributions
        make_units = units * make_pct
        buy_units = units * buy_pct
        
        weighted_buy_cost += buy_cost_per_unit * buy_units
        weighted_make_cost += make_cost_per_unit * make_units
        
        # Blended cost for this part
        blended = (make_cost_per_unit * make_pct + buy_cost_per_unit * buy_pct) * units
        weighted_blended_cost += blended
        
        total_make_units += make_units
        total_buy_units += buy_units
    
    overall_make_pct = total_make_units / total_units * 100 if total_units > 0 else 0
    
    # Calculate proper per-unit averages
    # weighted_buy_cost is total spend on bought units, divide by buy_units for per-unit average
    avg_buy_cost_per_unit = weighted_buy_cost / total_buy_units if total_buy_units > 0 else 0
    # weighted_make_cost is total spend on made units, divide by make_units for per-unit average
    avg_make_cost_per_unit = weighted_make_cost / total_make_units if total_make_units > 0 else 0
    # Blended cost per unit across all units
    avg_blended_cost_per_unit = weighted_blended_cost / total_units if total_units > 0 else 0
    
    # Calculate potential savings: difference between what we'd pay to buy vs make for manufactured units
    # If we manufacture, we pay avg_make_cost; if we bought those same units, we'd pay avg_buy_cost
    # Need to use the buy cost for the parts we're choosing to manufacture
    potential_savings_per_unit = avg_buy_cost_per_unit - avg_make_cost_per_unit if avg_buy_cost_per_unit > 0 else 0
    total_potential_savings = potential_savings_per_unit * total_make_units
    
    return {
        'weighted_buy_cost': avg_buy_cost_per_unit,  # Average cost per unit when buying
        'weighted_make_cost': avg_make_cost_per_unit,  # Average cost per unit when manufacturing
        'weighted_blended_cost': avg_blended_cost_per_unit,  # Blended average across make/buy mix
        'total_make_units': total_make_units,
        'total_buy_units': total_buy_units,
        'total_units': total_units,
        'overall_make_pct': overall_make_pct,
        'total_buy_spend': weighted_buy_cost,  # Total $ spent on buying
        'total_make_spend': weighted_make_cost,  # Total $ spent on manufacturing
        'potential_savings': total_potential_savings,  # Total savings from manufacturing vs buying
        'savings_pct': (potential_savings_per_unit / avg_buy_cost_per_unit * 100) if avg_buy_cost_per_unit > 0 else 0
    }


def calculate_make_scenario(
    scenario: IntegrationScenario,
    months: int = 60
) -> Dict[str, Any]:
    """Calculate financials for Make (manufacturing) scenario."""
    
    results = {
        'months': [],
        'demand': [],
        'production': [],
        'outsourced': [],
        'revenue': [],
        'cogs_make': [],
        'cogs_buy': [],
        'total_cogs': [],
        'gross_profit': [],
        'fixed_costs': [],
        'depreciation': [],
        'ebit': [],
        'cumulative_profit': [],
        'cumulative_cash': []
    }
    
    monthly_demand = scenario.annual_demand_units / 12
    cumulative_profit = -scenario.capex.total_capex  # Start with CAPEX investment
    cumulative_cash = cumulative_profit
    
    for month in range(months):
        # Demand for this month
        demand = monthly_demand
        
        # Capacity constraint
        capacity = scenario.capacity.get_capacity_for_month(month)
        production = min(demand, capacity)
        outsourced = demand - production
        
        # Revenue
        revenue = demand * scenario.selling_price_per_unit
        
        # COGS
        cogs_make = production * scenario.make_costs.total_variable_cost_per_unit
        cogs_buy = outsourced * scenario.buy_costs.total_landed_cost_per_unit
        total_cogs = cogs_make + cogs_buy
        
        # Gross profit
        gross_profit = revenue - total_cogs
        
        # Operating costs
        fixed_costs = scenario.manufacturing_fixed_costs_monthly
        depreciation = scenario.capex.monthly_depreciation
        
        # EBIT
        ebit = gross_profit - fixed_costs - depreciation
        
        # Cumulative
        cumulative_profit += ebit
        cumulative_cash += (ebit + depreciation)  # Add back non-cash depreciation
        
        # Store results
        results['months'].append(month + 1)
        results['demand'].append(demand)
        results['production'].append(production)
        results['outsourced'].append(outsourced)
        results['revenue'].append(revenue)
        results['cogs_make'].append(cogs_make)
        results['cogs_buy'].append(cogs_buy)
        results['total_cogs'].append(total_cogs)
        results['gross_profit'].append(gross_profit)
        results['fixed_costs'].append(fixed_costs)
        results['depreciation'].append(depreciation)
        results['ebit'].append(ebit)
        results['cumulative_profit'].append(cumulative_profit)
        results['cumulative_cash'].append(cumulative_cash)
    
    return results


def calculate_buy_scenario(
    scenario: IntegrationScenario,
    months: int = 60
) -> Dict[str, Any]:
    """Calculate financials for Buy (outsource) scenario."""
    
    results = {
        'months': [],
        'demand': [],
        'revenue': [],
        'cogs': [],
        'gross_profit': [],
        'cumulative_profit': [],
        'cumulative_cash': []
    }
    
    monthly_demand = scenario.annual_demand_units / 12
    cumulative_profit = 0
    cumulative_cash = 0
    
    for month in range(months):
        demand = monthly_demand
        revenue = demand * scenario.selling_price_per_unit
        cogs = demand * scenario.buy_costs.total_landed_cost_per_unit
        gross_profit = revenue - cogs
        
        cumulative_profit += gross_profit
        cumulative_cash += gross_profit
        
        results['months'].append(month + 1)
        results['demand'].append(demand)
        results['revenue'].append(revenue)
        results['cogs'].append(cogs)
        results['gross_profit'].append(gross_profit)
        results['cumulative_profit'].append(cumulative_profit)
        results['cumulative_cash'].append(cumulative_cash)
    
    return results


def calculate_hybrid_scenario(
    scenario: IntegrationScenario,
    months: int = 60
) -> Dict[str, Any]:
    """Calculate financials for Hybrid scenario."""
    
    results = {
        'months': [],
        'demand': [],
        'production': [],
        'outsourced': [],
        'revenue': [],
        'cogs_make': [],
        'cogs_buy': [],
        'total_cogs': [],
        'gross_profit': [],
        'fixed_costs': [],
        'depreciation': [],
        'ebit': [],
        'cumulative_profit': [],
        'cumulative_cash': []
    }
    
    monthly_demand = scenario.annual_demand_units / 12
    
    # Scale CAPEX based on hybrid percentage
    scaled_capex = scenario.capex.total_capex * scenario.hybrid_make_pct
    cumulative_profit = -scaled_capex
    cumulative_cash = cumulative_profit
    
    for month in range(months):
        demand = monthly_demand
        target_make = demand * scenario.hybrid_make_pct
        
        # Capacity constraint (scaled)
        full_capacity = scenario.capacity.get_capacity_for_month(month)
        scaled_capacity = full_capacity * scenario.hybrid_make_pct
        production = min(target_make, scaled_capacity)
        outsourced = demand - production
        
        # Revenue
        revenue = demand * scenario.selling_price_per_unit
        
        # COGS
        cogs_make = production * scenario.make_costs.total_variable_cost_per_unit
        cogs_buy = outsourced * scenario.buy_costs.total_landed_cost_per_unit
        total_cogs = cogs_make + cogs_buy
        
        # Gross profit
        gross_profit = revenue - total_cogs
        
        # Operating costs (scaled)
        fixed_costs = scenario.manufacturing_fixed_costs_monthly * scenario.hybrid_make_pct
        depreciation = scenario.capex.monthly_depreciation * scenario.hybrid_make_pct
        
        # EBIT
        ebit = gross_profit - fixed_costs - depreciation
        
        # Cumulative
        cumulative_profit += ebit
        cumulative_cash += (ebit + depreciation)
        
        results['months'].append(month + 1)
        results['demand'].append(demand)
        results['production'].append(production)
        results['outsourced'].append(outsourced)
        results['revenue'].append(revenue)
        results['cogs_make'].append(cogs_make)
        results['cogs_buy'].append(cogs_buy)
        results['total_cogs'].append(total_cogs)
        results['gross_profit'].append(gross_profit)
        results['fixed_costs'].append(fixed_costs)
        results['depreciation'].append(depreciation)
        results['ebit'].append(ebit)
        results['cumulative_profit'].append(cumulative_profit)
        results['cumulative_cash'].append(cumulative_cash)
    
    return results


def calculate_npv(cash_flows: List[float], discount_rate: float = 0.12) -> float:
    """Calculate NPV of cash flows."""
    monthly_rate = (1 + discount_rate) ** (1/12) - 1
    npv = sum(cf / (1 + monthly_rate) ** i for i, cf in enumerate(cash_flows))
    return npv


def calculate_irr(cash_flows: List[float]) -> Optional[float]:
    """Calculate IRR of cash flows using iterative method."""
    try:
        # Initial guess
        low, high = -0.99, 10.0
        
        for _ in range(100):
            mid = (low + high) / 2
            npv = sum(cf / (1 + mid) ** i for i, cf in enumerate(cash_flows))
            
            if abs(npv) < 0.01:
                return mid * 12  # Annualize
            
            if npv > 0:
                low = mid
            else:
                high = mid
        
        return mid * 12
    except:
        return None


def calculate_payback_months(cumulative_cash: List[float]) -> int:
    """Calculate payback period in months."""
    for i, cash in enumerate(cumulative_cash):
        if cash >= 0:
            return i + 1
    return len(cumulative_cash)  # Not recovered within period


def calculate_break_even_volume(scenario: IntegrationScenario) -> Dict[str, Any]:
    """Calculate break-even analysis."""
    
    # Make scenario costs
    make_variable = scenario.make_costs.total_variable_cost_per_unit
    make_fixed_monthly = scenario.manufacturing_fixed_costs_monthly + scenario.capex.monthly_depreciation
    
    # Buy scenario costs
    buy_variable = scenario.buy_costs.total_landed_cost_per_unit
    
    # Break-even calculation
    # At break-even: Make_Total_Cost = Buy_Total_Cost
    # make_fixed + make_var * Q = buy_var * Q
    # make_fixed = (buy_var - make_var) * Q
    # Q = make_fixed / (buy_var - make_var)
    
    variable_savings_per_unit = buy_variable - make_variable
    
    if variable_savings_per_unit <= 0:
        # Manufacturing is more expensive per unit - never breaks even
        break_even_monthly = None
        break_even_annual = None
        crossover_message = "Manufacturing cost per unit exceeds purchase cost - consider reviewing cost structure"
    else:
        break_even_monthly = make_fixed_monthly / variable_savings_per_unit
        break_even_annual = break_even_monthly * 12
        crossover_message = None
    
    return {
        'make_variable_cost': make_variable,
        'buy_variable_cost': buy_variable,
        'make_fixed_monthly': make_fixed_monthly,
        'variable_savings_per_unit': variable_savings_per_unit,
        'break_even_monthly': break_even_monthly,
        'break_even_annual': break_even_annual,
        'crossover_message': crossover_message
    }


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_vertical_integration_section(db, scenario_id: str, user_id: str):
    """
    Main entry point for rendering the Vertical Integration / Manufacturing Strategy section.
    """
    st.header("🏭 Manufacturing Strategy")
    st.caption("Make vs Buy analysis with historical data integration")
    
    # Initialize scenario in session state - TRY DATABASE FIRST
    if 'vi_scenario' not in st.session_state or st.session_state.get('vi_scenario_id') != scenario_id:
        # Try to load from database
        loaded_scenario = load_manufacturing_strategy(db, scenario_id, user_id)
        if loaded_scenario:
            st.session_state.vi_scenario = loaded_scenario
            st.session_state.vi_scenario_loaded_from_db = True
        else:
            st.session_state.vi_scenario = IntegrationScenario()
            st.session_state.vi_scenario_loaded_from_db = False
        st.session_state.vi_scenario_id = scenario_id
    
    scenario = st.session_state.vi_scenario
    
    # Show save status
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.session_state.get('vi_scenario_loaded_from_db'):
            st.success("✅ Manufacturing strategy loaded from saved data")
        else:
            st.info("ℹ️ New manufacturing strategy - configure and save to persist")
    
    with col2:
        if st.button("💾 Save Strategy", type="primary", key="vi_save_btn", use_container_width=True):
            with st.spinner("Saving..."):
                if save_manufacturing_strategy(db, scenario_id, user_id, scenario):
                    st.session_state.vi_scenario_loaded_from_db = True
                    # Mark workflow stage as complete
                    try:
                        from app_refactored import mark_workflow_stage_complete
                        mark_workflow_stage_complete(db, scenario_id, 'manufacturing', user_id)
                        # Clear cached progress to force refresh
                        cache_key = f"workflow_progress_{scenario_id}"
                        if cache_key in st.session_state:
                            del st.session_state[cache_key]
                    except:
                        pass  # Non-critical
                    st.success("✅ Saved!")
                    st.rerun()
    
    with col3:
        if st.button("🔄 Reset", key="vi_reset_btn", use_container_width=True):
            st.session_state.vi_scenario = IntegrationScenario()
            st.session_state.vi_scenario_loaded_from_db = False
            if 'vi_historical_loaded' in st.session_state:
                del st.session_state.vi_historical_loaded
            st.rerun()
    
    st.markdown("---")
    
    # Load historical data
    if 'vi_historical_loaded' not in st.session_state:
        with st.spinner("Loading historical liner data..."):
            summary, parts = load_historical_liner_data(db, scenario_id)
            
            # If no parts data, use sample data
            if not parts:
                parts = generate_sample_parts_data()
                st.info("📊 Using sample parts data. Import your actual parts list via the template for accurate analysis.")
            
            scenario.historical_summary = summary
            scenario.parts_data = parts
            
            # Auto-populate costs from historical data if available
            if summary.avg_selling_price > 0:
                scenario.selling_price_per_unit = summary.avg_selling_price
            if summary.avg_cogs_per_unit > 0:
                scenario.buy_costs.purchase_price_per_unit = summary.avg_cogs_per_unit
                # Estimate manufacturing cost as 75% of buy cost
                scenario.make_costs.raw_material_per_unit = summary.avg_cogs_per_unit * 0.50
                scenario.make_costs.direct_labor_per_unit = summary.avg_cogs_per_unit * 0.20
            if summary.total_units > 0:
                scenario.annual_demand_units = summary.total_units
            
            st.session_state.vi_historical_loaded = True
    
    # NEW IN v3.0: Load AI defaults if not already loaded
    if not scenario.ai_defaults_loaded:
        ai_defaults = load_ai_defaults(db, scenario_id)
        if ai_defaults['loaded']:
            scenario.ai_annual_demand = ai_defaults.get('annual_demand')
            scenario.ai_avg_selling_price = ai_defaults.get('avg_selling_price')
            scenario.ai_avg_cogs = ai_defaults.get('avg_cogs')
            scenario.ai_defaults_loaded = True
            
            # Apply defaults if scenario values are still at defaults
            if scenario.annual_demand_units == 10000 and scenario.ai_annual_demand:
                scenario.annual_demand_units = scenario.ai_annual_demand
            if scenario.selling_price_per_unit == 0 and scenario.ai_avg_selling_price:
                scenario.selling_price_per_unit = scenario.ai_avg_selling_price
    
    # Show AI defaults status
    if scenario.ai_defaults_loaded:
        with st.expander("🤖 AI-Derived Defaults (v3.0)", expanded=False):
            st.success("✅ AI defaults loaded from historical analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Est. Annual Demand", f"{scenario.ai_annual_demand:,}" if scenario.ai_annual_demand else "N/A")
            with col2:
                st.metric("Avg Selling Price", format_currency(scenario.ai_avg_selling_price or 0))
            with col3:
                st.metric("Avg COGS", format_currency(scenario.ai_avg_cogs or 0))
    
    # Display historical summary
    if scenario.historical_summary.total_revenue > 0:
        with st.expander("📊 Historical Data Summary", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Revenue", format_currency(scenario.historical_summary.total_revenue))
            with col2:
                st.metric("Total COGS", format_currency(scenario.historical_summary.total_cogs))
            with col3:
                st.metric("Avg Selling Price", format_currency(scenario.historical_summary.avg_selling_price))
            with col4:
                st.metric("Avg Gross Margin", f"{scenario.historical_summary.avg_gross_margin_pct:.1f}%")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚙️ Configuration",
        "📊 Cost Comparison",
        "📈 Break-Even",
        "💰 NPV Analysis",
        "📋 Summary"
    ])
    
    # ==========================================================================
    # TAB 1: Configuration
    # ==========================================================================
    with tab1:
        render_configuration_tab_v2(scenario, db, scenario_id)
    
    # ==========================================================================
    # TAB 2: Cost Comparison
    # ==========================================================================
    with tab2:
        render_cost_comparison_tab(scenario)
    
    # ==========================================================================
    # TAB 3: Break-Even Analysis
    # ==========================================================================
    with tab3:
        render_break_even_tab(scenario)
    
    # ==========================================================================
    # TAB 4: NPV Analysis
    # ==========================================================================
    with tab4:
        render_npv_tab(scenario)
    
    # ==========================================================================
    # TAB 5: Summary & Recommendation
    # ==========================================================================
    with tab5:
        render_summary_tab(scenario)


def render_configuration_tab_v2(scenario: IntegrationScenario, db, scenario_id: str):
    """Render the enhanced configuration/input tab with analysis mode toggle."""
    
    section_header("Analysis Mode Selection", "Choose how to configure manufacturing decisions")
    
    # ==========================================================================
    # ANALYSIS MODE TOGGLE
    # ==========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: {bg};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <h4 style="color: {gold}; margin: 0;">📊 Average Mode</h4>
            <p style="color: {muted}; font-size: 0.85rem; margin: 0.5rem 0 0 0;">
            Use historical averages with a blanket % manufactured vs outsourced.
            Quick analysis without part-level detail.
            </p>
        </div>
        """.format(bg=GOLD_LIGHT, border=BORDER_COLOR, gold=GOLD, muted=TEXT_MUTED), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid {border};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <h4 style="color: {blue}; margin: 0;">📋 Part-Level Mode</h4>
            <p style="color: {muted}; font-size: 0.85rem; margin: 0.5rem 0 0 0;">
            Select individual liners and apply specific cost % to each.
            Use export/import template for bulk configuration.
            </p>
        </div>
        """.format(border=BORDER_COLOR, blue=BLUE, muted=TEXT_MUTED), unsafe_allow_html=True)
    
    scenario.analysis_mode = st.radio(
        "Select Analysis Mode",
        options=['average', 'part_level'],
        format_func=lambda x: '📊 Average Mode (Blanket %)' if x == 'average' else '📋 Part-Level Mode (Individual Selection)',
        key="vi_analysis_mode",
        horizontal=True
    )
    
    st.markdown("---")
    
    # ==========================================================================
    # AVERAGE MODE CONFIGURATION
    # ==========================================================================
    if scenario.analysis_mode == 'average':
        render_average_mode_config(scenario)
    
    # ==========================================================================
    # PART-LEVEL MODE CONFIGURATION
    # ==========================================================================
    else:
        render_part_level_mode_config(scenario)


def render_average_mode_config(scenario: IntegrationScenario):
    """Render configuration for Average Mode."""
    
    section_header("Average Mode Configuration", "Based on historical averages with blanket manufacturing percentage")
    
    # Basic Info from Historical Data
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Product Information")
        st.caption("Pre-populated from historical data")
        
        scenario.product_name = st.text_input(
            "Product Name",
            value=scenario.product_name,
            key="vi_product_name_avg"
        )
        scenario.annual_demand_units = st.number_input(
            "Annual Demand (units)",
            min_value=100,
            max_value=1000000,
            value=scenario.annual_demand_units,
            step=100,
            key="vi_annual_demand_avg",
            help="From historical data"
        )
        scenario.selling_price_per_unit = st.number_input(
            "Avg Selling Price per Unit (R)",
            min_value=0.0,
            value=float(scenario.selling_price_per_unit),
            step=100.0,
            format="%.2f",
            key="vi_selling_price_avg",
            help="Historical average selling price"
        )
    
    with col2:
        st.markdown("### Strategy Selection")
        strategy_options = {
            'buy': '🛒 Buy (Outsource)',
            'make': '🏭 Make (Manufacture)',
            'hybrid': '🔄 Hybrid (Both)'
        }
        scenario.strategy = st.radio(
            "Manufacturing Strategy",
            options=list(strategy_options.keys()),
            format_func=lambda x: strategy_options[x],
            key="vi_strategy_avg",
            horizontal=True
        )
        
        if scenario.strategy == 'hybrid':
            scenario.hybrid_make_pct = st.slider(
                "% to Manufacture In-House",
                min_value=10,
                max_value=90,
                value=int(scenario.hybrid_make_pct * 100),
                key="vi_hybrid_pct_avg"
            ) / 100
            
            # Show impact
            make_units = scenario.annual_demand_units * scenario.hybrid_make_pct
            buy_units = scenario.annual_demand_units * (1 - scenario.hybrid_make_pct)
            st.info(f"**Make:** {make_units:,.0f} units/year | **Buy:** {buy_units:,.0f} units/year")
    
    st.markdown("---")
    
    # Cost Structures
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏭 Manufacturing Costs (Make)")
        st.caption("Estimate your in-house manufacturing costs per unit")
        
        # Direct Costs
        st.markdown("**Direct Costs (per unit)**")
        scenario.make_costs.raw_material_per_unit = st.number_input(
            "Raw Material Cost (R/unit)",
            min_value=0.0,
            value=float(scenario.make_costs.raw_material_per_unit),
            step=100.0,
            format="%.2f",
            key="vi_raw_material_avg",
            help="Cost of raw materials per finished unit"
        )
        scenario.make_costs.direct_labor_per_unit = st.number_input(
            "Direct Labor Cost (R/unit)",
            min_value=0.0,
            value=float(scenario.make_costs.direct_labor_per_unit),
            step=50.0,
            format="%.2f",
            key="vi_direct_labor_avg",
            help="Direct labor cost per finished unit"
        )
        
        # Variable Overhead - choice between % or fixed per unit
        st.markdown("**Variable Overhead (per unit)**")
        overhead_method = st.radio(
            "Overhead calculation method",
            options=["Percentage of direct costs", "Fixed amount per unit"],
            index=0 if scenario.make_costs.use_overhead_pct else 1,
            key="vi_overhead_method",
            horizontal=True
        )
        scenario.make_costs.use_overhead_pct = (overhead_method == "Percentage of direct costs")
        
        if scenario.make_costs.use_overhead_pct:
            scenario.make_costs.manufacturing_overhead_pct = st.slider(
                "Variable Overhead (% of direct costs)",
                min_value=0,
                max_value=50,
                value=int(scenario.make_costs.manufacturing_overhead_pct * 100),
                key="vi_mfg_overhead_pct",
                help="Variable manufacturing overhead as % of direct costs (materials + labor)"
            ) / 100
            calc_var_overhead = scenario.make_costs.variable_overhead_amount
            st.caption(f"= {format_currency(calc_var_overhead)}/unit")
        else:
            scenario.make_costs.variable_overhead_per_unit = st.number_input(
                "Variable Overhead (R/unit)",
                min_value=0.0,
                value=float(scenario.make_costs.variable_overhead_per_unit),
                step=50.0,
                format="%.2f",
                key="vi_mfg_overhead_unit",
                help="Variable overhead per unit (utilities, consumables, etc.)"
            )
        
        # Other variable costs
        st.markdown("**Other Variable Costs**")
        col_a, col_b = st.columns(2)
        with col_a:
            scenario.make_costs.wastage_pct = st.slider(
                "Material Wastage (%)",
                min_value=0,
                max_value=15,
                value=int(scenario.make_costs.wastage_pct * 100),
                key="vi_wastage_avg"
            ) / 100
        with col_b:
            scenario.make_costs.quality_control_pct = st.slider(
                "Quality Control (%)",
                min_value=0,
                max_value=10,
                value=int(scenario.make_costs.quality_control_pct * 100),
                key="vi_qc_avg"
            ) / 100
        
        st.markdown("---")
        
        # Fixed Monthly Overhead
        st.markdown("**Fixed Monthly Overhead**")
        scenario.manufacturing_fixed_costs_monthly = st.number_input(
            "Monthly Fixed Costs (R)",
            min_value=0.0,
            value=float(scenario.manufacturing_fixed_costs_monthly),
            step=10000.0,
            format="%.0f",
            key="vi_fixed_costs_avg",
            help="Fixed costs: Salaries, rent, utilities, insurance, maintenance"
        )
        
        # Show cost breakdown
        breakdown = scenario.make_costs.cost_breakdown
        st.markdown("**Cost Breakdown (per unit):**")
        breakdown_text = f"""
        | Component | Amount |
        |-----------|--------|
        | Raw Materials (incl. wastage) | {format_currency(breakdown['raw_material'])} |
        | Direct Labor | {format_currency(breakdown['direct_labor'])} |
        | Variable Overhead | {format_currency(breakdown['variable_overhead'])} |
        | Quality Control | {format_currency(breakdown['quality_control'])} |
        | **Total Variable Cost** | **{format_currency(breakdown['total'])}** |
        """
        st.markdown(breakdown_text)
        
        if scenario.manufacturing_fixed_costs_monthly > 0:
            st.caption(f"Plus fixed overhead: {format_currency(scenario.manufacturing_fixed_costs_monthly)}/month")
    
    with col2:
        st.markdown("### 🛒 Purchase Costs (Buy)")
        st.caption("Current outsourcing costs from historical data")
        
        scenario.buy_costs.purchase_price_per_unit = st.number_input(
            "Supplier Price per Unit (R)",
            min_value=0.0,
            value=float(scenario.buy_costs.purchase_price_per_unit),
            step=100.0,
            format="%.2f",
            key="vi_purchase_price_avg",
            help="Historical average COGS per unit"
        )
        scenario.buy_costs.freight_per_unit = st.number_input(
            "Freight per Unit (R)",
            min_value=0.0,
            value=float(scenario.buy_costs.freight_per_unit),
            step=50.0,
            format="%.2f",
            key="vi_freight_avg"
        )
        scenario.buy_costs.import_duty_pct = st.slider(
            "Import Duty (%)",
            min_value=0,
            max_value=30,
            value=int(scenario.buy_costs.import_duty_pct * 100),
            key="vi_duty_avg"
        ) / 100
        scenario.buy_costs.quality_inspection_pct = st.slider(
            "Quality Inspection (%)",
            min_value=0,
            max_value=10,
            value=int(scenario.buy_costs.quality_inspection_pct * 100),
            key="vi_inspection_avg"
        ) / 100
        
        # Show calculated cost
        st.info(f"**Calculated Landed Cost:** {format_currency(scenario.buy_costs.total_landed_cost_per_unit)} per unit")
        
        # Cost comparison
        make_cost = scenario.make_costs.total_variable_cost_per_unit
        buy_cost = scenario.buy_costs.total_landed_cost_per_unit
        
        if make_cost > 0 and buy_cost > 0:
            savings_pct = (buy_cost - make_cost) / buy_cost * 100
            st.markdown("---")
            st.markdown("**📊 Cost Comparison**")
            
            col_x, col_y = st.columns(2)
            with col_x:
                st.metric("Make Cost", format_currency(make_cost))
            with col_y:
                st.metric("Buy Cost", format_currency(buy_cost))
            
            if savings_pct > 0:
                st.success(f"📉 Manufacturing saves **{savings_pct:.1f}%** per unit (variable cost only)")
            else:
                st.warning(f"📈 Buying is **{abs(savings_pct):.1f}%** cheaper per unit")
            
            # Note about fixed costs
            if scenario.manufacturing_fixed_costs_monthly > 0:
                monthly_units = scenario.annual_demand_units / 12
                fixed_per_unit = scenario.manufacturing_fixed_costs_monthly / max(monthly_units, 1)
                st.caption(f"Note: Fixed overhead adds {format_currency(fixed_per_unit)}/unit at current volume")
    
    # CAPEX Section
    render_capex_section(scenario)


def render_part_level_mode_config(scenario: IntegrationScenario):
    """Render configuration for Part-Level Mode."""
    
    section_header("Part-Level Configuration", "Select individual liners and configure manufacturing percentage")
    
    # ==========================================================================
    # IMPORT/EXPORT TEMPLATE
    # ==========================================================================
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("#### 📥 Import/Export Template")
        st.caption("Use templates for bulk configuration of manufacturing decisions")
    
    with col2:
        # Export button
        if scenario.parts_data:
            template_df = generate_export_template(scenario.parts_data)
            csv_data = template_df.to_csv(index=False)
            
            st.download_button(
                label="📤 Export Template",
                data=csv_data,
                file_name=f"manufacturing_template_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="vi_export_template",
                help="Download current parts list as CSV template"
            )
    
    with col3:
        # Import button
        uploaded_file = st.file_uploader(
            "📥 Import",
            type=['csv'],
            key="vi_import_template",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            imported_parts = parse_import_template(uploaded_file)
            if imported_parts:
                scenario.parts_data = imported_parts
                st.success(f"✅ Imported {len(imported_parts)} parts")
                st.rerun()
    
    st.markdown("---")
    
    # ==========================================================================
    # PART LIST WITH FILTERS
    # ==========================================================================
    if not scenario.parts_data:
        st.warning("No parts data available. Please import a template or load historical data.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        categories = list(set(p.category for p in scenario.parts_data))
        selected_categories = st.multiselect(
            "Filter by Category",
            options=categories,
            default=categories,
            key="vi_category_filter"
        )
    
    with col2:
        search_term = st.text_input(
            "Search Parts",
            placeholder="Part number or description...",
            key="vi_part_search"
        )
    
    with col3:
        show_only_mfg = st.checkbox(
            "Show only parts to manufacture",
            value=False,
            key="vi_show_mfg_only"
        )
    
    # ==========================================================================
    # QUICK ACTIONS
    # ==========================================================================
    st.markdown("#### ⚡ Quick Actions")
    
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    
    with qcol1:
        if st.button("Set All to 0%", key="vi_set_all_0"):
            for part in scenario.parts_data:
                part.manufacture_pct = 0
            st.rerun()
    
    with qcol2:
        if st.button("Set All to 50%", key="vi_set_all_50"):
            for part in scenario.parts_data:
                part.manufacture_pct = 50
            st.rerun()
    
    with qcol3:
        if st.button("Set All to 100%", key="vi_set_all_100"):
            for part in scenario.parts_data:
                part.manufacture_pct = 100
            st.rerun()
    
    with qcol4:
        default_mfg_cost_pct = st.number_input(
            "Default Mfg Cost %",
            min_value=50,
            max_value=100,
            value=75,
            key="vi_default_mfg_cost"
        )
        if st.button("Apply to All", key="vi_apply_mfg_cost"):
            for part in scenario.parts_data:
                part.mfg_cost_pct_of_buy = default_mfg_cost_pct
            st.rerun()
    
    st.markdown("---")
    
    # ==========================================================================
    # PARTS TABLE
    # ==========================================================================
    st.markdown("#### 📋 Parts Configuration")
    
    # Filter parts
    filtered_parts = scenario.parts_data
    
    if selected_categories:
        filtered_parts = [p for p in filtered_parts if p.category in selected_categories]
    
    if search_term:
        search_lower = search_term.lower()
        filtered_parts = [p for p in filtered_parts if 
                        search_lower in p.part_number.lower() or 
                        search_lower in p.description.lower()]
    
    if show_only_mfg:
        filtered_parts = [p for p in filtered_parts if p.manufacture_pct > 0]
    
    if not filtered_parts:
        st.info("No parts match the current filters.")
        return
    
    # Create editable dataframe
    parts_df = pd.DataFrame([{
        'Part Number': p.part_number,
        'Description': p.description[:30] + '...' if len(p.description) > 30 else p.description,
        'Category': p.category,
        'Sell Price (R)': p.historical_avg_selling_price,
        'Buy Cost (R)': p.historical_avg_cogs,
        'Margin %': p.gross_margin_pct,
        'Units': p.historical_units_sold,
        'Mfg %': p.manufacture_pct,
        'Mfg Cost %': p.mfg_cost_pct_of_buy
    } for p in filtered_parts])
    
    # Display as styled table with edit capability
    st.markdown(f"""
    <style>
    .parts-table {{
        font-size: 0.85rem;
    }}
    .parts-table th {{
        background: {GOLD_LIGHT};
        color: {GOLD};
        padding: 0.5rem;
    }}
    .parts-table td {{
        padding: 0.3rem 0.5rem;
        border-bottom: 1px solid {BORDER_COLOR};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Use data editor for interactive editing
    edited_df = st.data_editor(
        parts_df,
        column_config={
            "Part Number": st.column_config.TextColumn("Part Number", disabled=True),
            "Description": st.column_config.TextColumn("Description", disabled=True, width="medium"),
            "Category": st.column_config.TextColumn("Category", disabled=True),
            "Sell Price (R)": st.column_config.NumberColumn("Sell Price", format="R %.0f", disabled=True),
            "Buy Cost (R)": st.column_config.NumberColumn("Buy Cost", format="R %.0f", disabled=True),
            "Margin %": st.column_config.NumberColumn("Margin", format="%.1f%%", disabled=True),
            "Units": st.column_config.NumberColumn("Units", disabled=True),
            "Mfg %": st.column_config.NumberColumn(
                "Manufacture %",
                help="% of this part to manufacture in-house",
                min_value=0,
                max_value=100,
                step=5
            ),
            "Mfg Cost %": st.column_config.NumberColumn(
                "Mfg Cost % of Buy",
                help="Expected manufacturing cost as % of current buy cost",
                min_value=50,
                max_value=120,
                step=5
            )
        },
        hide_index=True,
        use_container_width=True,
        key="vi_parts_editor"
    )
    
    # Update scenario parts data from edited dataframe
    if edited_df is not None:
        for idx, row in edited_df.iterrows():
            # Find matching part
            for part in scenario.parts_data:
                if part.part_number == row['Part Number']:
                    part.manufacture_pct = row['Mfg %']
                    part.mfg_cost_pct_of_buy = row['Mfg Cost %']
                    break
    
    # ==========================================================================
    # PART-LEVEL SUMMARY
    # ==========================================================================
    st.markdown("---")
    st.markdown("#### 📊 Configuration Summary")
    
    # Calculate weighted costs
    weighted = calculate_weighted_costs(scenario.parts_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Mfg %",
            f"{weighted['overall_make_pct']:.1f}%",
            help="Weighted average of manufacturing percentage"
        )
    
    with col2:
        st.metric(
            "Units to Make",
            f"{weighted['total_make_units']:,.0f}",
            delta=f"{weighted['total_buy_units']:,.0f} to buy"
        )
    
    with col3:
        # Show buy cost - use baseline from parts even if 100% manufacturing
        baseline_buy_cost = sum(p.historical_avg_cogs * p.historical_units_sold for p in scenario.parts_data) / weighted['total_units'] if weighted['total_units'] > 0 else 0
        display_buy_cost = weighted['weighted_buy_cost'] if weighted['total_buy_units'] > 0 else baseline_buy_cost
        st.metric(
            "Avg Buy Cost",
            format_currency(display_buy_cost),
            help="Weighted average COGS (what you pay suppliers)"
        )
    
    with col4:
        if weighted['total_make_units'] > 0:
            st.metric(
                "Avg Mfg Cost",
                format_currency(weighted['weighted_make_cost']),
                help="Weighted average manufacturing cost"
            )
        else:
            st.metric("Avg Mfg Cost", "-")
    
    # Show savings summary if manufacturing
    if weighted['total_make_units'] > 0 and weighted['savings_pct'] > 0:
        st.markdown("---")
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            st.metric(
                "Cost Savings %",
                f"{weighted['savings_pct']:.1f}%",
                help="Manufacturing cost savings vs buying"
            )
        with scol2:
            st.metric(
                "Est. Annual Savings",
                format_currency(weighted['potential_savings']),
                help="Total annual savings from manufacturing"
            )
        with scol3:
            st.metric(
                "Blended Cost/Unit",
                format_currency(weighted['weighted_blended_cost']),
                help="Average cost across make/buy mix"
            )
    
    # Update scenario with part-level calculations
    if weighted['total_units'] > 0:
        scenario.hybrid_make_pct = weighted['overall_make_pct'] / 100
        
        # Keep baseline buy cost for comparison even if 100% manufacturing
        baseline_buy_cost = sum(p.historical_avg_cogs * p.historical_units_sold for p in scenario.parts_data) / weighted['total_units']
        scenario.buy_costs.purchase_price_per_unit = weighted['weighted_buy_cost'] if weighted['total_buy_units'] > 0 else baseline_buy_cost
        
        if weighted['total_make_units'] > 0:
            # Set manufacturing costs based on the weighted average
            # Breakdown: 60% raw materials, 25% direct labor, 10% variable overhead, 5% QC
            avg_make_cost = weighted['weighted_make_cost']
            scenario.make_costs.raw_material_per_unit = avg_make_cost * 0.60
            scenario.make_costs.direct_labor_per_unit = avg_make_cost * 0.25
            scenario.make_costs.variable_overhead_per_unit = avg_make_cost * 0.10
            scenario.make_costs.use_overhead_pct = False  # Use per-unit value from part-level
            scenario.make_costs.quality_control_pct = 0.05
            scenario.make_costs.wastage_pct = 0.03
            
            # Show sync notification
            st.success(f"""
            ✅ **Costs synced from part-level analysis:**
            - Buy Cost: {format_currency(scenario.buy_costs.purchase_price_per_unit)}/unit
            - Make Cost: {format_currency(avg_make_cost)}/unit
              - Raw Materials: {format_currency(scenario.make_costs.raw_material_per_unit)}
              - Direct Labor: {format_currency(scenario.make_costs.direct_labor_per_unit)}
              - Variable Overhead: {format_currency(scenario.make_costs.variable_overhead_per_unit)}
            """)
        
        scenario.annual_demand_units = int(weighted['total_units'])
        
        # Calculate selling price from parts
        total_revenue = sum(p.historical_avg_selling_price * p.historical_units_sold for p in scenario.parts_data)
        scenario.selling_price_per_unit = total_revenue / weighted['total_units'] if weighted['total_units'] > 0 else 0
    
    # Set strategy based on configuration
    if weighted['overall_make_pct'] >= 95:
        scenario.strategy = 'make'
    elif weighted['overall_make_pct'] <= 5:
        scenario.strategy = 'buy'
    else:
        scenario.strategy = 'hybrid'
    
    # CAPEX Section
    render_capex_section(scenario)


def render_capex_section(scenario: IntegrationScenario):
    """Render the CAPEX requirements section."""
    
    st.markdown("---")
    
    with st.expander("💰 CAPEX Requirements (Manufacturing)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            scenario.capex.equipment_cost = st.number_input(
                "Equipment Cost (R)",
                min_value=0.0,
                value=float(scenario.capex.equipment_cost),
                step=100000.0,
                format="%.0f",
                key="vi_capex_equip"
            )
            scenario.capex.facility_cost = st.number_input(
                "Facility Cost (R)",
                min_value=0.0,
                value=float(scenario.capex.facility_cost),
                step=100000.0,
                format="%.0f",
                key="vi_capex_facility"
            )
            scenario.capex.tooling_cost = st.number_input(
                "Tooling Cost (R)",
                min_value=0.0,
                value=float(scenario.capex.tooling_cost),
                step=50000.0,
                format="%.0f",
                key="vi_capex_tooling"
            )
        
        with col2:
            scenario.capex.installation_cost = st.number_input(
                "Installation Cost (R)",
                min_value=0.0,
                value=float(scenario.capex.installation_cost),
                step=50000.0,
                format="%.0f",
                key="vi_capex_install"
            )
            scenario.capex.working_capital = st.number_input(
                "Working Capital (R)",
                min_value=0.0,
                value=float(scenario.capex.working_capital),
                step=100000.0,
                format="%.0f",
                key="vi_capex_wc"
            )
            
            st.markdown("---")
            st.markdown(f"**Total CAPEX:** {format_currency(scenario.capex.total_capex)}")
            st.markdown(f"**Monthly Depreciation:** {format_currency(scenario.capex.monthly_depreciation)}")
    
    # Capacity Section
    with st.expander("🏭 Capacity Planning", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            scenario.capacity.monthly_capacity_units = st.number_input(
                "Monthly Capacity (units)",
                min_value=100,
                max_value=100000,
                value=scenario.capacity.monthly_capacity_units,
                step=100,
                key="vi_capacity_units"
            )
            scenario.capacity.ramp_up_months = st.slider(
                "Ramp-up Period (months)",
                min_value=1,
                max_value=24,
                value=scenario.capacity.ramp_up_months,
                key="vi_ramp_up"
            )
        
        with col2:
            scenario.capacity.initial_capacity_pct = st.slider(
                "Initial Capacity (%)",
                min_value=10,
                max_value=80,
                value=int(scenario.capacity.initial_capacity_pct * 100),
                key="vi_initial_capacity"
            ) / 100
    
    # NEW IN v3.0: Commissioning Schedule Section
    with st.expander("🚀 Commissioning Schedule (v3.0)", expanded=False):
        st.markdown("""
        Configure when manufacturing will start and the commissioning timeline.
        **Manufacturing impact will only begin after commissioning is complete.**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario.commissioning.start_month = st.number_input(
                "Commissioning Start Month",
                min_value=1,
                max_value=60,
                value=scenario.commissioning.start_month,
                help="Month number from forecast start when commissioning begins",
                key="vi_comm_start"
            )
        
        with col2:
            scenario.commissioning.duration_months = st.number_input(
                "Commissioning Duration (months)",
                min_value=1,
                max_value=24,
                value=scenario.commissioning.duration_months,
                help="How many months commissioning will take",
                key="vi_comm_duration"
            )
        
        st.info(f"📅 Manufacturing will be active from **Month {scenario.commissioning.completion_month}** onwards")
        
        st.markdown("---")
        st.markdown("**Monthly Commissioning Costs**")
        st.caption("Enter the expected costs for each month during commissioning")
        
        # Initialize monthly_costs if needed
        if len(scenario.commissioning.monthly_costs) != scenario.commissioning.duration_months:
            default_monthly = scenario.capex.installation_cost / max(scenario.commissioning.duration_months, 1)
            scenario.commissioning.monthly_costs = [default_monthly] * scenario.commissioning.duration_months
        
        # Create columns for cost inputs (max 4 per row)
        n_cols = min(4, scenario.commissioning.duration_months)
        if n_cols > 0:
            cols = st.columns(n_cols)
            
            new_costs = []
            for i in range(scenario.commissioning.duration_months):
                col_idx = i % n_cols
                with cols[col_idx]:
                    month_num = scenario.commissioning.start_month + i
                    current_value = scenario.commissioning.monthly_costs[i] if i < len(scenario.commissioning.monthly_costs) else 0
                    
                    cost = st.number_input(
                        f"Month {month_num}",
                        min_value=0.0,
                        value=float(current_value),
                        step=50000.0,
                        format="%.0f",
                        key=f"vi_comm_cost_{i}"
                    )
                    new_costs.append(cost)
            
            scenario.commissioning.monthly_costs = new_costs
        
        # Display totals and CAPEX validation
        total_comm_cost = scenario.commissioning.total_commissioning_cost
        capex_for_commissioning = scenario.capex.total_capex_for_commissioning
        working_capital = scenario.capex.working_capital
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Total Commissioning Cost:** {format_currency(total_comm_cost)}")
        with col2:
            st.markdown(f"**CAPEX (excl. WC):** {format_currency(capex_for_commissioning)}")
        with col3:
            st.markdown(f"**Working Capital:** {format_currency(working_capital)}")
        
        st.caption("💡 Working capital is separate from commissioning costs and flows into forecast working capital requirements")
        
        # Validation: Commissioning costs should equal CAPEX (excluding working capital)
        if capex_for_commissioning > 0:
            diff = abs(total_comm_cost - capex_for_commissioning)
            diff_pct = (diff / capex_for_commissioning) * 100 if capex_for_commissioning > 0 else 0
            
            if diff < 1:  # Allow for rounding (within R1)
                st.success("✅ Commissioning costs equal CAPEX (excluding working capital)")
            elif diff_pct <= 1:  # Within 1% tolerance
                st.warning(f"⚠️ Minor difference: {format_currency(diff)} ({diff_pct:.1f}%)")
            else:
                st.error(f"""
                ❌ **Mismatch:** Commissioning costs ({format_currency(total_comm_cost)}) 
                do not equal CAPEX excluding WC ({format_currency(capex_for_commissioning)}).
                
                **Difference:** {format_currency(diff)} ({diff_pct:.1f}%)
                
                Please adjust monthly costs to match your CAPEX (excluding working capital).
                """)
                
                # Offer auto-distribute button
                if st.button("📊 Auto-distribute CAPEX across months", key="vi_auto_distribute_capex"):
                    if scenario.commissioning.duration_months > 0:
                        monthly_amount = capex_for_commissioning / scenario.commissioning.duration_months
                        scenario.commissioning.monthly_costs = [monthly_amount] * scenario.commissioning.duration_months
                        st.rerun()
    
    # NEW IN v3.0: Working Capital Calculation Section
    with st.expander("💼 Working Capital Calculation (v3.0)", expanded=False):
        st.markdown("""
        Working capital is calculated based on raw material inventory days and manufacturing cycle time.
        **Labour is expensed when incurred** (make-to-order model).
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario.wc_params.raw_material_days = st.number_input(
                "Raw Material Inventory (days)",
                min_value=0,
                max_value=180,
                value=scenario.wc_params.raw_material_days,
                help="Days of raw material stock to hold",
                key="vi_wc_rm_days"
            )
            
            scenario.wc_params.manufacturing_days = st.number_input(
                "Manufacturing Cycle Time (days)",
                min_value=1,
                max_value=90,
                value=scenario.wc_params.manufacturing_days,
                help="Days from raw material to finished goods",
                key="vi_wc_mfg_days"
            )
        
        with col2:
            # Calculate monthly COGS for manufactured portion
            monthly_demand = scenario.annual_demand_units / 12
            mfg_pct = scenario.hybrid_make_pct if scenario.strategy == 'hybrid' else (1.0 if scenario.strategy == 'make' else 0.0)
            monthly_mfg_units = monthly_demand * mfg_pct
            monthly_mfg_cogs = monthly_mfg_units * scenario.make_costs.total_variable_cost_per_unit
            
            # Calculate WC
            calculated_wc = scenario.wc_params.calculate_working_capital(monthly_mfg_cogs, mfg_pct)
            
            st.markdown("**Calculated Working Capital:**")
            st.markdown(f"""
            | Component | Value |
            |-----------|-------|
            | Monthly Mfg Units | {monthly_mfg_units:,.0f} |
            | Monthly Mfg COGS | {format_currency(monthly_mfg_cogs)} |
            | Total Days | {scenario.wc_params.raw_material_days + scenario.wc_params.manufacturing_days} |
            | **Calculated WC** | **{format_currency(calculated_wc)}** |
            """)
            
            # Option to use calculated vs manual
            use_calculated = st.checkbox(
                "Use calculated WC instead of manual entry",
                value=False,
                key="vi_use_calc_wc",
                help="Override the manual working capital entry above"
            )
            
            if use_calculated:
                scenario.capex.working_capital = calculated_wc
                st.success(f"✅ Working capital set to calculated value: {format_currency(calculated_wc)}")


def render_cost_comparison_tab(scenario: IntegrationScenario):
    """Render the cost comparison tab."""
    
    section_header("Cost Comparison Analysis", "Compare Make vs Buy costs over time")
    
    # Calculate scenarios
    make_results = calculate_make_scenario(scenario, months=60)
    buy_results = calculate_buy_scenario(scenario, months=60)
    
    if scenario.strategy == 'hybrid':
        hybrid_results = calculate_hybrid_scenario(scenario, months=60)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        make_5yr_profit = make_results['cumulative_profit'][-1] if make_results['cumulative_profit'] else 0
        metric_card("Make - 5 Year Profit", format_currency(make_5yr_profit, 1))
    
    with col2:
        buy_5yr_profit = buy_results['cumulative_profit'][-1] if buy_results['cumulative_profit'] else 0
        metric_card("Buy - 5 Year Profit", format_currency(buy_5yr_profit, 1))
    
    with col3:
        advantage = make_5yr_profit - buy_5yr_profit
        advantage_label = "Make Advantage" if advantage > 0 else "Buy Advantage"
        metric_card(advantage_label, format_currency(abs(advantage), 1))
    
    with col4:
        if scenario.strategy == 'hybrid':
            hybrid_5yr_profit = hybrid_results['cumulative_profit'][-1] if hybrid_results['cumulative_profit'] else 0
            metric_card("Hybrid - 5 Year Profit", format_currency(hybrid_5yr_profit, 1))
        else:
            metric_card("Strategy", scenario.strategy.upper())
    
    st.markdown("---")
    
    # Cumulative Profit Chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=make_results['months'],
        y=make_results['cumulative_profit'],
        name='Make',
        line=dict(color=GREEN, width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=buy_results['months'],
        y=buy_results['cumulative_profit'],
        name='Buy',
        line=dict(color=BLUE, width=2)
    ))
    
    if scenario.strategy == 'hybrid':
        fig.add_trace(go.Scatter(
            x=hybrid_results['months'],
            y=hybrid_results['cumulative_profit'],
            name='Hybrid',
            line=dict(color=PURPLE, width=2, dash='dash')
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color=TEXT_MUTED)
    
    fig.update_layout(
        title="Cumulative Profit Comparison",
        xaxis_title="Month",
        yaxis_title="Cumulative Profit (R)",
        yaxis_tickformat=',.0f',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_WHITE),
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly P&L comparison
    st.markdown("### Monthly P&L Comparison (Steady State)")
    
    # Use month 24 as steady state (after ramp-up)
    steady_month = min(24, len(make_results['months']) - 1)
    
    comparison_data = {
        'Metric': ['Revenue', 'COGS', 'Gross Profit', 'Fixed Costs', 'Depreciation', 'EBIT', 'Margin %'],
        'Make': [
            make_results['revenue'][steady_month],
            make_results['total_cogs'][steady_month],
            make_results['gross_profit'][steady_month],
            make_results['fixed_costs'][steady_month],
            make_results['depreciation'][steady_month],
            make_results['ebit'][steady_month],
            make_results['ebit'][steady_month] / make_results['revenue'][steady_month] * 100 if make_results['revenue'][steady_month] > 0 else 0
        ],
        'Buy': [
            buy_results['revenue'][steady_month],
            buy_results['cogs'][steady_month],
            buy_results['gross_profit'][steady_month],
            0,
            0,
            buy_results['gross_profit'][steady_month],
            buy_results['gross_profit'][steady_month] / buy_results['revenue'][steady_month] * 100 if buy_results['revenue'][steady_month] > 0 else 0
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format for display
    for col in ['Make', 'Buy']:
        comparison_df[col] = comparison_df.apply(
            lambda row: f"{row[col]:.1f}%" if row['Metric'] == 'Margin %' else format_currency(row[col]),
            axis=1
        )
    
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)


def render_break_even_tab(scenario: IntegrationScenario):
    """Render break-even analysis tab."""
    
    section_header("Break-Even Analysis", "Volume and time to break-even")
    
    # Calculate break-even
    be_analysis = calculate_break_even_volume(scenario)
    
    # Display results
    if be_analysis['crossover_message']:
        alert_box(be_analysis['crossover_message'], "warning")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metric_card(
                "Break-Even Volume (Monthly)",
                f"{be_analysis['break_even_monthly']:,.0f} units" if be_analysis['break_even_monthly'] else "N/A"
            )
        
        with col2:
            metric_card(
                "Break-Even Volume (Annual)",
                f"{be_analysis['break_even_annual']:,.0f} units" if be_analysis['break_even_annual'] else "N/A"
            )
        
        with col3:
            current_demand = scenario.annual_demand_units
            if be_analysis['break_even_annual'] and be_analysis['break_even_annual'] > 0:
                capacity_util = current_demand / be_analysis['break_even_annual'] * 100
                metric_card(
                    "Demand vs Break-Even",
                    f"{capacity_util:.0f}%",
                    "Above" if capacity_util > 100 else "Below"
                )
    
    st.markdown("---")
    
    # Cost structure visualization
    st.markdown("### Cost Structure Comparison")
    
    volumes = list(range(100, int(scenario.annual_demand_units * 1.5), max(100, scenario.annual_demand_units // 20)))
    
    make_costs = []
    buy_costs = []
    
    for vol in volumes:
        # Make: Fixed + Variable
        monthly_vol = vol / 12
        make_total = (be_analysis['make_fixed_monthly'] * 12) + (be_analysis['make_variable_cost'] * vol)
        make_costs.append(make_total)
        
        # Buy: Variable only
        buy_total = be_analysis['buy_variable_cost'] * vol
        buy_costs.append(buy_total)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=volumes,
        y=make_costs,
        name='Make (Total Cost)',
        line=dict(color=GREEN, width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=volumes,
        y=buy_costs,
        name='Buy (Total Cost)',
        line=dict(color=BLUE, width=2)
    ))
    
    # Break-even point
    if be_analysis['break_even_annual']:
        be_cost = be_analysis['buy_variable_cost'] * be_analysis['break_even_annual']
        fig.add_trace(go.Scatter(
            x=[be_analysis['break_even_annual']],
            y=[be_cost],
            mode='markers',
            name='Break-Even Point',
            marker=dict(size=15, color=GOLD, symbol='star')
        ))
    
    # Current demand line
    fig.add_vline(
        x=scenario.annual_demand_units,
        line_dash="dash",
        line_color=TEXT_MUTED,
        annotation_text="Current Demand"
    )
    
    fig.update_layout(
        title="Annual Cost by Volume",
        xaxis_title="Annual Volume (units)",
        yaxis_title="Total Annual Cost (R)",
        yaxis_tickformat=',.0f',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_WHITE),
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost breakdown table
    st.markdown("### Cost Breakdown per Unit")
    
    breakdown_data = {
        'Component': [
            'Raw Materials',
            'Direct Labor',
            'Manufacturing Overhead',
            'Wastage',
            'Quality Control',
            'Total Variable (Make)',
            '',
            'Purchase Price',
            'Freight',
            'Import Duty',
            'Inspection',
            'Total Landed (Buy)'
        ],
        'Cost (R/unit)': [
            scenario.make_costs.raw_material_per_unit,
            scenario.make_costs.direct_labor_per_unit,
            scenario.make_costs.raw_material_per_unit * scenario.make_costs.manufacturing_overhead_pct,
            scenario.make_costs.raw_material_per_unit * scenario.make_costs.wastage_pct,
            scenario.make_costs.total_variable_cost_per_unit * scenario.make_costs.quality_control_pct,
            scenario.make_costs.total_variable_cost_per_unit,
            '',
            scenario.buy_costs.purchase_price_per_unit,
            scenario.buy_costs.freight_per_unit,
            (scenario.buy_costs.purchase_price_per_unit + scenario.buy_costs.freight_per_unit) * scenario.buy_costs.import_duty_pct,
            scenario.buy_costs.total_landed_cost_per_unit * scenario.buy_costs.quality_inspection_pct,
            scenario.buy_costs.total_landed_cost_per_unit
        ]
    }
    
    breakdown_df = pd.DataFrame(breakdown_data)
    breakdown_df['Cost (R/unit)'] = breakdown_df['Cost (R/unit)'].apply(
        lambda x: format_currency(x) if x != '' else ''
    )
    
    st.dataframe(breakdown_df, hide_index=True, use_container_width=True)


def render_npv_tab(scenario: IntegrationScenario):
    """Render NPV analysis tab."""
    
    section_header("NPV Analysis", "Net Present Value comparison with sensitivity")
    
    # Discount rate input
    discount_rate = st.slider(
        "Discount Rate (%)",
        min_value=5,
        max_value=25,
        value=12,
        key="vi_discount_rate"
    ) / 100
    
    # Calculate scenarios
    make_results = calculate_make_scenario(scenario, months=60)
    buy_results = calculate_buy_scenario(scenario, months=60)
    
    # Calculate monthly cash flows
    make_cash_flows = [-scenario.capex.total_capex]  # Initial investment
    for i in range(60):
        cf = make_results['ebit'][i] + make_results['depreciation'][i]
        make_cash_flows.append(cf)
    
    buy_cash_flows = [0]  # No initial investment
    for i in range(60):
        cf = buy_results['gross_profit'][i]
        buy_cash_flows.append(cf)
    
    # Calculate NPV and IRR
    make_npv = calculate_npv(make_cash_flows, discount_rate)
    buy_npv = calculate_npv(buy_cash_flows, discount_rate)
    make_irr = calculate_irr(make_cash_flows)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card("Make NPV", format_currency(make_npv, 1))
    
    with col2:
        metric_card("Buy NPV", format_currency(buy_npv, 1))
    
    with col3:
        npv_diff = make_npv - buy_npv
        metric_card(
            "NPV Advantage",
            format_currency(abs(npv_diff), 1),
            "Make" if npv_diff > 0 else "Buy"
        )
    
    with col4:
        if make_irr:
            metric_card("Make IRR", f"{make_irr*100:.1f}%")
        else:
            metric_card("Make IRR", "N/A")
    
    st.markdown("---")
    
    # Payback analysis
    st.markdown("### Payback Analysis")
    
    make_payback = calculate_payback_months(make_results['cumulative_cash'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if make_payback < 60:
            st.success(f"✅ Make Scenario Payback: **{make_payback} months** ({make_payback/12:.1f} years)")
        else:
            st.warning(f"⚠️ Make Scenario: Investment not recovered within 5 years")
    
    with col2:
        st.info(f"📊 CAPEX Investment: **{format_currency(scenario.capex.total_capex)}**")
    
    # Sensitivity Analysis
    st.markdown("---")
    st.markdown("### NPV Sensitivity Analysis")
    
    # Volume sensitivity
    volumes_pct = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    discount_rates = [0.08, 0.10, 0.12, 0.14, 0.16]
    
    sensitivity_matrix = []
    
    for vol_pct in volumes_pct:
        row = []
        for dr in discount_rates:
            # Recalculate with adjusted volume
            temp_scenario = IntegrationScenario(
                annual_demand_units=int(scenario.annual_demand_units * vol_pct),
                selling_price_per_unit=scenario.selling_price_per_unit,
                make_costs=scenario.make_costs,
                buy_costs=scenario.buy_costs,
                capex=scenario.capex,
                capacity=scenario.capacity,
                manufacturing_fixed_costs_monthly=scenario.manufacturing_fixed_costs_monthly
            )
            
            temp_make = calculate_make_scenario(temp_scenario, months=60)
            temp_buy = calculate_buy_scenario(temp_scenario, months=60)
            
            temp_make_cf = [-scenario.capex.total_capex]
            for i in range(60):
                cf = temp_make['ebit'][i] + temp_make['depreciation'][i]
                temp_make_cf.append(cf)
            
            temp_buy_cf = [0]
            for i in range(60):
                cf = temp_buy['gross_profit'][i]
                temp_buy_cf.append(cf)
            
            make_npv_temp = calculate_npv(temp_make_cf, dr)
            buy_npv_temp = calculate_npv(temp_buy_cf, dr)
            
            row.append(make_npv_temp - buy_npv_temp)
        
        sensitivity_matrix.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sensitivity_matrix,
        x=[f"{dr*100:.0f}%" for dr in discount_rates],
        y=[f"{vol*100:.0f}%" for vol in volumes_pct],
        colorscale=[
            [0, RED],
            [0.5, '#FFFF99'],
            [1, GREEN]
        ],
        zmid=0,
        text=[[format_currency(val, 1) for val in row] for row in sensitivity_matrix],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="Volume: %{y}<br>Discount Rate: %{x}<br>NPV Advantage: %{z:,.0f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Make vs Buy NPV Advantage (Green = Make Better, Red = Buy Better)",
        xaxis_title="Discount Rate",
        yaxis_title="Volume (% of Base)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_WHITE)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_summary_tab(scenario: IntegrationScenario):
    """Render summary and recommendation tab."""
    
    section_header("Analysis Summary & Recommendation", "Key findings and suggested strategy")
    
    # Calculate all scenarios
    make_results = calculate_make_scenario(scenario, months=60)
    buy_results = calculate_buy_scenario(scenario, months=60)
    be_analysis = calculate_break_even_volume(scenario)
    
    make_5yr = make_results['cumulative_profit'][-1]
    buy_5yr = buy_results['cumulative_profit'][-1]
    
    # Decision matrix
    st.markdown("### 📊 Decision Matrix")
    
    criteria = []
    
    # Profitability
    if make_5yr > buy_5yr:
        criteria.append(("5-Year Profitability", "Make", f"+{format_currency(make_5yr - buy_5yr)}"))
    else:
        criteria.append(("5-Year Profitability", "Buy", f"+{format_currency(buy_5yr - make_5yr)}"))
    
    # Break-even
    if be_analysis['break_even_annual'] and scenario.annual_demand_units > be_analysis['break_even_annual']:
        criteria.append(("Break-Even Test", "Make", f"Demand > BE by {(scenario.annual_demand_units/be_analysis['break_even_annual']-1)*100:.0f}%"))
    else:
        criteria.append(("Break-Even Test", "Buy", "Demand below break-even"))
    
    # Variable cost
    buy_cost = scenario.buy_costs.total_landed_cost_per_unit
    make_cost = scenario.make_costs.total_variable_cost_per_unit
    
    if buy_cost > 0 and make_cost < buy_cost:
        savings = (buy_cost - make_cost) / buy_cost * 100
        criteria.append(("Unit Cost", "Make", f"{savings:.1f}% lower"))
    elif buy_cost > 0 and make_cost > buy_cost:
        premium = (make_cost - buy_cost) / buy_cost * 100
        criteria.append(("Unit Cost", "Buy", f"{premium:.1f}% lower"))
    elif buy_cost == 0 and make_cost > 0:
        criteria.append(("Unit Cost", "Make", "100% manufacturing - no buy comparison"))
    else:
        criteria.append(("Unit Cost", "Neutral", "Cost data incomplete"))
    
    # Capital requirement
    if scenario.capex.total_capex > 0:
        criteria.append(("Capital Requirement", "Buy", f"No CAPEX vs {format_currency(scenario.capex.total_capex)}"))
    else:
        criteria.append(("Capital Requirement", "Neutral", "No significant CAPEX"))
    
    # Risk
    criteria.append(("Operational Risk", "Buy", "Lower complexity"))
    criteria.append(("Supply Chain Risk", "Make", "Better control"))
    
    # Count
    make_wins = sum(1 for c in criteria if c[1] == 'Make')
    buy_wins = sum(1 for c in criteria if c[1] == 'Buy')
    
    # Display criteria
    criteria_df = pd.DataFrame(criteria, columns=['Criterion', 'Favors', 'Reason'])
    
    def highlight_favor(val):
        if val == 'Make':
            return f'background-color: {GREEN}22; color: {GREEN}'
        elif val == 'Buy':
            return f'background-color: {BLUE}22; color: {BLUE}'
        return ''
    
    st.dataframe(
        criteria_df.style.applymap(highlight_favor, subset=['Favors']),
        hide_index=True,
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Recommendation
    st.markdown("### 💡 Recommendation")
    
    if make_wins > buy_wins + 1 and make_5yr > buy_5yr:
        recommendation = "MAKE"
        rec_color = GREEN
        
        # Calculate unit cost advantage safely
        if buy_cost > 0:
            unit_cost_advantage = (buy_cost - make_cost) / buy_cost * 100
            unit_cost_text = f"- Unit cost advantage of {unit_cost_advantage:.1f}%"
        else:
            unit_cost_text = "- Full manufacturing cost control"
        
        rec_text = f"""
        Based on the analysis, **Manufacturing In-House (Make)** is the recommended strategy.
        
        **Key Reasons:**
        - Higher 5-year profitability by {format_currency(make_5yr - buy_5yr)}
        {unit_cost_text}
        - Better supply chain control
        
        **Considerations:**
        - Requires CAPEX investment of {format_currency(scenario.capex.total_capex)}
        - Payback period of {calculate_payback_months(make_results['cumulative_cash'])} months
        - Increased operational complexity
        """
    elif buy_wins > make_wins + 1 or (be_analysis['break_even_annual'] and scenario.annual_demand_units < be_analysis['break_even_annual']):
        recommendation = "BUY"
        rec_color = BLUE
        rec_text = f"""
        Based on the analysis, **Outsourcing (Buy)** is the recommended strategy.
        
        **Key Reasons:**
        - No capital investment required
        - Lower operational risk and complexity
        - Current demand {'below' if be_analysis['break_even_annual'] and scenario.annual_demand_units < be_analysis['break_even_annual'] else 'near'} break-even point
        
        **Considerations:**
        - Higher unit costs in the long term
        - Less control over supply chain
        - Dependent on supplier reliability
        """
    else:
        recommendation = "HYBRID"
        rec_color = PURPLE
        rec_text = f"""
        Based on the analysis, a **Hybrid Strategy** is recommended.
        
        **Key Reasons:**
        - Balanced risk between make and buy
        - Allows gradual transition to manufacturing
        - Maintains supplier relationships as backup
        
        **Suggested Split:**
        - Manufacture: {scenario.hybrid_make_pct * 100:.0f}% of volume (high-margin items)
        - Outsource: {(1 - scenario.hybrid_make_pct) * 100:.0f}% of volume (low-margin or specialty items)
        
        **Considerations:**
        - Requires scaled CAPEX of {format_currency(scenario.capex.total_capex * scenario.hybrid_make_pct)}
        - More complex operations management
        """
    
    st.markdown(f"""
    <div style="
        background: {rec_color}22;
        border: 2px solid {rec_color};
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    ">
        <h2 style="color: {rec_color}; margin: 0 0 1rem 0;">
            Recommended Strategy: {recommendation}
        </h2>
        <div style="color: {TEXT_WHITE};">
            {rec_text.replace(chr(10), '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Export report
    st.markdown("---")
    
    if st.button("📥 Export Full Analysis Report", type="primary", use_container_width=True, key="vi_export_report"):
        # Generate summary report
        report_data = {
            'Parameter': [
                'Scenario Name', 'Strategy', 'Annual Demand', 'Selling Price/Unit',
                'Make Variable Cost/Unit', 'Buy Landed Cost/Unit',
                'Total CAPEX', 'Monthly Fixed Costs',
                '5-Year Make Profit', '5-Year Buy Profit', 'Recommendation'
            ],
            'Value': [
                scenario.name,
                scenario.strategy.upper(),
                f"{scenario.annual_demand_units:,} units",
                format_currency(scenario.selling_price_per_unit),
                format_currency(scenario.make_costs.total_variable_cost_per_unit),
                format_currency(scenario.buy_costs.total_landed_cost_per_unit),
                format_currency(scenario.capex.total_capex),
                format_currency(scenario.manufacturing_fixed_costs_monthly),
                format_currency(make_5yr),
                format_currency(buy_5yr),
                recommendation
            ]
        }
        
        report_df = pd.DataFrame(report_data)
        csv_report = report_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Report CSV",
            data=csv_report,
            file_name=f"make_vs_buy_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


# =============================================================================
# NEW IN V3.0: FORECAST INTEGRATION API
# =============================================================================

def get_manufacturing_impact_for_forecast(
    scenario: IntegrationScenario, 
    month_index: int,
    base_cogs: float
) -> Dict[str, float]:
    """
    NEW IN v3.0: Get manufacturing impact for a specific forecast month.
    UPDATED v3.2: Separate variable overhead from direct COGS.
    
    Use this function from forecast_section.py to integrate manufacturing
    strategy into the income statement.
    
    Args:
        scenario: IntegrationScenario with manufacturing configuration
        month_index: Month index from forecast start (0-based)
        base_cogs: Base COGS for this month (from forecast engine)
    
    Returns dict with:
        - buy_cogs: COGS for bought portion
        - mfg_cogs: Direct manufacturing COGS (materials + labor)
        - mfg_variable_overhead: Variable manufacturing overhead
        - mfg_fixed_overhead: Fixed manufacturing overhead (monthly)
        - mfg_depreciation: Manufacturing depreciation
        - commissioning_cost: Commissioning cost this month
        - total_cogs: Total COGS including all manufacturing impact
        - manufacturing_active: Whether manufacturing is active this month
    """
    result = {
        'buy_cogs': base_cogs,
        'mfg_cogs': 0.0,
        'mfg_variable_overhead': 0.0,
        'mfg_fixed_overhead': 0.0,
        'mfg_overhead': 0.0,  # Combined for backward compatibility
        'mfg_depreciation': 0.0,
        'commissioning_cost': 0.0,
        'total_cogs': base_cogs,
        'manufacturing_active': False
    }
    
    if scenario.strategy == 'buy':
        return result
    
    # Check if manufacturing is active (commissioning complete)
    mfg_active = scenario.commissioning.is_manufacturing_active(month_index)
    result['manufacturing_active'] = mfg_active
    
    # Commissioning cost for this month
    result['commissioning_cost'] = scenario.commissioning.get_cost_at_month(month_index)
    
    if mfg_active:
        # Determine manufacturing percentage
        mfg_pct = scenario.hybrid_make_pct if scenario.strategy == 'hybrid' else 1.0
        
        # Split COGS between buy and make
        result['buy_cogs'] = base_cogs * (1 - mfg_pct)
        
        # Get cost breakdown from make_costs
        cost_breakdown = scenario.make_costs.cost_breakdown
        total_var_cost = scenario.make_costs.total_variable_cost_per_unit
        
        # Calculate ratio vs buy cost to scale base_cogs appropriately
        if scenario.buy_costs.total_landed_cost_per_unit > 0 and total_var_cost > 0:
            # Scale base_cogs by the make/buy ratio
            mfg_portion_of_base_cogs = base_cogs * mfg_pct
            
            # Direct costs ratio (materials + labor) vs total
            direct_ratio = (cost_breakdown['raw_material'] + cost_breakdown['direct_labor']) / total_var_cost if total_var_cost > 0 else 0.7
            variable_overhead_ratio = cost_breakdown['variable_overhead'] / total_var_cost if total_var_cost > 0 else 0.15
            
            # Calculate each component
            make_ratio = total_var_cost / scenario.buy_costs.total_landed_cost_per_unit
            total_mfg_cogs = mfg_portion_of_base_cogs * make_ratio
            
            result['mfg_cogs'] = total_mfg_cogs * direct_ratio  # Direct costs only
            result['mfg_variable_overhead'] = total_mfg_cogs * variable_overhead_ratio
        else:
            # Fallback: assume 75% of buy cost, 70% direct / 15% variable overhead
            result['mfg_cogs'] = base_cogs * mfg_pct * 0.75 * 0.70
            result['mfg_variable_overhead'] = base_cogs * mfg_pct * 0.75 * 0.15
        
        # Fixed manufacturing overhead and depreciation (monthly amounts)
        result['mfg_fixed_overhead'] = scenario.manufacturing_fixed_costs_monthly * mfg_pct
        result['mfg_depreciation'] = scenario.capex.monthly_depreciation * mfg_pct
        
        # Combined overhead for backward compatibility
        result['mfg_overhead'] = result['mfg_variable_overhead'] + result['mfg_fixed_overhead']
        
        # Total COGS
        result['total_cogs'] = (result['buy_cogs'] + result['mfg_cogs'] + 
                               result['mfg_variable_overhead'] + result['mfg_fixed_overhead'] + 
                               result['mfg_depreciation'])
    
    return result


def get_manufacturing_assumptions(scenario: IntegrationScenario) -> Dict[str, Any]:
    """
    NEW IN v3.0: Get manufacturing assumptions for AI assumptions engine.
    
    Returns a dictionary of manufacturing-related assumptions that can be
    displayed in the AI Assumptions section.
    """
    return {
        'strategy': scenario.strategy,
        'hybrid_make_pct': scenario.hybrid_make_pct * 100 if scenario.strategy == 'hybrid' else (100 if scenario.strategy == 'make' else 0),
        'commissioning_start_month': scenario.commissioning.start_month,
        'commissioning_duration': scenario.commissioning.duration_months,
        'commissioning_completion_month': scenario.commissioning.completion_month,
        'total_commissioning_cost': scenario.commissioning.total_commissioning_cost,
        'make_cost_per_unit': scenario.make_costs.total_variable_cost_per_unit,
        'buy_cost_per_unit': scenario.buy_costs.total_landed_cost_per_unit,
        'monthly_fixed_costs': scenario.manufacturing_fixed_costs_monthly,
        'total_capex': scenario.capex.total_capex,
        'monthly_depreciation': scenario.capex.monthly_depreciation,
        'raw_material_days': scenario.wc_params.raw_material_days,
        'manufacturing_days': scenario.wc_params.manufacturing_days,
    }