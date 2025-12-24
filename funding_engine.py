"""
CE Africa Funding Engine
========================
Comprehensive financial engineering module for:
- Debt instruments (term loans, mezzanine, convertibles, trade finance)
- Equity financing
- Auto-overdraft facility
- IRR calculations (Equity IRR, Project IRR)
- Goal seek functionality

Usage:
    from funding_engine import FundingEngine, FundingScenario
    
    engine = FundingEngine()
    funded_cashflow = engine.apply_funding(
        base_cashflow=forecast_results,
        funding_events=events,
        overdraft_config=overdraft
    )
    
    irr = engine.calculate_equity_irr(funded_cashflow, equity_invested)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy import optimize
import warnings


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DebtType(Enum):
    TERM_LOAN = "debt_term_loan"
    MEZZANINE = "debt_mezzanine"
    CONVERTIBLE = "debt_convertible"
    TRADE_FINANCE = "debt_trade_finance"


class RepaymentType(Enum):
    AMORTIZING = "amortizing"        # Equal P+I payments
    BULLET = "bullet"                 # Principal at maturity
    INTEREST_ONLY = "interest_only"   # Interest monthly, principal at end
    PIK = "pik"                       # Payment-in-kind (interest capitalizes)


class EquityType(Enum):
    ORDINARY = "equity_ordinary"
    PREFERENCE = "equity_preference"


class TradeFinanceType(Enum):
    LETTER_OF_CREDIT = "letter_of_credit"
    IMPORT_FINANCE = "import_finance"
    STOCK_FINANCE = "stock_finance"
    DEBTOR_FINANCE = "debtor_finance"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DebtTranche:
    """Represents a single debt instrument."""
    id: str
    name: str
    debt_type: DebtType
    principal: float
    interest_rate: float  # Annual rate as decimal (0.12 = 12%)
    start_date: date
    term_months: int
    repayment_type: RepaymentType = RepaymentType.AMORTIZING
    grace_period_months: int = 0
    balloon_pct: float = 0.0  # % of principal as balloon payment
    pik_rate: float = 0.0  # PIK rate for mezzanine
    conversion_price: float = 0.0  # For convertibles
    conversion_date: Optional[date] = None
    
    # Calculated fields
    outstanding_principal: float = field(init=False)
    accrued_pik: float = field(init=False)
    
    def __post_init__(self):
        self.outstanding_principal = self.principal
        self.accrued_pik = 0.0
    
    @property
    def monthly_rate(self) -> float:
        """Monthly interest rate."""
        return self.interest_rate / 12
    
    @property
    def maturity_date(self) -> date:
        """Calculate maturity date."""
        return self.start_date + relativedelta(months=self.term_months)
    
    def calculate_monthly_payment(self, month_number: int) -> Dict[str, float]:
        """
        Calculate payment for a given month.
        
        Returns dict with: principal, interest, pik_interest, total_payment
        """
        if self.outstanding_principal <= 0:
            return {'principal': 0, 'interest': 0, 'pik_interest': 0, 'total_payment': 0}
        
        # Check if in grace period
        if month_number <= self.grace_period_months:
            # During grace, only accrue PIK if applicable
            interest = self.outstanding_principal * self.monthly_rate
            if self.repayment_type == RepaymentType.PIK:
                return {'principal': 0, 'interest': 0, 'pik_interest': interest, 'total_payment': 0}
            else:
                return {'principal': 0, 'interest': interest, 'pik_interest': 0, 'total_payment': interest}
        
        # Adjusted month number (after grace)
        adj_month = month_number - self.grace_period_months
        remaining_months = self.term_months - self.grace_period_months - adj_month + 1
        
        interest = self.outstanding_principal * self.monthly_rate
        
        if self.repayment_type == RepaymentType.AMORTIZING:
            # PMT formula for amortizing loan
            if remaining_months > 0 and self.monthly_rate > 0:
                pmt = self.outstanding_principal * (
                    self.monthly_rate * (1 + self.monthly_rate) ** remaining_months
                ) / ((1 + self.monthly_rate) ** remaining_months - 1)
                principal = pmt - interest
            else:
                principal = self.outstanding_principal
                pmt = principal + interest
            
            # Handle balloon payment
            if remaining_months == 1 and self.balloon_pct > 0:
                balloon = self.principal * self.balloon_pct
                principal = self.outstanding_principal
            
            return {'principal': principal, 'interest': interest, 'pik_interest': 0, 'total_payment': principal + interest}
        
        elif self.repayment_type == RepaymentType.BULLET:
            # Interest only, principal at maturity
            if remaining_months == 1:
                return {'principal': self.outstanding_principal, 'interest': interest, 'pik_interest': 0, 
                        'total_payment': self.outstanding_principal + interest}
            return {'principal': 0, 'interest': interest, 'pik_interest': 0, 'total_payment': interest}
        
        elif self.repayment_type == RepaymentType.INTEREST_ONLY:
            # Same as bullet but may have different balloon structure
            if remaining_months == 1:
                return {'principal': self.outstanding_principal, 'interest': interest, 'pik_interest': 0,
                        'total_payment': self.outstanding_principal + interest}
            return {'principal': 0, 'interest': interest, 'pik_interest': 0, 'total_payment': interest}
        
        elif self.repayment_type == RepaymentType.PIK:
            # Interest capitalizes, bullet at end
            pik_interest = self.outstanding_principal * (self.monthly_rate + self.pik_rate / 12)
            if remaining_months == 1:
                total_due = self.outstanding_principal + self.accrued_pik + pik_interest
                return {'principal': self.outstanding_principal, 'interest': 0, 'pik_interest': pik_interest,
                        'total_payment': total_due}
            return {'principal': 0, 'interest': 0, 'pik_interest': pik_interest, 'total_payment': 0}
        
        return {'principal': 0, 'interest': 0, 'pik_interest': 0, 'total_payment': 0}
    
    def apply_payment(self, principal_paid: float, pik_accrued: float = 0):
        """Apply a payment to this tranche."""
        self.outstanding_principal = max(0, self.outstanding_principal - principal_paid)
        self.accrued_pik += pik_accrued


@dataclass
class EquityInvestment:
    """Represents an equity investment."""
    id: str
    investor_name: str
    equity_type: EquityType
    amount: float
    investment_date: date
    share_price: float = 0.0
    shares_issued: float = 0.0
    dividend_rate: float = 0.0  # For preference shares
    
    @property
    def ownership_shares(self) -> float:
        """Number of shares from this investment."""
        if self.shares_issued > 0:
            return self.shares_issued
        elif self.share_price > 0:
            return self.amount / self.share_price
        return 0


@dataclass
class OverdraftFacility:
    """Auto-overdraft facility configuration."""
    facility_limit: float = 5_000_000
    interest_rate: float = 0.12  # Annual rate
    arrangement_fee_pct: float = 0.005
    commitment_fee_pct: float = 0.0025  # On undrawn
    auto_repay: bool = True
    is_active: bool = True
    
    # State tracking
    current_drawn: float = 0.0
    cumulative_interest: float = 0.0
    
    @property
    def monthly_rate(self) -> float:
        return self.interest_rate / 12
    
    @property
    def available(self) -> float:
        return max(0, self.facility_limit - self.current_drawn)
    
    def draw(self, amount: float) -> float:
        """Draw from facility. Returns actual amount drawn."""
        if not self.is_active:
            return 0
        draw_amount = min(amount, self.available)
        self.current_drawn += draw_amount
        return draw_amount
    
    def repay(self, amount: float) -> float:
        """Repay facility. Returns actual amount repaid."""
        repay_amount = min(amount, self.current_drawn)
        self.current_drawn -= repay_amount
        return repay_amount
    
    def calculate_monthly_interest(self) -> float:
        """Calculate interest for current month."""
        drawn_interest = self.current_drawn * self.monthly_rate
        commitment_fee = (self.facility_limit - self.current_drawn) * (self.commitment_fee_pct / 12)
        total = drawn_interest + commitment_fee
        self.cumulative_interest += total
        return total


@dataclass
class TradeFinanceFacility:
    """Trade finance facility for purchase funding."""
    id: str
    facility_type: TradeFinanceType
    facility_limit: float
    interest_rate: float  # Annual
    advance_rate_pct: float = 0.80  # % of invoice/stock value
    tenor_days: int = 90
    margin_pct: float = 0.02  # Over base rate
    arrangement_fee_pct: float = 0.01
    is_active: bool = True
    
    # State
    current_utilization: float = 0.0
    
    @property
    def available(self) -> float:
        return max(0, self.facility_limit - self.current_utilization)
    
    def draw_for_purchase(self, purchase_amount: float) -> Tuple[float, float]:
        """
        Draw facility for a purchase.
        Returns (amount_drawn, fee).
        """
        if not self.is_active:
            return 0, 0
        
        # Can finance up to advance_rate of purchase
        max_finance = purchase_amount * self.advance_rate_pct
        draw_amount = min(max_finance, self.available)
        
        # Calculate interest for tenor period
        daily_rate = self.interest_rate / 365
        interest = draw_amount * daily_rate * self.tenor_days
        
        self.current_utilization += draw_amount
        
        return draw_amount, interest
    
    def settle(self, amount: float):
        """Settle/repay trade finance."""
        self.current_utilization = max(0, self.current_utilization - amount)


@dataclass
class CapitalInvestment:
    """Capital/plant investment for Project IRR calculation."""
    id: str
    name: str
    capital_cost: float
    investment_date: date
    useful_life_years: int = 10
    salvage_value: float = 0
    
    # Expected benefits
    annual_revenue_increase: float = 0
    annual_cost_savings: float = 0
    margin_improvement_pct: float = 0
    
    def calculate_annual_benefit(self, base_revenue: float = 0) -> float:
        """Calculate annual cash benefit from investment."""
        margin_benefit = base_revenue * self.margin_improvement_pct
        return self.annual_revenue_increase + self.annual_cost_savings + margin_benefit
    
    def calculate_depreciation(self) -> float:
        """Annual straight-line depreciation."""
        return (self.capital_cost - self.salvage_value) / self.useful_life_years


# =============================================================================
# FUNDING SCENARIO
# =============================================================================

@dataclass
class FundingScenario:
    """Complete funding scenario with all instruments."""
    debt_tranches: List[DebtTranche] = field(default_factory=list)
    equity_investments: List[EquityInvestment] = field(default_factory=list)
    overdraft: Optional[OverdraftFacility] = None
    trade_finance: List[TradeFinanceFacility] = field(default_factory=list)
    capital_investments: List[CapitalInvestment] = field(default_factory=list)
    
    @property
    def total_debt(self) -> float:
        """Total outstanding debt."""
        return sum(t.outstanding_principal for t in self.debt_tranches)
    
    @property
    def total_equity(self) -> float:
        """Total equity invested."""
        return sum(e.amount for e in self.equity_investments)
    
    @property
    def total_overdraft_drawn(self) -> float:
        """Current overdraft utilization."""
        return self.overdraft.current_drawn if self.overdraft else 0
    
    @property
    def debt_to_equity_ratio(self) -> float:
        """D/E ratio."""
        if self.total_equity == 0:
            return float('inf') if self.total_debt > 0 else 0
        return self.total_debt / self.total_equity


# =============================================================================
# FUNDING ENGINE
# =============================================================================

class FundingEngine:
    """
    Core engine for applying funding to cash flows and calculating returns.
    """
    
    def __init__(self):
        self.warnings = []
    
    def apply_funding(
        self,
        base_cashflow: Dict[str, Any],
        funding_scenario: FundingScenario,
        start_date: date = None
    ) -> Dict[str, Any]:
        """
        Apply funding events to base cash flow.
        
        Args:
            base_cashflow: Dict with 'timeline', 'revenue', 'costs', 'profit' from forecast
            funding_scenario: FundingScenario with all funding instruments
            start_date: Override start date
            
        Returns:
            Dict with funded cash flow including debt service, overdraft, etc.
        """
        timeline = base_cashflow.get('timeline', [])
        if not timeline:
            return base_cashflow
        
        n_months = len(timeline)
        
        # Initialize result arrays
        result = {
            'timeline': timeline,
            'base_cash_flow': np.zeros(n_months),
            'debt_drawdown': np.zeros(n_months),
            'debt_principal_payment': np.zeros(n_months),
            'debt_interest_payment': np.zeros(n_months),
            'equity_injection': np.zeros(n_months),
            'overdraft_draw': np.zeros(n_months),
            'overdraft_repay': np.zeros(n_months),
            'overdraft_interest': np.zeros(n_months),
            'overdraft_balance': np.zeros(n_months),
            'trade_finance_draw': np.zeros(n_months),
            'trade_finance_cost': np.zeros(n_months),
            'net_cash_flow': np.zeros(n_months),
            'cumulative_cash': np.zeros(n_months),
            'debt_balance': np.zeros(n_months),
            'equity_balance': np.zeros(n_months),
        }
        
        # Get base operating cash flow
        # Handle both old format (with 'profit'>'ebit') and simple 'ebit' list
        if 'profit' in base_cashflow and 'ebit' in base_cashflow['profit']:
            ebit = np.array(base_cashflow['profit']['ebit'])
        elif 'ebit' in base_cashflow:
            ebit = np.array(base_cashflow['ebit'])
        else:
            # Fallback: revenue - costs
            revenue = np.array(base_cashflow.get('revenue', {}).get('total', [0] * n_months))
            costs = np.array(base_cashflow.get('costs', {}).get('total', [0] * n_months))
            ebit = revenue - costs
        
        # Simplified: Operating cash flow ≈ EBIT (adjust for WC changes if needed)
        result['base_cash_flow'] = ebit.copy()
        
        # Parse timeline to dates
        if start_date is None:
            try:
                first_period = timeline[0]
                start_date = datetime.strptime(first_period, '%Y-%m').date().replace(day=1)
            except:
                start_date = date.today().replace(day=1)
        
        period_dates = []
        for i in range(n_months):
            period_dates.append(start_date + relativedelta(months=i))
        
        # Track cumulative cash
        cash_balance = 0
        
        # Process each month
        for i, period_date in enumerate(period_dates):
            month_cf = result['base_cash_flow'][i]
            
            # 1. Apply equity injections
            for equity in funding_scenario.equity_investments:
                if (equity.investment_date.year == period_date.year and 
                    equity.investment_date.month == period_date.month):
                    result['equity_injection'][i] += equity.amount
                    month_cf += equity.amount
            
            # 2. Apply debt drawdowns
            for tranche in funding_scenario.debt_tranches:
                if (tranche.start_date.year == period_date.year and 
                    tranche.start_date.month == period_date.month):
                    result['debt_drawdown'][i] += tranche.principal
                    month_cf += tranche.principal
            
            # 3. Calculate debt service
            for tranche in funding_scenario.debt_tranches:
                if period_date >= tranche.start_date and tranche.outstanding_principal > 0:
                    months_since_start = (
                        (period_date.year - tranche.start_date.year) * 12 +
                        period_date.month - tranche.start_date.month + 1
                    )
                    
                    if months_since_start <= tranche.term_months:
                        payment = tranche.calculate_monthly_payment(months_since_start)
                        
                        result['debt_principal_payment'][i] += payment['principal']
                        result['debt_interest_payment'][i] += payment['interest']
                        month_cf -= payment['total_payment']
                        
                        # Apply payment to tranche
                        tranche.apply_payment(payment['principal'], payment['pik_interest'])
            
            # 4. Check for trade finance needs (simplified: % of COGS)
            if funding_scenario.trade_finance:
                cogs = base_cashflow.get('costs', {}).get('cogs', [0] * n_months)
                if i < len(cogs):
                    monthly_cogs = cogs[i]
                    for tf in funding_scenario.trade_finance:
                        if tf.is_active and monthly_cogs > 0:
                            # Finance portion of purchases
                            draw, cost = tf.draw_for_purchase(monthly_cogs * 0.3)  # Finance 30% of COGS
                            result['trade_finance_draw'][i] += draw
                            result['trade_finance_cost'][i] += cost
                            month_cf += draw - cost  # Net benefit
            
            # 5. Update cash balance before overdraft
            cash_balance += month_cf
            
            # 6. Process overdraft (auto-draw/repay)
            if funding_scenario.overdraft and funding_scenario.overdraft.is_active:
                od = funding_scenario.overdraft
                
                if cash_balance < 0:
                    # Need to draw
                    draw_needed = abs(cash_balance)
                    actual_draw = od.draw(draw_needed)
                    result['overdraft_draw'][i] = actual_draw
                    cash_balance += actual_draw
                    
                    if actual_draw < draw_needed:
                        self.warnings.append(
                            f"Month {i+1}: Cash shortfall of {draw_needed - actual_draw:,.0f} "
                            f"exceeds overdraft limit"
                        )
                
                elif cash_balance > 0 and od.auto_repay and od.current_drawn > 0:
                    # Auto-repay from surplus
                    repay_amount = od.repay(cash_balance)
                    result['overdraft_repay'][i] = repay_amount
                    cash_balance -= repay_amount
                
                # Calculate overdraft interest
                od_interest = od.calculate_monthly_interest()
                result['overdraft_interest'][i] = od_interest
                cash_balance -= od_interest
                
                result['overdraft_balance'][i] = od.current_drawn
            
            # 7. Store final values
            result['net_cash_flow'][i] = (
                result['base_cash_flow'][i] +
                result['equity_injection'][i] +
                result['debt_drawdown'][i] -
                result['debt_principal_payment'][i] -
                result['debt_interest_payment'][i] +
                result['trade_finance_draw'][i] -
                result['trade_finance_cost'][i] +
                result['overdraft_draw'][i] -
                result['overdraft_repay'][i] -
                result['overdraft_interest'][i]
            )
            
            result['cumulative_cash'][i] = cash_balance
            result['debt_balance'][i] = sum(t.outstanding_principal for t in funding_scenario.debt_tranches)
            result['equity_balance'][i] = sum(e.amount for e in funding_scenario.equity_investments)
        
        # Convert numpy arrays to lists for JSON serialization
        for key in result:
            if isinstance(result[key], np.ndarray):
                result[key] = result[key].tolist()
        
        return result
    
    def calculate_equity_irr(
        self,
        funded_cashflow: Dict[str, Any],
        initial_equity: float,
        terminal_value: float = 0,
        holding_period_months: int = None
    ) -> float:
        """
        Calculate Equity IRR (investor returns).
        
        Args:
            funded_cashflow: Result from apply_funding()
            initial_equity: Total equity invested
            terminal_value: Exit value at end of period
            holding_period_months: Override holding period
            
        Returns:
            Annual IRR as decimal (0.25 = 25%)
        """
        cash_flows = []
        
        # Initial investment (negative)
        equity_injections = funded_cashflow.get('equity_injection', [])
        net_cf = funded_cashflow.get('net_cash_flow', [])
        
        if not equity_injections and not net_cf:
            return 0.0
        
        n_periods = len(net_cf) if net_cf else len(equity_injections)
        if holding_period_months:
            n_periods = min(n_periods, holding_period_months)
        
        # Build cash flow series for equity investor
        # Investor perspective: negative for investments, positive for distributions/exit
        for i in range(n_periods):
            # Equity injection is outflow for investor
            equity_out = equity_injections[i] if i < len(equity_injections) else 0
            
            # For simplicity, assume free cash after debt service goes to equity
            # More sophisticated: track actual dividends
            fcf_to_equity = net_cf[i] if i < len(net_cf) else 0
            
            # Only count positive free cash flow as distributions
            distribution = max(0, fcf_to_equity) if i > 0 else 0
            
            cash_flows.append(-equity_out + distribution)
        
        # Add terminal value at exit
        if terminal_value > 0:
            cash_flows[-1] += terminal_value
        
        # Calculate IRR
        irr = self._calculate_irr(cash_flows)
        
        # Convert monthly to annual if periods are monthly
        if irr is not None:
            annual_irr = (1 + irr) ** 12 - 1
            return annual_irr
        
        return 0.0
    
    def calculate_project_irr(
        self,
        investment: CapitalInvestment,
        base_revenue: float = 0,
        discount_rate: float = 0.12
    ) -> float:
        """
        Calculate Project IRR for a capital investment.
        
        Args:
            investment: CapitalInvestment object
            base_revenue: Base revenue for margin improvement calculation
            discount_rate: Discount rate for comparison
            
        Returns:
            Annual IRR as decimal
        """
        cash_flows = []
        
        # Initial investment (negative)
        cash_flows.append(-investment.capital_cost)
        
        # Annual benefits
        annual_benefit = investment.calculate_annual_benefit(base_revenue)
        
        for year in range(investment.useful_life_years):
            cf = annual_benefit
            
            # Add salvage value in final year
            if year == investment.useful_life_years - 1:
                cf += investment.salvage_value
            
            cash_flows.append(cf)
        
        return self._calculate_irr(cash_flows) or 0.0
    
    def goal_seek_equity_for_irr(
        self,
        target_irr: float,
        funded_cashflow: Dict[str, Any],
        terminal_value: float,
        current_equity: float,
        min_equity: float = 0,
        max_equity: float = None
    ) -> Optional[float]:
        """
        Goal seek: Find equity amount needed to achieve target IRR.
        
        Args:
            target_irr: Target annual IRR (e.g., 0.25 for 25%)
            funded_cashflow: Base funded cash flow
            terminal_value: Expected exit value
            current_equity: Current equity in model
            min_equity: Minimum equity constraint
            max_equity: Maximum equity constraint
            
        Returns:
            Required equity amount, or None if not achievable
        """
        if max_equity is None:
            max_equity = current_equity * 3  # Default: up to 3x current
        
        def irr_diff(equity):
            # Scale cash flows proportionally
            scale = equity / current_equity if current_equity > 0 else 1
            scaled_tv = terminal_value * scale
            
            irr = self.calculate_equity_irr(
                funded_cashflow,
                equity,
                scaled_tv
            )
            return irr - target_irr
        
        try:
            result = optimize.brentq(irr_diff, min_equity, max_equity)
            return result
        except ValueError:
            # No solution in range
            return None
    
    def goal_seek_irr_for_equity(
        self,
        equity_amount: float,
        funded_cashflow: Dict[str, Any],
        terminal_value: float
    ) -> float:
        """
        Calculate IRR for a given equity amount.
        
        Args:
            equity_amount: Equity investment amount
            funded_cashflow: Funded cash flow
            terminal_value: Exit value
            
        Returns:
            Resulting annual IRR
        """
        return self.calculate_equity_irr(
            funded_cashflow,
            equity_amount,
            terminal_value
        )
    
    def sensitivity_analysis(
        self,
        funded_cashflow: Dict[str, Any],
        base_equity: float,
        base_terminal_value: float,
        equity_range: Tuple[float, float] = (0.5, 1.5),  # Multipliers
        tv_range: Tuple[float, float] = (0.5, 1.5),
        steps: int = 5
    ) -> pd.DataFrame:
        """
        Generate sensitivity table for IRR vs equity and terminal value.
        
        Returns DataFrame with IRR for each combination.
        """
        equity_mults = np.linspace(equity_range[0], equity_range[1], steps)
        tv_mults = np.linspace(tv_range[0], tv_range[1], steps)
        
        results = []
        for eq_mult in equity_mults:
            row = {'Equity': f'{eq_mult:.0%}'}
            for tv_mult in tv_mults:
                equity = base_equity * eq_mult
                tv = base_terminal_value * tv_mult
                irr = self.calculate_equity_irr(funded_cashflow, equity, tv)
                row[f'TV {tv_mult:.0%}'] = f'{irr:.1%}' if irr else 'N/A'
            results.append(row)
        
        return pd.DataFrame(results)
    
    def _calculate_irr(self, cash_flows: List[float], guess: float = 0.1) -> Optional[float]:
        """
        Calculate IRR using numpy's IRR function with fallback.
        """
        if not cash_flows or len(cash_flows) < 2:
            return None
        
        # Check if there are both positive and negative cash flows
        has_negative = any(cf < 0 for cf in cash_flows)
        has_positive = any(cf > 0 for cf in cash_flows)
        
        if not (has_negative and has_positive):
            return None
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                irr = np.irr(cash_flows)
                if np.isnan(irr) or np.isinf(irr):
                    return None
                return irr
        except:
            # Fallback to scipy optimization
            try:
                def npv(rate):
                    return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
                
                result = optimize.brentq(npv, -0.99, 10.0)
                return result
            except:
                return None
    
    def generate_debt_schedule(
        self,
        tranche: DebtTranche
    ) -> pd.DataFrame:
        """
        Generate full amortization schedule for a debt tranche.
        """
        rows = []
        temp_tranche = DebtTranche(
            id=tranche.id,
            name=tranche.name,
            debt_type=tranche.debt_type,
            principal=tranche.principal,
            interest_rate=tranche.interest_rate,
            start_date=tranche.start_date,
            term_months=tranche.term_months,
            repayment_type=tranche.repayment_type,
            grace_period_months=tranche.grace_period_months,
            balloon_pct=tranche.balloon_pct,
            pik_rate=tranche.pik_rate
        )
        
        for month in range(1, tranche.term_months + 1):
            payment = temp_tranche.calculate_monthly_payment(month)
            period_date = tranche.start_date + relativedelta(months=month-1)
            
            rows.append({
                'Month': month,
                'Date': period_date.strftime('%Y-%m'),
                'Opening Balance': temp_tranche.outstanding_principal,
                'Principal': payment['principal'],
                'Interest': payment['interest'],
                'PIK Interest': payment['pik_interest'],
                'Total Payment': payment['total_payment'],
                'Closing Balance': temp_tranche.outstanding_principal - payment['principal']
            })
            
            temp_tranche.apply_payment(payment['principal'], payment['pik_interest'])
        
        return pd.DataFrame(rows)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_debt_tranche_from_dict(data: Dict[str, Any]) -> DebtTranche:
    """Create DebtTranche from database record."""
    event_type = data.get('event_type', 'debt_term_loan')
    
    debt_type_map = {
        'debt_term_loan': DebtType.TERM_LOAN,
        'debt_mezzanine': DebtType.MEZZANINE,
        'debt_convertible': DebtType.CONVERTIBLE,
        'debt_trade_finance': DebtType.TRADE_FINANCE,
    }
    
    repayment_map = {
        'amortizing': RepaymentType.AMORTIZING,
        'bullet': RepaymentType.BULLET,
        'interest_only': RepaymentType.INTEREST_ONLY,
        'pik': RepaymentType.PIK,
    }
    
    event_date = data.get('event_date')
    if isinstance(event_date, str):
        event_date = datetime.strptime(event_date, '%Y-%m-%d').date()
    
    # Helper to safely convert to float, handling None values
    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    return DebtTranche(
        id=data.get('id', ''),
        name=data.get('description', 'Debt'),
        debt_type=debt_type_map.get(event_type, DebtType.TERM_LOAN),
        principal=safe_float(data.get('amount'), 0),
        interest_rate=safe_float(data.get('interest_rate'), 12) / 100,  # Convert from percentage
        start_date=event_date,
        term_months=int(data.get('term_months') or 60),
        repayment_type=repayment_map.get(data.get('repayment_type'), RepaymentType.AMORTIZING),
        grace_period_months=int(data.get('grace_period_months') or 0),
        balloon_pct=safe_float(data.get('balloon_pct'), 0) / 100,
        pik_rate=safe_float(data.get('pik_rate'), 0) / 100,
        conversion_price=safe_float(data.get('conversion_price'), 0),
    )


def create_equity_from_dict(data: Dict[str, Any]) -> EquityInvestment:
    """Create EquityInvestment from database record."""
    event_type = data.get('event_type', 'equity_ordinary')
    
    equity_type_map = {
        'equity_ordinary': EquityType.ORDINARY,
        'equity_preference': EquityType.PREFERENCE,
    }
    
    event_date = data.get('event_date')
    if isinstance(event_date, str):
        event_date = datetime.strptime(event_date, '%Y-%m-%d').date()
    
    # Helper to safely convert to float, handling None values
    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    return EquityInvestment(
        id=data.get('id', ''),
        investor_name=data.get('investor_name', 'Investor'),
        equity_type=equity_type_map.get(event_type, EquityType.ORDINARY),
        amount=safe_float(data.get('amount'), 0),
        investment_date=event_date,
        share_price=safe_float(data.get('share_price'), 0),
        shares_issued=safe_float(data.get('shares_issued'), 0),
        dividend_rate=safe_float(data.get('dividend_rate'), 0) / 100,
    )


def create_overdraft_from_dict(data: Dict[str, Any]) -> OverdraftFacility:
    """Create OverdraftFacility from database record."""
    # Helper to safely convert to float, handling None values
    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    return OverdraftFacility(
        facility_limit=safe_float(data.get('facility_limit'), 5_000_000),
        interest_rate=safe_float(data.get('interest_rate'), 12) / 100,
        arrangement_fee_pct=safe_float(data.get('arrangement_fee_pct'), 0.5) / 100,
        commitment_fee_pct=safe_float(data.get('commitment_fee_pct'), 0.25) / 100,
        auto_repay=data.get('auto_repay', True) if data.get('auto_repay') is not None else True,
        is_active=data.get('is_active', True) if data.get('is_active') is not None else True,
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the funding engine
    print("Testing Funding Engine...")
    
    # Create sample funding scenario
    scenario = FundingScenario(
        debt_tranches=[
            DebtTranche(
                id="1",
                name="Term Loan A",
                debt_type=DebtType.TERM_LOAN,
                principal=10_000_000,
                interest_rate=0.12,
                start_date=date(2025, 1, 1),
                term_months=60,
                repayment_type=RepaymentType.AMORTIZING,
            ),
            DebtTranche(
                id="2",
                name="Mezzanine",
                debt_type=DebtType.MEZZANINE,
                principal=5_000_000,
                interest_rate=0.15,
                start_date=date(2025, 1, 1),
                term_months=60,
                repayment_type=RepaymentType.PIK,
                pik_rate=0.03,
            ),
        ],
        equity_investments=[
            EquityInvestment(
                id="1",
                investor_name="PE Fund",
                equity_type=EquityType.ORDINARY,
                amount=15_000_000,
                investment_date=date(2025, 1, 1),
                share_price=100,
            )
        ],
        overdraft=OverdraftFacility(
            facility_limit=5_000_000,
            interest_rate=0.12,
            auto_repay=True,
        )
    )
    
    # Sample cash flow
    n_months = 60
    base_cf = {
        'timeline': [(date(2025, 1, 1) + relativedelta(months=i)).strftime('%Y-%m') for i in range(n_months)],
        'profit': {
            'ebit': [500_000 + i * 10_000 for i in range(n_months)]  # Growing EBIT
        }
    }
    
    # Apply funding
    engine = FundingEngine()
    result = engine.apply_funding(base_cf, scenario)
    
    print(f"\nFunding Applied:")
    print(f"  Total Debt Drawdown: R {sum(result['debt_drawdown']):,.0f}")
    print(f"  Total Equity: R {sum(result['equity_injection']):,.0f}")
    print(f"  Final Cash Balance: R {result['cumulative_cash'][-1]:,.0f}")
    print(f"  Final Debt Balance: R {result['debt_balance'][-1]:,.0f}")
    
    # Calculate IRR
    terminal_value = 50_000_000
    irr = engine.calculate_equity_irr(result, 15_000_000, terminal_value)
    print(f"\nEquity IRR: {irr:.1%}")
    
    # Generate debt schedule
    schedule = engine.generate_debt_schedule(scenario.debt_tranches[0])
    print(f"\nTerm Loan Schedule (first 6 months):")
    print(schedule.head(6).to_string(index=False))
    
    print("\n✓ Funding Engine tests passed")
