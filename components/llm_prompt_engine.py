"""
LLM Prompt Engine for What-If Agent
===================================
Sprint 23: Natural language interface for scenario optimization.

Enables users to ask questions in natural language and automatically
optimize scenarios based on constraints and objectives.
"""

import streamlit as st
import json
from typing import Dict, Any, Optional, List, Tuple
from scipy.optimize import minimize, differential_evolution
import numpy as np


# Try to import LLM libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    try:
        from anthropic import Anthropic
        ANTHROPIC_AVAILABLE = True
    except ImportError:
        ANTHROPIC_AVAILABLE = False


class LLMPromptEngine:
    """
    LLM-powered prompt engine for natural language scenario queries.
    """
    
    def __init__(self):
        """Initialize LLM prompt engine."""
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM clients from secrets."""
        # Try OpenAI
        if OPENAI_AVAILABLE:
            try:
                api_key = st.secrets.get("openai", {}).get("api_key")
                if api_key:
                    self.openai_client = openai.OpenAI(api_key=api_key)
            except:
                pass
        
        # Try Anthropic
        if ANTHROPIC_AVAILABLE:
            try:
                api_key = st.secrets.get("anthropic", {}).get("api_key")
                if api_key:
                    self.anthropic_client = Anthropic(api_key=api_key)
            except:
                pass
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.openai_client is not None or self.anthropic_client is not None
    
    def parse_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parse natural language query into structured parameters.
        
        Args:
            query: Natural language query
            context: Optional context (current forecast, assumptions, etc.)
        
        Returns:
            Dictionary with parsed intent and parameters
        """
        if not self.is_available():
            # Fallback: Simple keyword-based parsing
            return self._fallback_parse(query)
        
        # Build prompt for LLM
        system_prompt = self._build_system_prompt(context)
        user_prompt = f"Parse this query: {query}"
        
        try:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                result = json.loads(response.choices[0].message.content)
                return result
            elif self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                result = json.loads(response.content[0].text)
                return result
        except Exception as e:
            st.warning(f"LLM parsing failed, using fallback: {e}")
            return self._fallback_parse(query)
        
        return self._fallback_parse(query)
    
    def _build_system_prompt(self, context: Optional[Dict[str, Any]]) -> str:
        """Build system prompt for LLM."""
        prompt = """You are a financial modeling assistant. Parse user queries about scenario optimization.

Return a JSON object with this structure:
{
    "intent": "optimize" | "adjust" | "analyze" | "compare",
    "objective": {
        "type": "maximize" | "minimize",
        "metric": "equity_irr" | "ebit" | "revenue" | "margin" | "return_to_shareholders",
        "description": "What to optimize"
    },
    "constraints": [
        {
            "type": "equity_limit" | "debt_limit" | "revenue_limit" | "margin_limit" | "parameter_range",
            "parameter": "equity_pct" | "debt_pct" | "revenue_pct" | etc.,
            "operator": "<=" | ">=" | "==",
            "value": number,
            "description": "Constraint description"
        }
    ],
    "parameters": {
        "revenue_pct": null | number,
        "utilization_pct": null | number,
        "cogs_pct": null | number,
        "opex_pct": null | number,
        "debt_pct": null | number,
        "equity_pct": null | number,
        "overdraft_pct": null | number,
        "trade_finance_pct": null | number
    },
    "notes": "Additional context or clarifications"
}

Examples:
- "Maximize return to shareholders with max 25% equity dilution"
  -> {
      "intent": "optimize",
      "objective": {"type": "maximize", "metric": "return_to_shareholders"},
      "constraints": [{"type": "equity_limit", "parameter": "equity_pct", "operator": "<=", "value": 25}],
      "parameters": {}
  }

- "Find optimal debt/equity mix to maximize IRR with debt < 50%"
  -> {
      "intent": "optimize",
      "objective": {"type": "maximize", "metric": "equity_irr"},
      "constraints": [{"type": "debt_limit", "parameter": "debt_pct", "operator": "<=", "value": 50}],
      "parameters": {}
  }
"""
        return prompt
    
    def _fallback_parse(self, query: str) -> Dict[str, Any]:
        """Fallback keyword-based parsing when LLM unavailable."""
        query_lower = query.lower()
        
        result = {
            "intent": "optimize",
            "objective": {"type": "maximize", "metric": "return_to_shareholders"},
            "constraints": [],
            "parameters": {},
            "notes": "Using fallback parser - LLM not available"
        }
        
        # Detect objective
        if "maximize" in query_lower or "maximise" in query_lower:
            result["objective"]["type"] = "maximize"
        elif "minimize" in query_lower or "minimise" in query_lower:
            result["objective"]["type"] = "minimize"
        
        # Detect metric
        if "irr" in query_lower or "return" in query_lower:
            if "equity" in query_lower:
                result["objective"]["metric"] = "equity_irr"
            else:
                result["objective"]["metric"] = "return_to_shareholders"
        elif "ebit" in query_lower:
            result["objective"]["metric"] = "ebit"
        elif "revenue" in query_lower:
            result["objective"]["metric"] = "revenue"
        elif "margin" in query_lower:
            result["objective"]["metric"] = "margin"
        
        # Detect equity constraint
        if "equity" in query_lower and ("25%" in query or "25 %" in query_lower):
            result["constraints"].append({
                "type": "equity_limit",
                "parameter": "equity_pct",
                "operator": "<=",
                "value": 25,
                "description": "Equity dilution limit: 25%"
            })
        elif "equity" in query_lower:
            # Try to extract percentage
            import re
            equity_match = re.search(r'(\d+)%', query)
            if equity_match:
                result["constraints"].append({
                    "type": "equity_limit",
                    "parameter": "equity_pct",
                    "operator": "<=",
                    "value": float(equity_match.group(1)),
                    "description": f"Equity dilution limit: {equity_match.group(1)}%"
                })
        
        # Detect debt constraint
        if "debt" in query_lower:
            import re
            debt_match = re.search(r'debt\s*[<>=]?\s*(\d+)%', query_lower)
            if debt_match:
                result["constraints"].append({
                    "type": "debt_limit",
                    "parameter": "debt_pct",
                    "operator": "<=",
                    "value": float(debt_match.group(1)),
                    "description": f"Debt limit: {debt_match.group(1)}%"
                })
        
        return result


class ScenarioOptimizer:
    """
    Optimize scenarios based on objectives and constraints.
    """
    
    def __init__(self, baseline_forecast: Dict[str, Any], funding_scenario: Optional[Any] = None):
        """
        Initialize optimizer.
        
        Args:
            baseline_forecast: Baseline forecast results
            funding_scenario: Optional funding scenario for IRR calculations
        """
        self.baseline = baseline_forecast
        self.funding_scenario = funding_scenario
    
    def optimize(
        self,
        objective: Dict[str, Any],
        constraints: List[Dict[str, Any]],
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Optimize scenario based on objective and constraints.
        
        Args:
            objective: Objective specification (maximize/minimize metric)
            constraints: List of constraint specifications
            parameter_bounds: Optional bounds for parameters
        
        Returns:
            Dictionary with optimal parameters and results
        """
        from components.whatif_agent import calculate_adjusted_forecast
        
        # Default parameter bounds
        if parameter_bounds is None:
            parameter_bounds = {
                'revenue_pct': (-0.5, 0.5),
                'utilization_pct': (-0.3, 0.3),
                'cogs_pct': (-0.2, 0.2),
                'opex_pct': (-0.3, 0.3),
                'debt_pct': (0.0, 1.0),  # 0-100% of funding
                'equity_pct': (0.0, 1.0),  # 0-100% of funding
                'overdraft_pct': (0.0, 0.5),
                'trade_finance_pct': (0.0, 0.5)
            }
        
        # Objective function
        def objective_function(params):
            # Extract parameters
            adjustments = {
                'revenue_pct': params[0],
                'utilization_pct': params[1],
                'cogs_pct': params[2],
                'opex_pct': params[3]
            }
            
            # Calculate adjusted forecast
            adjusted = calculate_adjusted_forecast(self.baseline, adjustments)
            if not adjusted:
                return -1e10 if objective['type'] == 'maximize' else 1e10
            
            # Calculate metric value
            metric_value = self._calculate_metric(adjusted, objective['metric'])
            
            # Return negative for maximize (minimize negative = maximize)
            if objective['type'] == 'maximize':
                return -metric_value
            else:
                return metric_value
        
        # Constraint functions
        constraint_functions = []
        for constraint in constraints:
            if constraint['type'] == 'equity_limit':
                # This would need funding scenario integration
                # For now, skip funding constraints in optimization
                pass
        
        # Bounds for optimization
        bounds = [
            parameter_bounds['revenue_pct'],
            parameter_bounds['utilization_pct'],
            parameter_bounds['cogs_pct'],
            parameter_bounds['opex_pct']
        ]
        
        # Initial guess (no adjustments)
        x0 = [0.0, 0.0, 0.0, 0.0]
        
        # Run optimization
        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_functions if constraint_functions else None,
                options={'maxiter': 100}
            )
            
            if result.success:
                optimal_adjustments = {
                    'revenue_pct': result.x[0],
                    'utilization_pct': result.x[1],
                    'cogs_pct': result.x[2],
                    'opex_pct': result.x[3]
                }
                
                # Calculate final forecast
                optimal_forecast = calculate_adjusted_forecast(self.baseline, optimal_adjustments)
                optimal_metric = self._calculate_metric(optimal_forecast, objective['metric'])
                
                return {
                    'success': True,
                    'optimal_parameters': optimal_adjustments,
                    'optimal_forecast': optimal_forecast,
                    'optimal_metric_value': optimal_metric,
                    'iterations': result.nit,
                    'message': result.message
                }
            else:
                return {
                    'success': False,
                    'error': f"Optimization failed: {result.message}",
                    'optimal_parameters': None
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"Optimization error: {str(e)}",
                'optimal_parameters': None
            }
    
    def _calculate_metric(self, forecast: Dict[str, Any], metric: str) -> float:
        """Calculate metric value from forecast."""
        summary = forecast.get('summary', {})
        
        if metric == 'equity_irr':
            # Would need funding scenario for IRR
            # For now, return EBIT as proxy
            return summary.get('total_ebit', 0)
        elif metric == 'return_to_shareholders':
            # Proxy: EBIT / Equity (would need funding data)
            return summary.get('total_ebit', 0)
        elif metric == 'ebit':
            return summary.get('total_ebit', 0)
        elif metric == 'revenue':
            return summary.get('total_revenue', 0)
        elif metric == 'margin':
            return summary.get('avg_ebit_margin', 0)
        else:
            return summary.get('total_ebit', 0)


def optimize_funding_mix(
    baseline_forecast: Dict[str, Any],
    objective: str,
    constraints: List[Dict[str, Any]],
    funding_engine: Optional[Any] = None,
    db: Optional[Any] = None,
    scenario_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Optimize funding mix (debt, equity, overdraft, trade finance).
    
    Args:
        baseline_forecast: Baseline forecast results
        objective: Objective (e.g., "maximize_irr", "minimize_cost")
        constraints: List of constraints (e.g., equity_pct <= 25%)
        funding_engine: Optional funding engine for IRR calculations
        db: Database handler
        scenario_id: Scenario ID
        user_id: User ID
    
    Returns:
        Dictionary with optimal funding mix and results
    """
    try:
        from funding_engine import FundingEngine, FundingScenario, DebtTranche, EquityInvestment
    except ImportError:
        return {
            'success': False,
            'error': 'Funding engine not available'
        }
    
    # Extract constraints
    equity_limit = None
    debt_limit = None
    
    for constraint in constraints:
        if constraint.get('type') == 'equity_limit':
            equity_limit = constraint.get('value', 100) / 100  # Convert to decimal
        elif constraint.get('type') == 'debt_limit':
            debt_limit = constraint.get('value', 100) / 100
    
    # Objective function for optimization
    def objective_function(params):
        """
        Calculate negative IRR (for maximization).
        params: [debt_pct, equity_pct, overdraft_pct, trade_finance_pct]
        """
        debt_pct, equity_pct, overdraft_pct, trade_finance_pct = params
        
        # Normalize to sum to 1.0 (100%)
        total = debt_pct + equity_pct + overdraft_pct + trade_finance_pct
        if total == 0:
            return 1e10  # Penalty for invalid mix
        
        debt_pct /= total
        equity_pct /= total
        overdraft_pct /= total
        trade_finance_pct /= total
        
        # Check constraints
        if equity_limit and equity_pct > equity_limit:
            return 1e10  # Penalty for constraint violation
        if debt_limit and debt_pct > debt_limit:
            return 1e10
        
        # Calculate total funding needed (simplified: use forecast EBIT as proxy)
        total_ebit = baseline_forecast.get('summary', {}).get('total_ebit', 0)
        if total_ebit <= 0:
            return 1e10
        
        # Estimate funding need (simplified: 2x annual EBIT)
        annual_ebit = total_ebit / (len(baseline_forecast.get('timeline', [60])) / 12)
        funding_need = annual_ebit * 2
        
        # Create funding scenario
        scenario = FundingScenario()
        
        if equity_pct > 0:
            equity_amount = funding_need * equity_pct
            scenario.equity_investments.append(
                EquityInvestment(
                    id='opt_equity',
                    investor_name='Optimized Equity',
                    equity_type='ORDINARY',
                    amount=equity_amount,
                    investment_date=None,
                    share_price=1.0,
                    shares_issued=equity_amount,
                    dividend_rate=0.0
                )
            )
        
        if debt_pct > 0:
            debt_amount = funding_need * debt_pct
            scenario.debt_tranches.append(
                DebtTranche(
                    id='opt_debt',
                    name='Optimized Debt',
                    debt_type='TERM_LOAN',
                    principal=debt_amount,
                    interest_rate=0.12,  # 12% default
                    start_date=None,
                    term_months=60,
                    repayment_type='AMORTIZING'
                )
            )
        
        # Apply funding and calculate IRR
        try:
            engine = FundingEngine()
            funded_cf = engine.apply_funding(baseline_forecast, scenario)
            
            # Calculate IRR (simplified)
            total_equity = scenario.total_equity
            if total_equity > 0:
                # Use terminal value estimate
                terminal_multiple = 6.0
                final_ebit = baseline_forecast.get('profit', {}).get('ebit', [])
                if final_ebit:
                    final_12m_ebit = sum(final_ebit[-12:]) if len(final_ebit) >= 12 else sum(final_ebit)
                    terminal_value = final_12m_ebit * 1.1 * terminal_multiple
                    
                    holding_period = 5  # years
                    irr = engine.calculate_equity_irr(
                        funded_cf,
                        total_equity,
                        terminal_value,
                        holding_period * 12
                    )
                    return -irr  # Negative for maximization
        except Exception:
            pass
        
        return 1e10  # Penalty for calculation failure
    
    # Bounds: each component 0-1 (will be normalized)
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5), (0.0, 0.5)]
    
    # Constraint: sum should be > 0 (at least some funding)
    def constraint_sum(params):
        return sum(params) - 0.1  # At least 10% total
    
    constraints_opt = [{'type': 'ineq', 'fun': constraint_sum}]
    
    # Add equity constraint if specified
    if equity_limit:
        def equity_constraint(params):
            total = sum(params)
            if total == 0:
                return -1
            return equity_limit - (params[1] / total)  # equity_pct <= equity_limit
        constraints_opt.append({'type': 'ineq', 'fun': equity_constraint})
    
    # Initial guess (balanced mix)
    x0 = [0.4, 0.3, 0.2, 0.1]
    
    try:
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_opt,
            options={'maxiter': 50}
        )
        
        if result.success:
            # Normalize results
            total = sum(result.x)
            optimal_mix = {
                'debt_pct': result.x[0] / total,
                'equity_pct': result.x[1] / total,
                'overdraft_pct': result.x[2] / total,
                'trade_finance_pct': result.x[3] / total
            }
            
            # Calculate final IRR
            optimal_irr = -result.fun
            
            return {
                'success': True,
                'optimal_mix': optimal_mix,
                'optimal_irr': optimal_irr,
                'iterations': result.nit,
                'message': 'Optimization complete'
            }
        else:
            return {
                'success': False,
                'error': f"Optimization failed: {result.message}",
                'optimal_mix': None
            }
    except Exception as e:
        return {
            'success': False,
            'error': f"Optimization error: {str(e)}",
            'optimal_mix': None
        }
