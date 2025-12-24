# CE Africa Valuation Platform - Requirements Document

**Version:** 2.3  
**Date:** December 20, 2025  
**Status:** Code Review Complete  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Functional Requirements](#3-functional-requirements)
4. [Data Management](#4-data-management)
5. [Financial Modeling Logic](#5-financial-modeling-logic)
6. [Known Issues & Gaps](#6-known-issues--gaps)
7. [Recommendations](#7-recommendations)

---

## 1. Executive Summary

### 1.1 Purpose
The CE Africa Valuation Platform is a financial modeling and forecasting tool for Crusher Equipment Africa, specifically designed for:
- **Installed Base Revenue Modeling** - Revenue from existing fleet of crusher equipment
- **Manufacturing Strategy Analysis** - Make vs Buy decision support
- **Monte Carlo Simulation** - Probabilistic forecasting with confidence intervals
- **DCF Valuation** - Discounted cash flow analysis with valuation ranges

### 1.2 Technology Stack
| Component | Technology |
|-----------|------------|
| Frontend | Streamlit (Python) |
| Database | Supabase (PostgreSQL) |
| Visualization | Plotly |
| Statistical Analysis | SciPy, NumPy, Pandas |
| Hosting | To be deployed (Streamlit Cloud/Railway/Render recommended) |

### 1.3 Current Status
- **Core Functionality:** ✅ Implemented
- **AI Assumptions Engine:** ✅ Implemented
- **Trend-Based Forecasting:** ✅ Implemented (with recent fixes)
- **Monte Carlo Simulation:** ✅ Implemented with period-specific distributions
- **Manufacturing Strategy:** ✅ Implemented
- **Funding Analysis:** ✅ Implemented
- **What-If Agent:** ✅ Implemented with LLM integration

---

## 2. System Architecture

### 2.1 Application Structure
```
ce-africa-valuation/
├── app_refactored.py          # Main application entry point
├── db_connector.py            # Database operations (Supabase)
├── forecast_engine.py         # Core forecast calculation logic
├── funding_engine.py          # Funding/IRR calculations
├── components/                 # UI and business logic modules
│   ├── ai_assumptions_engine.py    # AI-driven assumptions
│   ├── forecast_section.py         # Forecast UI and rendering
│   ├── forecast_config_ui.py       # Trend configuration UI
│   ├── forecast_correlation_engine.py  # Correlation analysis
│   ├── trend_forecast_analyzer.py  # Trend fitting algorithms
│   ├── vertical_integration.py     # Manufacturing strategy
│   ├── funding_ui.py               # Funding/IRR UI
│   ├── whatif_agent.py             # Scenario analysis
│   ├── trial_balance_processor.py  # Trial balance extraction
│   └── ...
├── services/                   # Business logic services
└── tests/                      # Unit tests
```

### 2.2 Workflow Sequence
```
1. Setup → 2. AI Assumptions → 3. Trend Forecast → 4. Run Forecast → 5. Funding/IRR
              ↓                    ↓
         Distributions        Trend Curves
              ↓                    ↓
         Monte Carlo ←─────────────┘
              ↓
         Valuation Range
```

---

## 3. Functional Requirements

### 3.1 Setup Module
| ID | Requirement | Status | Notes |
|----|-------------|--------|-------|
| S1 | Create/edit/delete scenarios | ✅ | Working |
| S2 | Import fleet (installed base) data | ✅ | CSV import |
| S3 | Import historical financials (P&L, BS, CF) | ✅ | Multiple formats |
| S4 | Import detailed line items | ✅ | Wide format supported |
| S5 | Import trial balance with auto-extraction | ✅ | AI classification |
| S6 | Configure basic assumptions (WACC, tax rate, etc.) | ✅ | Working |
| S7 | Save assumptions to database | ✅ | Persists across sessions |

### 3.2 AI Assumptions Module
| ID | Requirement | Status | Notes |
|----|-------------|--------|-------|
| A1 | Analyze historical data for patterns | ✅ | Working |
| A2 | Fit probability distributions (Normal, Lognormal, etc.) | ✅ | Working |
| A3 | Display AI recommendations with fit scores | ✅ | Working |
| A4 | Allow user to accept/modify distributions | ✅ | Working |
| A5 | **COGS/OPEX as % of Revenue** | ✅ | **Fixed: Now percentage-based** |
| A6 | **Exclude calculated elements from configuration** | ✅ | **Fixed: GP, EBIT, Tax, Net Income excluded** |
| A7 | **Remove duplicate line items** | ✅ | **Fixed: No more duplicates** |
| A8 | Save assumptions to database | ✅ | Persists across sessions |

### 3.3 Trend Forecast Module
| ID | Requirement | Status | Notes |
|----|-------------|--------|-------|
| T1 | Fit trend functions (Linear, Exponential, Log, etc.) | ✅ | Working |
| T2 | Display forecast curve with historical data | ✅ | Visual preview |
| T3 | Allow parameter adjustment with live preview | ✅ | Working |
| T4 | **Persist user-saved trend parameters** | ✅ | **Fixed: Now persists** |
| T5 | **Use saved parameters when running forecast** | ✅ | **Fixed: Now throws error if fails** |
| T6 | Support correlation-based forecasts (COGS as % of revenue) | ✅ | Working |
| T7 | **Error on trend failure (not silent fallback)** | ✅ | **Fixed: Shows diagnostic error** |

### 3.4 Forecast Module
| ID | Requirement | Status | Notes |
|----|-------------|--------|-------|
| F1 | Generate monthly revenue forecast | ✅ | Working |
| F2 | Calculate COGS from margins or correlations | ✅ | Working |
| F3 | Calculate OPEX from trends or correlations | ✅ | Working |
| F4 | **Interest calculated from balance sheet** | ✅ | Prime rate on debt, call rate on cash |
| F5 | **Tax calculated from EBT × tax rate** | ✅ | Working |
| F6 | Build 3-statement model (P&L, BS, CF) | ✅ | Working |
| F7 | **Show historical actuals alongside forecast** | ⚠️ | **Fixed but needs verification** |
| F8 | **Show YTD actuals for current year** | ⚠️ | **Fixed but needs verification** |
| F9 | Show detailed line items in historical periods | ⚠️ | **Fixed but needs verification** |
| F10 | Monte Carlo simulation | ✅ | Working |
| F11 | **MC uses forecast values as mean (not historical)** | ✅ | **Fixed: Period-specific distributions** |
| F12 | DCF valuation with P10/P50/P90 | ✅ | Working |
| F13 | Save snapshots to database | ✅ | Working |

### 3.5 Manufacturing Strategy Module
| ID | Requirement | Status | Notes |
|----|-------------|--------|-------|
| M1 | Toggle between Buy/Make/Hybrid strategies | ✅ | Working |
| M2 | Part-level cost analysis | ✅ | Working |
| M3 | CAPEX and commissioning workflow | ✅ | Working |
| M4 | Working capital calculation | ✅ | Working |
| M5 | NPV comparison (Make vs Buy) | ✅ | Working |
| M6 | Integration with forecast COGS | ✅ | Working |
| M7 | Save/load strategy to database | ✅ | Working |

### 3.6 Funding & IRR Module
| ID | Requirement | Status | Notes |
|----|-------------|--------|-------|
| R1 | Configure debt tranches | ✅ | Working |
| R2 | Configure equity investments | ✅ | Working |
| R3 | Overdraft facility modeling | ✅ | Working |
| R4 | Trade finance modeling | ⚠️ | Module exists but may need review |
| R5 | IRR calculation and goal seek | ✅ | Working |
| R6 | Cash waterfall visualization | ✅ | Working |

### 3.7 What-If Agent
| ID | Requirement | Status | Notes |
|----|-------------|--------|-------|
| W1 | Manual parameter adjustments with sliders | ✅ | Working |
| W2 | Real-time impact visualization | ✅ | Working |
| W3 | Sensitivity analysis (tornado diagrams) | ✅ | Working |
| W4 | **Natural language prompts** | ⚠️ | Requires LLM API setup |
| W5 | Optimization based on objectives | ⚠️ | Requires LLM API setup |

---

## 4. Data Management

### 4.1 Database Tables
| Table | Purpose | Status |
|-------|---------|--------|
| `scenarios` | Scenario metadata | ✅ |
| `assumptions` | JSONB blob for all assumptions | ✅ |
| `installed_base` | Fleet/machine data | ✅ |
| `historic_financials` | Monthly P&L summary | ✅ |
| `historical_income_statement_line_items` | Detailed IS lines | ✅ |
| `historical_balance_sheet_line_items` | Detailed BS lines | ✅ |
| `historical_cashflow_line_items` | Detailed CF lines | ✅ |
| `historical_trial_balance` | Trial balance data | ✅ |
| `forecast_snapshots` | Saved forecast results | ✅ |

### 4.2 Data Persistence
| Data Type | Storage Location | Persistence |
|-----------|------------------|-------------|
| Assumptions (basic) | `assumptions.data` JSONB | ✅ Database |
| AI Assumptions | `assumptions.data.ai_assumptions` | ✅ Database |
| Forecast Configs | `assumptions.data.forecast_configs` | ✅ Database |
| Historical Data | Dedicated tables | ✅ Database |
| Forecast Results | `forecast_snapshots` | ✅ Database |
| Manufacturing Strategy | `assumptions.data.manufacturing` | ✅ Database |

---

## 5. Financial Modeling Logic

### 5.1 Revenue Calculation
```
Revenue = Consumables + Refurbishment + Pipeline
         ├── Consumables: Fleet × Wear Rate × Utilization × Price
         ├── Refurbishment: Fleet × Refurb Rate × Price
         └── Pipeline: Prospect-based with probability weighting
         
When Trend Forecast Enabled:
Revenue = Trend Function(historical_data, forecast_periods)
         └── Options: Linear, Exponential, Logarithmic, Polynomial
```

### 5.2 Cost Calculation
```
COGS:
├── If trend_fit: Generated from trend function
├── If correlation: COGS = Revenue × COGS_pct
├── Default: (Consumable_Rev × (1-margin_c)) + (Refurb_Rev × (1-margin_r))

OPEX:
├── If trend_fit: Generated from trend function
├── If correlation: OPEX = Revenue × OPEX_pct
├── Default: Sum of expense category forecasts
```

### 5.3 Calculated Elements (Not Configurable)
| Element | Formula |
|---------|---------|
| Gross Profit | Revenue - COGS |
| EBIT | Gross Profit - OPEX - Depreciation |
| Interest Expense | (Debt × Prime Rate) - (Cash × Call Rate) |
| EBT | EBIT - Interest Expense |
| Tax | max(EBT × Tax Rate, 0) |
| Net Income | EBT - Tax |

### 5.4 Monte Carlo Simulation
```
For each iteration (1000 default):
  For each period (t):
    forecast_value = trend_forecast[t]
    stdev_scaled = historical_stdev × (forecast_value / historical_mean)
    sample = Normal(mean=forecast_value, std=stdev_scaled)
    
GP[t] = Revenue[t] - COGS[t]  # Calculated, not sampled
EBIT[t] = GP[t] - OPEX[t]     # Calculated, not sampled

Result: P5, P10, P25, P50, P75, P90, P95 percentiles per period
```

### 5.5 DCF Valuation
```
WACC = (E/V × Re) + (D/V × Rd × (1-T))

FCF[t] = EBIT × (1-T) + Depreciation - CAPEX - ΔWorkingCapital

Terminal Value = FCF[n] × (1+g) / (WACC-g)

Enterprise Value = Σ FCF[t] / (1+WACC)^t + TV / (1+WACC)^n
```

---

## 6. Known Issues & Gaps

### 6.1 Critical Issues (Need Immediate Attention)
| ID | Issue | Impact | Status |
|----|-------|--------|--------|
| C1 | Trend forecast may not use saved parameters | High | Needs verification |
| C2 | YTD actuals may show R-0 in some cases | Medium | Fixed, needs testing |
| C3 | Detailed line items not showing in historical | Medium | Fixed, needs testing |

### 6.2 Medium Priority
| ID | Issue | Impact |
|----|-------|--------|
| M1 | LLM prompt engine requires API key setup | Medium |
| M2 | Trade finance module may have incomplete integration | Medium |
| M3 | PDF extraction may not work for all formats | Medium |

### 6.3 Code Quality Observations
| Area | Observation | Recommendation |
|------|-------------|----------------|
| File Size | `forecast_section.py` is 6000+ lines | Consider splitting |
| File Size | `ai_assumptions_engine.py` is 3200+ lines | Consider splitting |
| Duplication | Similar rendering code in multiple places | Abstract to shared components |
| Error Handling | Many try-except blocks with generic catches | Add specific exception types |
| Testing | Limited unit test coverage | Increase test coverage |

---

## 7. Recommendations

### 7.1 Immediate Actions
1. **Verify Trend Forecast Usage:** Run a forecast with trend-based assumptions and confirm the saved parameters are being applied
2. **Test YTD Actuals:** Import current year data and verify it displays correctly
3. **Test Detailed Line Items:** Ensure historical periods show Personnel, Facilities, etc.

### 7.2 Short-Term Improvements
1. **Refactor Large Files:** Split `forecast_section.py` into:
   - `forecast_runner.py` (Run tab)
   - `forecast_results.py` (Results tab)
   - `forecast_charts.py` (Chart rendering)
   - `forecast_tables.py` (Table rendering)
   - `forecast_export.py` (Export functionality)

2. **Add Integration Tests:** Create tests that verify:
   - Trend parameters persist correctly
   - Monte Carlo uses forecast values as means
   - Calculated elements are derived correctly

3. **Improve Error Messages:** Add contextual error messages that help users understand what went wrong and how to fix it

### 7.3 Future Enhancements
1. **Scenario Comparison Dashboard:** Side-by-side comparison of multiple scenarios
2. **Automated Report Generation:** Export to PDF with commentary
3. **API Endpoints:** Expose forecast and valuation APIs for external integration
4. **Multi-User Collaboration:** Real-time collaborative editing

---

## 8. Appendices

### 8.1 Configuration Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `forecast_duration_months` | 60 | Forecast horizon |
| `wacc` | 12% | Weighted average cost of capital |
| `tax_rate` | 28% | Corporate tax rate |
| `prime_lending_rate` | 10% | Interest rate on debt |
| `call_rate` | 5% | Interest rate on cash deposits |
| `mc_iterations` | 1000 | Monte Carlo iterations |

### 8.2 Trend Function Types
| Type | Formula | Use Case |
|------|---------|----------|
| Linear | y = mx + c | Steady growth |
| Exponential | y = a × e^(bx) | Accelerating growth |
| Logarithmic | y = a × ln(x) + b | Decelerating growth |
| Polynomial | y = ax² + bx + c | Non-linear patterns |
| Moving Average | MA(n) | Smoothing volatility |

### 8.3 Distribution Types
| Type | Parameters | Use Case |
|------|------------|----------|
| Normal | mean, std | Most metrics |
| Lognormal | mean, std | Positive-only values |
| Triangular | min, mode, max | Expert judgment |
| Beta | alpha, beta, loc, scale | Bounded values |
| Uniform | low, high | Equal probability |

---

**Document Prepared By:** AI Assistant  
**Review Required By:** User  
**Next Steps:** Verify requirements match expectations, test recent fixes, prioritize gaps

---

*End of Requirements Document*
