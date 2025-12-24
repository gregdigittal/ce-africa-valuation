# Sprint Plan: AI Assumptions Engine Enhancements
**Sprint:** Pre-Deployment AI Assumptions Improvements  
**Duration:** 2-3 weeks  
**Priority:** High (Pre-Deployment)

---

## 🎯 Sprint Objectives

1. Enable correlation between assumptions (e.g., COGS correlated with Revenue)
2. AI-driven identification of calculated relationships (e.g., COGS = Revenue × (1 - Gross Profit %))
3. Fit distributions on percentage drivers (e.g., Gross Profit %) instead of absolute values where appropriate
4. Use most granular data available (monthly > annual) for better statistical fitting

---

## 📋 User Stories

### US-1: Assumption Correlation
**As a** financial modeler  
**I want** to correlate assumptions with other assumptions  
**So that** I can model relationships like "COGS is a percentage of Revenue" or "OPEX grows with Revenue"

**Acceptance Criteria:**
- [ ] User can select a source assumption for correlation
- [ ] AI suggests correlations based on historical data
- [ ] Correlation can be fixed percentage or curve-based
- [ ] Correlated assumptions update when source changes

### US-2: AI-Determined Calculated Relationships
**As a** financial modeler  
**I want** the AI to identify when line items are typically calculated from drivers  
**So that** I don't have to manually configure relationships like COGS = Revenue × (1 - GP%)

**Acceptance Criteria:**
- [ ] AI identifies common calculated relationships (COGS, Gross Profit, etc.)
- [ ] System suggests using percentage drivers instead of absolute values
- [ ] User can accept or override AI suggestions
- [ ] Calculated relationships are clearly displayed

### US-3: Percentage-Based Distribution Fitting
**As a** financial modeler  
**I want** distributions fitted on percentage drivers (e.g., Gross Profit %)  
**So that** the model scales correctly with revenue changes

**Acceptance Criteria:**
- [ ] For calculated items, fit distribution on percentage (e.g., GP% = 35% ± 5%)
- [ ] Distribution parameters are in percentage terms
- [ ] Forecast applies percentage to revenue to get absolute values
- [ ] Historical percentage analysis is shown

### US-4: Granular Data Prioritization
**As a** financial modeler  
**I want** the system to use the most granular data available  
**So that** I get better statistical fits with more data points

**Acceptance Criteria:**
- [ ] System detects monthly vs annual data availability
- [ ] Monthly data is prioritized over annual when available
- [ ] Data granularity is clearly indicated in UI
- [ ] Mixed granularity (some monthly, some annual) is handled gracefully

---

## 🔧 Technical Tasks

### Phase 1: Data Granularity Detection & Prioritization

#### Task 1.1: Enhance Data Loading to Detect Granularity
**Estimated Effort:** 1 day  
**Dependencies:** None

**Tasks:**
- [ ] Modify `load_historical_data()` to detect data frequency (monthly vs annual)
- [ ] Add `data_granularity` metadata to loaded data
- [ ] Prioritize monthly data over annual when both exist
- [ ] Handle mixed granularity scenarios

**Files to Modify:**
- `components/ai_assumptions_engine.py` - `load_historical_data()`
- `components/ai_assumptions_engine.py` - `aggregate_detailed_line_items_to_summary()`

**Acceptance Criteria:**
- Monthly data is used when available
- Annual data is used as fallback
- Granularity is logged and displayed

---

#### Task 1.2: Update UI to Show Data Granularity
**Estimated Effort:** 0.5 days  
**Dependencies:** Task 1.1

**Tasks:**
- [ ] Add granularity indicator in AI Assumptions UI
- [ ] Show data point count and frequency
- [ ] Display warning if insufficient data points

**Files to Modify:**
- `components/ai_assumptions_engine.py` - `render_analysis_tab()`

**Acceptance Criteria:**
- Users can see data granularity
- Data point count is displayed
- Warnings shown for insufficient data

---

### Phase 2: Calculated Relationship Detection

#### Task 2.1: Create Calculated Relationship Detector
**Estimated Effort:** 2 days  
**Dependencies:** None

**Tasks:**
- [ ] Create `CalculatedRelationshipDetector` class
- [ ] Define common financial relationships:
  - COGS = Revenue × (1 - Gross Profit %)
  - Gross Profit = Revenue - COGS
  - Gross Profit % = (Revenue - COGS) / Revenue
  - Operating Expenses % = OPEX / Revenue
  - EBITDA Margin = EBITDA / Revenue
- [ ] Implement relationship detection logic
- [ ] Return suggested relationships with confidence scores

**Files to Create:**
- `components/calculated_relationship_detector.py`

**Files to Modify:**
- `components/ai_assumptions_engine.py` - Integrate detector

**Acceptance Criteria:**
- Detects common calculated relationships
- Provides confidence scores
- Suggests percentage drivers where appropriate

---

#### Task 2.2: Integrate Relationship Detection into Analysis
**Estimated Effort:** 1 day  
**Dependencies:** Task 2.1

**Tasks:**
- [ ] Run relationship detector during AI analysis
- [ ] Store detected relationships in `AssumptionsSet`
- [ ] Display suggestions in UI
- [ ] Allow user to accept/reject suggestions

**Files to Modify:**
- `components/ai_assumptions_engine.py` - `analyze_all_financials()`
- `components/ai_assumptions_engine.py` - `render_analysis_tab()`

**Acceptance Criteria:**
- Relationships detected during analysis
- Suggestions shown to user
- User can accept/reject

---

### Phase 3: Percentage-Based Distribution Fitting

#### Task 3.1: Enhance Distribution Fitting for Percentages
**Estimated Effort:** 2 days  
**Dependencies:** Task 2.1

**Tasks:**
- [ ] Modify `analyze_metric()` to accept percentage mode
- [ ] Calculate percentage values (e.g., GP% = (Revenue - COGS) / Revenue)
- [ ] Fit distributions on percentage values
- [ ] Store percentage distribution parameters
- [ ] Add `is_percentage_based` flag to `Assumption` dataclass

**Files to Modify:**
- `components/ai_assumptions_engine.py` - `Assumption` dataclass
- `components/ai_assumptions_engine.py` - `analyze_metric()`
- `components/ai_assumptions_engine.py` - `HistoricalAnalyzer`

**Acceptance Criteria:**
- Distributions can be fitted on percentages
- Percentage parameters stored correctly
- Historical percentage analysis shown

---

#### Task 3.2: Update Forecast Generation for Percentage-Based Assumptions
**Estimated Effort:** 1.5 days  
**Dependencies:** Task 3.1

**Tasks:**
- [ ] Modify forecast generation to apply percentages
- [ ] For percentage-based assumptions: `value = revenue × percentage`
- [ ] Ensure percentage distributions are applied correctly
- [ ] Update Monte Carlo to handle percentage-based assumptions

**Files to Modify:**
- `components/forecast_section.py` - Forecast generation
- `components/ai_assumptions_engine.py` - `sample_from_assumptions()`

**Acceptance Criteria:**
- Percentage-based assumptions scale with revenue
- Forecast values calculated correctly
- Monte Carlo respects percentage distributions

---

### Phase 4: Assumption Correlation System

#### Task 4.1: Create Correlation Engine for Assumptions
**Estimated Effort:** 2 days  
**Dependencies:** None

**Tasks:**
- [ ] Create `AssumptionCorrelationEngine` class
- [ ] Analyze historical correlations between assumptions
- [ ] Support fixed percentage correlations
- [ ] Support curve-based correlations (linear, logarithmic, etc.)
- [ ] Calculate correlation strength (R², correlation coefficient)

**Files to Create:**
- `components/assumption_correlation_engine.py`

**Files to Modify:**
- `components/ai_assumptions_engine.py` - Integrate correlation engine

**Acceptance Criteria:**
- Correlations detected between assumptions
- Correlation strength calculated
- Multiple correlation types supported

---

#### Task 4.2: Add Correlation Configuration UI
**Estimated Effort:** 2 days  
**Dependencies:** Task 4.1

**Tasks:**
- [ ] Add correlation section to Financial Assumptions tab
- [ ] Show suggested correlations with strength indicators
- [ ] Allow user to configure correlations:
  - Select source assumption
  - Choose correlation type (fixed %, curve)
  - Set correlation parameters
- [ ] Display correlation preview/chart

**Files to Modify:**
- `components/ai_assumptions_engine.py` - `render_financial_assumptions_tab()`
- Create correlation configuration component

**Acceptance Criteria:**
- Users can see suggested correlations
- Users can configure correlations
- Correlation previews shown

---

#### Task 4.3: Update Assumption Data Model for Correlations
**Estimated Effort:** 1 day  
**Dependencies:** Task 4.1

**Tasks:**
- [ ] Add correlation fields to `Assumption` dataclass:
  - `correlation_source: Optional[str]`
  - `correlation_type: Optional[str]` (fixed_pct, curve)
  - `correlation_params: Dict[str, float]`
- [ ] Update `AssumptionsSet` to handle correlations
- [ ] Update serialization/deserialization

**Files to Modify:**
- `components/ai_assumptions_engine.py` - `Assumption` dataclass
- `components/ai_assumptions_engine.py` - `AssumptionsSet`

**Acceptance Criteria:**
- Correlations stored in assumption data
- Serialization works correctly
- Backward compatible with existing assumptions

---

#### Task 4.4: Implement Correlation Calculation in Forecast
**Estimated Effort:** 1.5 days  
**Dependencies:** Task 4.3

**Tasks:**
- [ ] Modify forecast generation to apply correlations
- [ ] For correlated assumptions: calculate from source assumption
- [ ] Handle dependency chains (A → B → C)
- [ ] Validate no circular dependencies

**Files to Modify:**
- `components/forecast_section.py` - Forecast generation
- `components/ai_assumptions_engine.py` - `sample_from_assumptions()`

**Acceptance Criteria:**
- Correlated assumptions calculated correctly
- Dependency chains work
- Circular dependencies prevented

---

### Phase 5: UI/UX Enhancements

#### Task 5.1: Enhance AI Analysis Display
**Estimated Effort:** 1 day  
**Dependencies:** Tasks 2.2, 3.1, 4.1

**Tasks:**
- [ ] Show detected relationships in analysis results
- [ ] Display percentage-based analysis separately
- [ ] Show correlation suggestions
- [ ] Add visual indicators for calculated vs. direct assumptions

**Files to Modify:**
- `components/ai_assumptions_engine.py` - `render_analysis_tab()`

**Acceptance Criteria:**
- Relationships clearly displayed
- Percentage analysis shown
- Correlations highlighted

---

#### Task 5.2: Improve Assumption Configuration UI
**Estimated Effort:** 1.5 days  
**Dependencies:** Tasks 4.2, 3.1

**Tasks:**
- [ ] Add correlation configuration to assumption editor
- [ ] Show percentage vs. absolute toggle for applicable assumptions
- [ ] Display relationship dependencies
- [ ] Add help text and tooltips

**Files to Modify:**
- `components/ai_assumptions_engine.py` - `render_financial_assumptions_tab()`
- `components/ai_assumptions_engine.py` - `render_assumption_editor()`

**Acceptance Criteria:**
- Correlation configuration intuitive
- Percentage/absolute toggle clear
- Dependencies visible

---

### Phase 6: Testing & Validation

#### Task 6.1: Unit Tests
**Estimated Effort:** 2 days  
**Dependencies:** All previous tasks

**Tasks:**
- [ ] Test relationship detection
- [ ] Test percentage distribution fitting
- [ ] Test correlation calculation
- [ ] Test data granularity prioritization
- [ ] Test forecast generation with correlations

**Files to Create:**
- `tests/test_calculated_relationship_detector.py`
- `tests/test_assumption_correlation.py`
- `tests/test_percentage_distributions.py`

**Acceptance Criteria:**
- >80% code coverage
- All tests pass
- Edge cases handled

---

#### Task 6.2: Integration Tests
**Estimated Effort:** 1.5 days  
**Dependencies:** Task 6.1

**Tasks:**
- [ ] Test end-to-end AI analysis with new features
- [ ] Test assumption correlation workflow
- [ ] Test percentage-based forecast generation
- [ ] Test with mixed granularity data

**Files to Modify:**
- `tests/test_integration.py`

**Acceptance Criteria:**
- All integration tests pass
- Real-world scenarios work

---

#### Task 6.3: User Acceptance Testing
**Estimated Effort:** 1 day  
**Dependencies:** Task 6.2

**Tasks:**
- [ ] Test with real financial data
- [ ] Validate percentage calculations
- [ ] Validate correlations work correctly
- [ ] Validate granularity prioritization

**Acceptance Criteria:**
- All user stories validated
- No critical bugs
- Performance acceptable

---

## 📊 Sprint Timeline

### Week 1: Foundation
- **Days 1-2:** Data granularity detection (Tasks 1.1, 1.2)
- **Days 3-4:** Calculated relationship detection (Tasks 2.1, 2.2)
- **Day 5:** Percentage distribution fitting (Task 3.1)

### Week 2: Core Features
- **Days 1-2:** Percentage-based forecast (Task 3.2)
- **Days 3-4:** Correlation engine (Tasks 4.1, 4.2)
- **Day 5:** Correlation data model (Task 4.3)

### Week 3: Integration & Polish
- **Days 1-2:** Correlation in forecast (Task 4.4)
- **Days 3-4:** UI enhancements (Tasks 5.1, 5.2)
- **Day 5:** Testing (Tasks 6.1, 6.2, 6.3)

---

## 🔗 Dependencies

```
Task 1.1 (Data Granularity)
    ↓
Task 1.2 (UI for Granularity)
    ↓
Task 2.1 (Relationship Detector)
    ↓
Task 2.2 (Integrate Detector) ──┐
    ↓                           │
Task 3.1 (Percentage Fitting)   │
    ↓                           │
Task 3.2 (Forecast Integration) │
    ↓                           │
Task 4.1 (Correlation Engine) ──┤
    ↓                           │
Task 4.2 (Correlation UI)        │
    ↓                           │
Task 4.3 (Data Model)            │
    ↓                           │
Task 4.4 (Forecast Correlation) ─┘
    ↓
Task 5.1 (Analysis UI)
    ↓
Task 5.2 (Config UI)
    ↓
Task 6.1 (Unit Tests)
    ↓
Task 6.2 (Integration Tests)
    ↓
Task 6.3 (UAT)
```

---

## ✅ Definition of Done

Each task is considered complete when:
- [ ] Code implemented and reviewed
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] UI tested and validated
- [ ] No regression in existing functionality

---

## 🐛 Risk Mitigation

### Risk 1: Complex Dependency Chains
**Mitigation:** Implement dependency validation early, add circular dependency detection

### Risk 2: Performance with Large Datasets
**Mitigation:** Optimize correlation calculations, add caching where appropriate

### Risk 3: Backward Compatibility
**Mitigation:** Ensure old assumptions still work, add migration if needed

### Risk 4: User Confusion with New Features
**Mitigation:** Add clear UI indicators, help text, and tooltips

---

## 📝 Technical Notes

### Calculated Relationships to Detect
1. **COGS** → Derived from Gross Profit %: `COGS = Revenue × (1 - GP%)`
2. **Gross Profit** → Derived from Revenue and COGS: `GP = Revenue - COGS`
3. **Gross Profit %** → Calculated: `GP% = (Revenue - COGS) / Revenue`
4. **Operating Expenses %** → Calculated: `OPEX% = OPEX / Revenue`
5. **EBITDA Margin** → Calculated: `EBITDA% = EBITDA / Revenue`

### Correlation Types
1. **Fixed Percentage** - `target = source × percentage`
2. **Linear** - `target = source × slope + intercept`
3. **Logarithmic** - `target = a × log(source) + b`
4. **Polynomial** - `target = a × source² + b × source + c`

### Data Granularity Priority
1. **Monthly** (highest priority) - 12+ data points per year
2. **Quarterly** - 4 data points per year
3. **Annual** (lowest priority) - 1 data point per year

---

## 🎯 Success Metrics

- [ ] All user stories completed
- [ ] >80% test coverage
- [ ] Zero critical bugs
- [ ] Performance: Analysis completes in <30 seconds
- [ ] User feedback: Features are intuitive and useful

---

**Sprint Start Date:** TBD  
**Sprint End Date:** TBD  
**Sprint Owner:** Development Team
