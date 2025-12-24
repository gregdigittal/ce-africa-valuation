# CE Africa Valuation Platform - User Manual

**Version:** 2.0  
**Last Updated:** December 17, 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Workflow Overview](#workflow-overview)
4. [Features Guide](#features-guide)
5. [Troubleshooting](#troubleshooting)

---

## Introduction

The CE Africa Valuation Platform is a comprehensive financial modeling tool designed for mining services companies. It enables you to:

- Model installed base revenue from machine fleets
- Analyze historical data with AI-powered assumptions
- Forecast financial performance
- Explore what-if scenarios
- Model manufacturing strategies
- Analyze funding and returns

---

## Getting Started

### Launching the Application

1. Ensure you have Python 3.11+ installed
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your Supabase credentials in `secrets.toml`
4. Launch: `streamlit run app_refactored.py`

### First Steps

1. **Create a Scenario**: Click "New Scenario" in the sidebar
2. **Complete Setup**: Follow the workflow to configure your scenario
3. **Import Data**: Upload machine fleet and historical financial data
4. **Run Forecast**: Generate your financial forecast

---

## Workflow Overview

The platform follows a linear workflow:

### 1. Setup
- Configure scenario basics
- Import machine fleet data
- Import historical financials
- Set manual assumptions

### 2. AI Analysis (Optional)
- Run AI analysis on historical data
- Review AI-derived assumptions
- Accept or modify assumptions
- Save assumptions

### 3. Manufacturing Strategy (Optional)
- Configure make vs. buy decisions
- Set manufacturing parameters
- Save strategy

### 4. Forecast
- Run financial forecast
- Review results
- Save snapshots
- Export reports

### 5. What-If Analysis
- Adjust key parameters
- See real-time impact
- Run sensitivity analysis
- Compare scenarios

### 6. Funding & Returns
- Configure funding structure
- Analyze IRR
- Model financing scenarios

---

## Features Guide

### Setup Wizard

**Purpose**: Configure your scenario and import data

**Steps**:
1. Enter scenario name and description
2. Import machine fleet (CSV or manual entry)
3. Import historical financials
4. Configure basic assumptions (WACC, inflation, margins)

**Tips**:
- Use CSV import for large datasets
- Historical data should include at least 12 months
- Assumptions can be refined later with AI analysis

### AI Assumptions Engine

**Purpose**: Derive assumptions from historical data

**Features**:
- Automatic analysis of historical trends
- Probability distribution fitting
- Manufacturing assumption recommendations

**Workflow**:
1. Run Analysis
2. Review Financial Assumptions tab
3. Review Manufacturing Assumptions tab
4. Accept or modify assumptions
5. Save to commit

**Note**: Analysis results auto-save, so you can return later without rerunning.

### Forecast Section

**Purpose**: Generate financial forecasts

**Tabs**:
- **Run Forecast**: Execute forecast calculation
- **Income Statement**: View P&L projections
- **Balance Sheet**: View balance sheet projections
- **Cash Flow**: View cash flow projections
- **Export**: Export to PDF/Excel
- **Commentary**: View forecast commentary

**Features**:
- Real-time calculation
- Manufacturing strategy integration
- Snapshot saving
- Monte Carlo simulation (optional)

### What-If Agent

**Purpose**: Explore different scenarios

**Features**:
- Real-time parameter adjustment
- Side-by-side comparison
- Sensitivity analysis
- Tornado diagrams

**Usage**:
1. Adjust sliders (Revenue, Utilization, COGS, OPEX)
2. View immediate impact on metrics
3. Run sensitivity analysis
4. Save scenarios

### Funding & Returns

**Purpose**: Model financing and calculate returns

**Tabs**:
- **Funding Structure**: Configure debt and equity
- **Overdraft**: Set up overdraft facilities
- **Trade Finance**: Configure trade finance (if available)
- **IRR Analysis**: Calculate equity and project IRR

---

## Troubleshooting

### Common Issues

**"No assumptions configured"**
- Complete Setup step first
- Ensure assumptions are saved

**"No machines found in fleet"**
- Import machine data in Setup
- Check that machines are marked as "Active"

**"Forecast results not persisting"**
- Results are saved in snapshots
- Check Forecast Snapshots tab
- Ensure you're using the same scenario

**"AI Analysis not saving"**
- Analysis auto-saves when complete
- Explicit save is for final assumptions
- Check database connection

### Getting Help

- Check the Troubleshooting Guide in `docs/TROUBLESHOOTING.md`
- Review component-specific help (ℹ️ icons)
- Check workflow guidance messages

---

## Best Practices

1. **Save Frequently**: Use snapshots to save forecast versions
2. **Use AI Analysis**: Leverage AI for better assumptions
3. **Validate Data**: Check imported data for accuracy
4. **Document Scenarios**: Use descriptive scenario names
5. **Compare Scenarios**: Use What-If Agent to explore options

---

## Support

For technical issues or questions, refer to:
- Developer Guide: `docs/DEVELOPER_GUIDE.md`
- API Documentation: `docs/API_DOCUMENTATION.md`
- Database Schema: `docs/DATABASE_SCHEMA.md`
