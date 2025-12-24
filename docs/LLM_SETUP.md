# LLM Setup Guide

This guide covers LLM setup for:
1. **What-If Agent Natural Language Optimization** (Sprint 23)
2. **AI Account Classification** (Trial Balance Processing)

---

## Overview

The platform uses LLM (Large Language Models) for two main features:
- **What-If Agent**: Natural language queries for scenario optimization
- **Account Classification**: Automatic trial balance account mapping

Both features require an LLM API key (OpenAI or Anthropic).

---

## Setup Instructions

### Option 1: OpenAI (Recommended)

1. **Get API Key**
   - Sign up at https://platform.openai.com
   - Create an API key in your account settings

2. **Configure Secrets**
   Add to `.streamlit/secrets.toml`:
   ```toml
   [openai]
   api_key = "sk-your-openai-api-key-here"
   ```

3. **Install SDK** (if not already installed)
   ```bash
   pip install openai
   ```

### Option 2: Anthropic Claude

1. **Get API Key**
   - Sign up at https://console.anthropic.com
   - Create an API key

2. **Configure Secrets**
   Add to `.streamlit/secrets.toml`:
   ```toml
   [anthropic]
   api_key = "sk-ant-your-anthropic-api-key-here"
   ```

3. **Install SDK** (if not already installed)
   ```bash
   pip install anthropic
   ```

---

## Usage

### Natural Language Queries

You can ask questions like:

1. **Funding Optimization:**
   - "Find the optimal mix of debt, equity, overdraft, and trade finance to maximize the return to shareholders but with the limit of equity that the shareholders are prepared to give up to a private equity investor of 25%"
   - "Maximize IRR with debt less than 50% and equity capped at 30%"

2. **Forecast Optimization:**
   - "Maximize EBIT with revenue increase capped at 20%"
   - "Minimize costs while maintaining 25% margin"

3. **Complex Scenarios:**
   - "Find the best combination of revenue growth and cost reduction to maximize shareholder returns"
   - "Optimize for maximum margin with revenue between 10% and 30% growth"

---

## How It Works

1. **Query Parsing**: LLM parses your natural language query
2. **Parameter Extraction**: Extracts objectives and constraints
3. **Optimization**: Uses scipy.optimize to find optimal solution
4. **Results Display**: Shows optimal parameters and forecast results

---

## Fallback Mode

If LLM is not available, the system uses a keyword-based fallback parser that can handle:
- Basic optimization objectives
- Simple constraints (equity limits, debt limits)
- Common patterns

---

## Troubleshooting

**"LLM integration not available"**
- Check API key is in secrets.toml
- Verify SDK is installed
- Check API key is valid

**"Optimization failed"**
- Check constraints are feasible
- Verify baseline forecast exists
- Try simpler query

**"Parse query failed"**
- Try rephrasing query
- Be more specific about objectives
- Check fallback parser works

---

## AI Account Classification

The system can automatically classify trial balance accounts using AI, eliminating the need for manual mapping.

### Setup

1. Add API key to `secrets.toml` (same as above)
2. Go to **Setup → Historics → Trial Balance → Map Accounts** tab
3. Click **"🧠 Classify All with AI"**
4. Review and adjust classifications as needed

### How It Works

The AI analyzes:
- Account names and codes
- Debit/credit balance patterns
- Standard accounting principles (IFRS/GAAP)

It classifies accounts into:
- **Income Statement**: revenue, cogs, opex, depreciation, tax, other_income, other_expense
- **Balance Sheet**: current_assets, fixed_assets, intangible_assets, current_liabilities, long_term_liabilities, equity

### Benefits

- **Automatic**: No manual mapping required
- **Intelligent**: Understands accounting standards and balance patterns
- **Transparent**: Shows confidence scores and reasoning for each classification
- **Flexible**: You can still manually adjust any classification

---

## Cost Considerations

### What-If Agent
- OpenAI GPT-4: ~$0.03 per query (parsing only)
- Anthropic Claude: ~$0.015 per query
- Optimization runs locally (no LLM cost)

### Account Classification
- OpenAI GPT-4o-mini: ~$0.001-0.002 per account (recommended, very cost-effective)
- Anthropic Claude Sonnet: ~$0.002-0.003 per account
- Typical trial balance (50-100 accounts): $0.05-0.30 per classification
- Classifications are cached in session state, so re-classification is free

### Best Practices
- Use GPT-4o-mini for account classification (cheapest, still very accurate)
- Cache parsed queries and classifications
- Batch process when possible
