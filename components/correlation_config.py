"""
Correlation Configuration for Monte Carlo Simulation
=====================================================
Allows users to configure correlations between line items for more
realistic Monte Carlo simulations.

Phase 3 of Unified Configuration Backlog.
Date: December 20, 2025

Example Correlations:
- Personnel expense correlates 90% with Revenue (more revenue = more staff)
- COGS correlates 95% with Revenue (direct relationship)
- Facilities correlates 30% with Revenue (mostly fixed)
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from scipy import stats
import json


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CorrelationPair:
    """Defines correlation between two line items."""
    item_1: str  # Key of first item
    item_2: str  # Key of second item
    correlation: float  # Correlation coefficient (-1 to 1)
    description: str = ""  # Optional description
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CorrelationPair':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CorrelationConfig:
    """Complete correlation configuration for a scenario."""
    scenario_id: str
    correlations: List[CorrelationPair] = field(default_factory=list)
    
    # Common presets
    use_preset: str = 'custom'  # 'custom', 'high_correlation', 'low_correlation', 'independent'
    
    def to_dict(self) -> dict:
        return {
            'scenario_id': self.scenario_id,
            'correlations': [c.to_dict() for c in self.correlations],
            'use_preset': self.use_preset
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CorrelationConfig':
        config = cls(scenario_id=data.get('scenario_id', ''))
        config.use_preset = data.get('use_preset', 'custom')
        config.correlations = [
            CorrelationPair.from_dict(c) for c in data.get('correlations', [])
        ]
        return config
    
    def get_correlation_matrix(self, item_keys: List[str]) -> np.ndarray:
        """Build correlation matrix from configured pairs."""
        n = len(item_keys)
        matrix = np.eye(n)  # Start with identity (perfect self-correlation)
        
        key_to_idx = {k: i for i, k in enumerate(item_keys)}
        
        for pair in self.correlations:
            if pair.item_1 in key_to_idx and pair.item_2 in key_to_idx:
                i = key_to_idx[pair.item_1]
                j = key_to_idx[pair.item_2]
                matrix[i, j] = pair.correlation
                matrix[j, i] = pair.correlation  # Symmetric
        
        return matrix


# =============================================================================
# CORRELATION PRESETS
# =============================================================================

def get_preset_correlations(preset: str, line_item_keys: List[str], categories: Dict[str, str]) -> List[CorrelationPair]:
    """
    Generate correlation pairs based on preset.
    
    Args:
        preset: 'high_correlation', 'low_correlation', 'independent'
        line_item_keys: List of line item keys
        categories: Dict mapping key to category
    """
    correlations = []
    
    # Find revenue items
    revenue_keys = [k for k in line_item_keys if 'revenue' in categories.get(k, '').lower()]
    
    if preset == 'high_correlation':
        # High correlation between revenue and expenses
        for key in line_item_keys:
            cat = categories.get(key, '').lower()
            if 'cogs' in cat or 'cost of' in cat:
                for rev_key in revenue_keys:
                    correlations.append(CorrelationPair(
                        item_1=key, item_2=rev_key,
                        correlation=0.95,
                        description="COGS strongly tied to Revenue"
                    ))
            elif 'personnel' in cat or 'salary' in cat:
                for rev_key in revenue_keys:
                    correlations.append(CorrelationPair(
                        item_1=key, item_2=rev_key,
                        correlation=0.85,
                        description="Personnel grows with Revenue"
                    ))
            elif 'opex' in cat or 'operating' in cat:
                for rev_key in revenue_keys:
                    correlations.append(CorrelationPair(
                        item_1=key, item_2=rev_key,
                        correlation=0.70,
                        description="OPEX loosely tied to Revenue"
                    ))
    
    elif preset == 'low_correlation':
        # Lower correlations
        for key in line_item_keys:
            cat = categories.get(key, '').lower()
            if 'cogs' in cat or 'cost of' in cat:
                for rev_key in revenue_keys:
                    correlations.append(CorrelationPair(
                        item_1=key, item_2=rev_key,
                        correlation=0.50,
                        description="COGS partially tied to Revenue"
                    ))
            elif 'personnel' in cat or 'salary' in cat or 'opex' in cat:
                for rev_key in revenue_keys:
                    correlations.append(CorrelationPair(
                        item_1=key, item_2=rev_key,
                        correlation=0.30,
                        description="Expenses mostly independent"
                    ))
    
    # 'independent' returns empty list (no correlations)
    return correlations


# =============================================================================
# CORRELATED SAMPLING
# =============================================================================

def generate_correlated_samples(
    means: np.ndarray,
    stds: np.ndarray,
    correlation_matrix: np.ndarray,
    n_samples: int,
    distribution_types: Optional[List[str]] = None
) -> np.ndarray:
    """
    Generate correlated random samples using Cholesky decomposition.
    
    Args:
        means: Array of means for each variable (n_vars,)
        stds: Array of standard deviations (n_vars,)
        correlation_matrix: Correlation matrix (n_vars, n_vars)
        n_samples: Number of samples to generate
        distribution_types: Optional list of distribution types per variable
        
    Returns:
        Array of shape (n_samples, n_vars) with correlated samples
    """
    n_vars = len(means)
    
    # Ensure correlation matrix is positive semi-definite
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # Matrix not positive definite - use nearest PSD matrix
        eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
        eigvals = np.maximum(eigvals, 1e-6)  # Ensure positive
        correlation_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Re-normalize to correlation matrix
        d = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(d, d)
        L = np.linalg.cholesky(correlation_matrix)
    
    # Generate independent standard normal samples
    Z = np.random.standard_normal((n_samples, n_vars))
    
    # Apply correlation structure
    correlated_Z = Z @ L.T
    
    # Transform to target distributions
    samples = np.zeros((n_samples, n_vars))
    
    for i in range(n_vars):
        mean = means[i]
        std = stds[i]
        z = correlated_Z[:, i]
        
        dist_type = distribution_types[i] if distribution_types else 'normal'
        
        if dist_type == 'normal':
            samples[:, i] = mean + std * z
        
        elif dist_type == 'lognormal':
            if mean > 0:
                # Transform standard normal to lognormal
                cv = std / mean if mean != 0 else 0.15
                sigma = np.sqrt(np.log(1 + cv ** 2))
                mu = np.log(mean) - sigma ** 2 / 2
                samples[:, i] = np.exp(mu + sigma * z)
            else:
                samples[:, i] = mean
        
        elif dist_type == 'triangular':
            # Use inverse CDF of triangular with mode at mean
            p = stats.norm.cdf(z)  # Convert to uniform [0, 1]
            low = mean - 2 * std
            high = mean + 2 * std
            mode = mean
            # Inverse CDF of triangular
            fc = (mode - low) / (high - low)
            samples[:, i] = np.where(
                p < fc,
                low + np.sqrt(p * (high - low) * (mode - low)),
                high - np.sqrt((1 - p) * (high - low) * (high - mode))
            )
        
        else:  # uniform or default
            p = stats.norm.cdf(z)
            low = mean - std * np.sqrt(3)
            high = mean + std * np.sqrt(3)
            samples[:, i] = low + p * (high - low)
    
    return samples


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_correlation_config_ui(
    db,
    scenario_id: str,
    user_id: str,
    line_items: Dict[str, Any]
):
    """Render the correlation configuration UI."""
    st.markdown("### 🔗 Line Item Correlations")
    st.caption("Configure how line items move together in Monte Carlo simulation")
    
    # Load or initialize config
    config_key = f'correlation_config_{scenario_id}'
    if config_key not in st.session_state:
        st.session_state[config_key] = load_correlation_config(db, scenario_id, user_id)
    
    config: CorrelationConfig = st.session_state[config_key]
    
    # Preset selector
    col1, col2 = st.columns([2, 3])
    with col1:
        preset = st.selectbox(
            "Correlation Preset",
            options=['custom', 'high_correlation', 'low_correlation', 'independent'],
            format_func=lambda x: {
                'custom': '🎛️ Custom',
                'high_correlation': '📈 High Correlation (Typical)',
                'low_correlation': '📉 Low Correlation',
                'independent': '🔀 Independent (No Correlation)'
            }.get(x, x),
            index=['custom', 'high_correlation', 'low_correlation', 'independent'].index(config.use_preset),
            key='correlation_preset'
        )
        
        if preset != config.use_preset:
            config.use_preset = preset
            if preset != 'custom':
                # Generate preset correlations
                categories = {k: v.get('category', '') for k, v in line_items.items()}
                config.correlations = get_preset_correlations(
                    preset, list(line_items.keys()), categories
                )
    
    with col2:
        st.info("""
**Correlation Impact:**
- **High**: Line items move together (COGS ↑ when Revenue ↑)
- **Low**: Items partially independent
- **Independent**: No relationship (pure random variation)
        """)
    
    st.markdown("---")
    
    # Custom correlation editor
    if config.use_preset == 'custom':
        st.markdown("#### Add Correlation")
        
        item_keys = list(line_items.keys())
        item_names = {k: line_items[k].get('name', k) for k in item_keys}
        
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            item1 = st.selectbox(
                "Item 1",
                options=item_keys,
                format_func=lambda k: item_names.get(k, k),
                key='corr_item1'
            )
        
        with col2:
            item2 = st.selectbox(
                "Item 2",
                options=[k for k in item_keys if k != item1],
                format_func=lambda k: item_names.get(k, k),
                key='corr_item2'
            )
        
        with col3:
            corr_value = st.slider(
                "Correlation",
                min_value=-1.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key='corr_value'
            )
        
        with col4:
            if st.button("➕ Add", key='add_corr_btn'):
                new_pair = CorrelationPair(
                    item_1=item1,
                    item_2=item2,
                    correlation=corr_value,
                    description=f"{item_names.get(item1)} ↔ {item_names.get(item2)}"
                )
                config.correlations.append(new_pair)
                st.rerun()
    
    # Show current correlations
    if config.correlations:
        st.markdown("#### Current Correlations")
        
        corr_data = []
        for i, pair in enumerate(config.correlations):
            item1_name = line_items.get(pair.item_1, {}).get('name', pair.item_1)
            item2_name = line_items.get(pair.item_2, {}).get('name', pair.item_2)
            corr_data.append({
                'idx': i,
                'Item 1': item1_name,
                'Item 2': item2_name,
                'Correlation': pair.correlation,
                'Strength': '🟢 Strong' if abs(pair.correlation) > 0.7 else ('🟡 Medium' if abs(pair.correlation) > 0.3 else '🔴 Weak')
            })
        
        df = pd.DataFrame(corr_data)
        st.dataframe(
            df[['Item 1', 'Item 2', 'Correlation', 'Strength']],
            hide_index=True,
            use_container_width=True
        )
        
        # Delete button
        if config.use_preset == 'custom':
            to_delete = st.multiselect(
                "Select correlations to remove",
                options=list(range(len(config.correlations))),
                format_func=lambda i: f"{corr_data[i]['Item 1']} ↔ {corr_data[i]['Item 2']}",
                key='corrs_to_delete'
            )
            
            if to_delete and st.button("🗑️ Remove Selected", key='delete_corrs_btn'):
                config.correlations = [c for i, c in enumerate(config.correlations) if i not in to_delete]
                st.rerun()
    else:
        st.info("No correlations configured. Items will vary independently in Monte Carlo.")
    
    # Save button
    st.markdown("---")
    if st.button("💾 Save Correlation Config", type='primary'):
        if save_correlation_config(db, scenario_id, user_id, config):
            st.success("✅ Correlation configuration saved!")
            st.session_state[config_key] = config
        else:
            st.error("Failed to save configuration")


# =============================================================================
# LOAD/SAVE
# =============================================================================

def load_correlation_config(db, scenario_id: str, user_id: str) -> CorrelationConfig:
    """Load correlation config from database."""
    config = CorrelationConfig(scenario_id=scenario_id)
    
    try:
        assumptions = db.get_scenario_assumptions(scenario_id, user_id)
        if assumptions and 'correlation_config' in assumptions:
            config = CorrelationConfig.from_dict(assumptions['correlation_config'])
            config.scenario_id = scenario_id
    except Exception as e:
        pass
    
    return config


def save_correlation_config(db, scenario_id: str, user_id: str, config: CorrelationConfig) -> bool:
    """Save correlation config to database."""
    try:
        assumptions = db.get_scenario_assumptions(scenario_id, user_id) or {}
        assumptions['correlation_config'] = config.to_dict()
        return db.update_assumptions(scenario_id, user_id, assumptions)
    except Exception as e:
        return False
