# Model Performance Comparison

## Overview
This document provides a detailed comparison of various machine learning models tested for predicting rental prices in Colombo, Sri Lanka.

## Evaluation Metrics
- **R² Score**: Coefficient of determination (higher is better)
- **RMSE**: Root Mean Square Error in LKR (lower is better)
- **MAE**: Mean Absolute Error in LKR (lower is better)
- **Training Time**: Time required for model training

## Model Performance Summary

| Model | R² Score | RMSE | MAE | Training Time | Robustness | Interpretability |
|-------|----------|------|-----|---------------|------------|-----------------|
| Ensemble (TheilSen + PassiveAggressive) | **0.8567** | **537.82** | **404.21** | 45s | **High** | Medium |
| TheilSen Regressor | 0.8544 | 539.89 | 406.44 | 38s | **Very High** | Medium |
| PassiveAggressive Regressor | 0.8533 | 541.23 | 408.12 | 12s | High | Low |
| ExtraTrees Regressor | 0.8489 | 548.76 | 415.33 | 28s | Medium | Medium |
| Huber Regressor | 0.8456 | 553.91 | 420.67 | 8s | **Very High** | Low |
| RANSAC Regressor | 0.8421 | 559.45 | 425.89 | 22s | **Very High** | Low |
| Bagging Regressor | 0.8398 | 562.34 | 429.12 | 35s | Medium | Low |

## Detailed Analysis

### 1. Ensemble Model (Best Performing)
**Architecture**: Voting ensemble of TheilSen and PassiveAggressive regressors
**Strengths**:
- Combines robustness of TheilSen with adaptability of PassiveAggressive
- Handles outliers effectively
- Good generalization performance
**Weaknesses**:
- Longer training time
- More complex deployment

### 2. TheilSen Regressor
**Algorithm**: Median-based estimator, robust to outliers
**Strengths**:
- Excellent outlier resistance (up to 29.3% breakdown point)
- Stable performance across different data subsets
- Good interpretability through robust statistics
**Weaknesses**:
- Computationally intensive for large datasets
- Slower convergence

### 3. PassiveAggressive Regressor
**Algorithm**: Online learning with margin-based updates
**Strengths**:
- Fast training time
- Adapts well to data patterns
- Good for streaming data scenarios
**Weaknesses**:
- Sensitive to learning rate parameters
- Less interpretable

### 4. Robust Regression Models Comparison

| Model | Outlier Resistance | Computational Cost | Parameter Sensitivity |
|-------|-------------------|-------------------|---------------------|
| TheilSen | **Very High** | High | Low |
| Huber | High | Low | Medium |
| RANSAC | **Very High** | Medium | High |


### Conclusion
The ensemble model provides statistically significant improvement over individual models while maintaining robustness properties crucial for real estate pricing.

