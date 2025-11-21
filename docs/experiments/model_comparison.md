# Model Performance Comparison

## Overview
This document provides a detailed comparison of various machine learning models tested for predicting rental prices in Colombo, Sri Lanka. The analysis follows a systematic approach: baseline model evaluation, ensemble construction, and hyperparameter optimization.

## Evaluation Metrics
- **R² Score**: Coefficient of determination (higher is better, range: 0-1)
- **RMSE**: Root Mean Square Error in LKR (lower is better)
- **MAE**: Mean Absolute Error in LKR (lower is better)

---

## Phase 1: Baseline Model Performance

### Initial Model Comparison

| Model | R² Score | RMSE (LKR) | MAE (LKR) | Robustness | Interpretability |
|-------|----------|------------|-----------|------------|------------------|
| **Huber Regressor** | **0.8265** | **589** | **462** | Very High | Low |
| **RANSAC Regressor** | **0.7998** | **633** | **501** | Very High | Low |
| **Random Forest** | **0.6908** | **787** | **597** | Medium | Medium |
| Bagging Regressor | 0.6531 | 834 | 657 | Medium | Low |
| XGBoost | 0.6078 | 886 | 684 | Low | Medium |
| Gradient Boosting | 0.4244 | 1074 | 828 | Low | Medium |
| Decision Tree | 0.1830 | 1279 | 1012 | Very Low | High |

### Key Observations

**Top 3 Models:**
1. **Huber Regressor**: R² = 0.8265, RMSE = 589 LKR
2. **RANSAC Regressor**: R² = 0.7998, RMSE = 633 LKR
3. **Random Forest**: R² = 0.6908, RMSE = 787 LKR

**Insights:**
- Robust regression methods (Huber, RANSAC) significantly outperform tree-based ensembles
- Gradient Boosting and XGBoost underperform, likely due to outlier sensitivity
- Decision Tree shows severe overfitting with lowest R² score

---

## Phase 2: Ensemble Model Construction

### Ensemble Strategy Exploration

We tested three ensemble configurations combining top-performing models:

| Ensemble Configuration | R² Score | RMSE (LKR) | Components |
|------------------------|----------|------------|------------|
| Equal Weights | 0.8176 | 604 | Huber (50%) + RANSAC (50%) |
| **Huber Favored** | **0.8216** | **598** | Huber (weighted higher) |
| With Random Forest | 0.8209 | 599 | Huber + RANSAC + RF |

### Ensemble Analysis

**Best Configuration: Huber Favored**
- R² Score: 0.8216
- RMSE: 598 LKR
- Strategy: Weighted voting with emphasis on Huber's predictions

**Why Ensembles Work:**
- Combines Huber's outlier resistance with RANSAC's consensus approach
- Reduces individual model weaknesses through aggregation
- Provides more stable predictions across diverse property types

---

## Phase 3: Hyperparameter Optimization

### Optimal Ensemble Weights

After systematic grid search and cross-validation:

| Configuration | Weights | R² Score | RMSE (LKR) | Improvement |
|---------------|---------|----------|------------|-------------|
| Equal Weights | [1, 1] | 0.8216 | 598 | Baseline |
| **Optimized** | **[3, 1]** | **0.8232** | **595** | **+0.2%** |

**Optimal Weights: [3:1]**
- Huber Regressor: 75% weight
- RANSAC Regressor: 25% weight

### Final Performance Comparison

```
Model Performance Summary:
┌──────────────────┬──────────┬─────────────┐
│ Model            │ R² Score │ RMSE (LKR)  │
├──────────────────┼──────────┼─────────────┤
│ Huber (Solo)     │ 0.8265   │ 589         │
│ RANSAC (Solo)    │ 0.7998   │ 633         │
│ Optimized        │ 0.8232   │ 595         │
│ Ensemble         │          │             │
└──────────────────┴──────────┴─────────────┘
```

**Key Finding:**
The optimized ensemble achieves comparable performance to the single best model (Huber) while providing:
- Better generalization (lower overfitting risk)
- More stable predictions across market segments
- Resilience to individual model failure

---

## Detailed Model Analysis

### 1. Huber Regressor (Best Single Model)
**Algorithm**: L2-regularized loss with linear tail for outliers

**Strengths:**
- Excellent outlier resistance through adaptive loss function
- Balances MSE (normal data) and MAE (outliers)
- Computationally efficient
- Strong theoretical foundations

**Weaknesses:**
- Less interpretable than linear models
- Requires epsilon parameter tuning
- May underperform on extreme outliers

**Best Use Cases:**
- Datasets with moderate outliers
- Real-time prediction requirements
- When training speed matters

### 2. RANSAC Regressor
**Algorithm**: Random Sample Consensus - iterative outlier detection

**Strengths:**
- Extremely robust to outliers (up to 50% contamination)
- Model-agnostic (works with any base estimator)
- No assumptions about outlier distribution

**Weaknesses:**
- Non-deterministic results (random sampling)
- Slower convergence
- Sensitive to minimum sample parameter

**Best Use Cases:**
- Highly contaminated datasets
- When outlier patterns are unknown
- Secondary validation model


## Lessons Learned

### Why Tree-Based Methods Underperformed
1. **Outlier Sensitivity**: Extreme rental prices created deep splits
2. **Feature Space**: Linear relationships dominate in rental pricing
3. **Dataset Size**: Limited samples reduce ensemble effectiveness
4. **Overfitting**: Complex models memorized noise in training data

