# Model Documentation

## Architecture Overview

### Preprocessing Pipeline
1. **Numerical Features**: Median imputation + StandardScaler
2. **Categorical Features**: Mode imputation + OneHotEncoding

### Model Ensemble
- **Theil-Sen Regressor**: Robust median-based estimator
- **Passive Aggressive Regressor**: Online learning with regularization
- **Voting Ensemble**: Combines both models

### Performance Metrics
- **R² Score**: 0.8567 (Ensemble)
- **RMSE**: 537.82 LKR
- **MAE**: 404.21 LKR

## Feature Importance
1. Property Size (Size_in_Sqft)
2. Number of Bedrooms
3. Distance to City Center
4. Neighborhood (Downtown/Suburbs)
5. Building Type