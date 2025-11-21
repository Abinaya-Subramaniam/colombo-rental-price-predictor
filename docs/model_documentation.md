# Model Documentation

## Architecture Overview

### Preprocessing Pipeline
1. **Numerical Features**: Median imputation + StandardScaler
2. **Categorical Features**: Mode imputation + OneHotEncoding

### Model Ensemble
- **Huber Regressor**: Robust linear regression resistant to outliers
- **RANSAC Regressor**: Random Sample Consensus for outlier rejection
- **Voting Ensemble**: Combines both models with optimized weights [3, 1]

### Performance Metrics
- **R² Score**: 0.8232 (Ensemble)
- **RMSE**: 595 LKR
- **MAE**: 477 LKR

