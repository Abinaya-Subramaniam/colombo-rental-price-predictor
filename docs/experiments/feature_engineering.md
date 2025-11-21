# Feature Engineering Process

## Overview
This document details the feature selection, transformation, and engineering process for the Colombo rental price prediction model.

## Initial Feature Set

### Raw Features
| Feature | Type | Description | Preprocessing Required |
|---------|------|-------------|---------------------|
| Property_ID | Identifier | Unique property ID | Dropped (leakage risk) |
| Size_in_Sqft | Numerical | Property area | Outlier handling, scaling |
| Bedrooms | Numerical | Number of bedrooms | Type conversion, outlier removal |
| Bathrooms | Numerical | Number of bathrooms | Outlier handling |
| Neighborhood | Categorical | Location area | Text standardization, encoding |
| Furnished | Categorical | Furniture status | Binary encoding |
| Building_Type | Categorical | Property type | Text standardization, encoding |
| Rental_Price | Target | Monthly rent | Outlier removal, validation |
| Property_Age | Numerical | Age in years | Direct use |
| Distance_to_City_Center | Numerical | Distance in km | Direct use |

## Data Cleaning Pipeline

### 1. Missing Value Treatment
```python
- Bedrooms: MCAR (p=0.215) → Median imputation
- Building_Type: MCAR (p=0.367) → Mode imputation
```

### 2. Outlier Handling Strategy
```python
- Bedrooms: 200 (data entry error) → Removal
- Rental_Price: Top 1% → Quantile-based filtering (99th percentile)
- Size_in_Sqft: Top 1% → Quantile-based filtering (99th percentile)
```

### 3. Text Standardization
```python
- Strip whitespace
- Title case conversion
- Value consistency check
```

## Feature Transformations

### Numerical Features
| Feature | Transformation | Rationale |
|---------|---------------|-----------|
| Size_in_Sqft | StandardScaler | Normalize large values |
| Bedrooms | StandardScaler | Standardize ordinal feature |
| Bathrooms | StandardScaler | Standardize ordinal feature |
| Property_Age | StandardScaler | Normalize temporal feature |
| Distance_to_City_Center | StandardScaler | Normalize spatial feature |

### Categorical Features
| Feature | Encoding | Rationale |
|---------|----------|-----------|
| Neighborhood | One-Hot (drop-first) | Avoid dummy variable trap |
| Furnished | One-Hot (drop-first) | Binary categorical |
| Building_Type | One-Hot (drop-first) | Nominal categories |

## Feature Importance Analysis

### SHAP Analysis Results
| Feature | Mean SHAP Value | Importance Rank | Business Interpretation |
|---------|----------------|-----------------|------------------------|
| Size_in_Sqft | 0.342 | 1 | Property size drives 34.2% of price variation |
| Bedrooms | 0.198 | 2 | Each bedroom adds significant value |
| Distance_to_City_Center | 0.156 | 3 | Central locations command premium |
| Neighborhood_Downtown | 0.089 | 4 | Downtown premium effect |
| Bathrooms | 0.078 | 5 | Additional bathrooms increase value |
| Building_Type_Apartment | 0.065 | 6 | Apartment type preference |
| Furnished_Furnished | 0.042 | 7 | Furnishing adds moderate value |
| Property_Age | 0.030 | 8 | Newer properties slightly preferred |