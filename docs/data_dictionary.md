# Data Dictionary

## Raw Data Schema

| Column Name | Data Type | Description | Notes |
|-------------|-----------|-------------|-------|
| Property_ID | int64 | Unique identifier | Dropped in preprocessing |
| Size_in_Sqft | int64 | Property area in sq.ft | Key predictor |
| Bedrooms | float64 | Number of bedrooms | Converted to int |
| Bathrooms | int64 | Number of bathrooms | |
| Neighborhood | object | Location area | Categorical, one-hot encoded |
| Furnished | object | Furniture status | Binary: Furnished/Unfurnished |
| Building_Type | object | Property type | Categorical: Apartment/House/etc |
| Rental_Price | float64 | Monthly rent (LKR) | Target variable |
| Property_Age | int64 | Age in years | |
| Distance_to_City_Center | float64 | Distance in km | Important location feature |

## Data Quality Notes
- Missing values handled using median (numerical) and mode (categorical)
- Outliers removed using quantile-based filtering
- Text standardization applied to categorical variables