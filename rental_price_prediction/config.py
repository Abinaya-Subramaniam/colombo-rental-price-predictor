import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_FILE = RAW_DATA_DIR / "colombo_rental_dataset_v1.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_rental_data.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test_data.csv"

MODEL_FILE = MODELS_DIR / "best_rental_price_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
FEATURE_IMPORTANCE_FILE = MODELS_DIR / "feature_importance.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERICAL_FEATURES = [
    'Size_in_Sqft', 
    'Bedrooms', 
    'Bathrooms', 
    'Property_Age', 
    'Distance_to_City_Center'
]

CATEGORICAL_FEATURES = [
    'Neighborhood', 
    'Furnished', 
    'Building_Type'
]

TARGET = 'Rental_Price'

IMPUTER_STRATEGY_NUM = 'median'
IMPUTER_STRATEGY_CAT = 'most_frequent'

MODEL_PARAMS = {
    'theilsen': {
        'max_subpopulation': 1000,
        'n_subsamples': None,
        'max_iter': 300,
        'random_state': RANDOM_STATE
    },
    'passive_aggressive': {
        'C': 1.0,
        'epsilon': 0.1,
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    }
}