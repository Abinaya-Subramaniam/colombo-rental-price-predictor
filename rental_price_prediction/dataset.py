import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .config import *

def load_raw_data():
    return pd.read_csv(RAW_DATA_FILE)

def clean_data(df):
    df_clean = df.copy()
    
    df_clean = df_clean.drop(columns=['Property_ID'], errors='ignore')
    
    text_columns = ['Neighborhood', 'Building_Type', 'Furnished']
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].str.strip().str.title()
    
    if 'Bedrooms' in df_clean.columns:
        df_clean = df_clean[df_clean['Bedrooms'] != 200] 
    
    if 'Rental_Price' in df_clean.columns:
        df_clean = df_clean[df_clean['Rental_Price'] < df_clean['Rental_Price'].quantile(0.99)]
    
    if 'Size_in_Sqft' in df_clean.columns:
        df_clean = df_clean[df_clean['Size_in_Sqft'] < df_clean['Size_in_Sqft'].quantile(0.99)]
    
    return df_clean

def save_processed_data(df, filename=None):
    if filename is None:
        filename = PROCESSED_DATA_FILE
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Processed data saved to {filename}")

def prepare_train_test_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    train_data = X_train.copy()
    train_data[TARGET] = y_train
    test_data = X_test.copy()
    test_data[TARGET] = y_test
    
    save_processed_data(train_data, TRAIN_DATA_FILE)
    save_processed_data(test_data, TEST_DATA_FILE)
    
    return X_train, X_test, y_train, y_test