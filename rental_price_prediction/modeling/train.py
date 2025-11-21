import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
import time

from ..config import *
from ..features import create_preprocessor

def train_models(X_train, y_train):
    preprocessor = create_preprocessor()
    
    models = {
        "Huber": HuberRegressor(**MODEL_PARAMS['huber']),
        "RANSAC": RANSACRegressor(**MODEL_PARAMS['ransac']),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe
        print(f"Trained {name}")
    
    return trained_models

def create_ensemble_model(X_train, y_train):
    preprocessor = create_preprocessor()
    
    ensemble = Pipeline([
        ('preprocessor', preprocessor),
        ('model', VotingRegressor([
            ('huber', HuberRegressor(**MODEL_PARAMS['huber'])),
            ('ransac', RANSACRegressor(**MODEL_PARAMS['ransac']))
        ], weights=MODEL_PARAMS['ensemble_weights']))
    ])
    
    ensemble.fit(X_train, y_train)
    return ensemble

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name}:")
    print(f"  R²: {r2:.4f} | RMSE: {rmse:.0f} | MAE: {mae:.0f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def save_model(model, filename=None):
    if filename is None:
        filename = MODEL_FILE
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def run_training_pipeline():
    print("Starting training pipeline...")
    
    from ..dataset import load_raw_data, clean_data, prepare_train_test_data
    df = load_raw_data()
    df_clean = clean_data(df)
    
    X_train, X_test, y_train, y_test = prepare_train_test_data(df_clean)
    
    print("Training individual models...")
    trained_models = train_models(X_train, y_train)
    
    print("Training ensemble model...")
    ensemble_model = create_ensemble_model(X_train, y_train)
    
    results = {}
    for name, model in trained_models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)
    
    results['Ensemble'] = evaluate_model(ensemble_model, X_test, y_test, 'Ensemble')
    
    save_model(ensemble_model)
    
    preprocessor = ensemble_model.named_steps['preprocessor']
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    
    return results, ensemble_model