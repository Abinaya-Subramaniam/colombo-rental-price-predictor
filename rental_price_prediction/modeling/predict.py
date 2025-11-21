import pandas as pd
import joblib
from ..config import *

class RentalPricePredictor:
    def __init__(self, model_path=None, preprocessor_path=None):
        if model_path is None:
            model_path = MODEL_FILE
        if preprocessor_path is None:
            preprocessor_path = PREPROCESSOR_FILE
            
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
    def predict(self, X):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
            
        return self.model.predict(X)
    
    def predict_single(self, property_features):
        prediction = self.predict(property_features)
        return prediction[0] if len(prediction) == 1 else prediction
    
    def get_feature_importance(self):
        try:
            return joblib.load(FEATURE_IMPORTANCE_FILE)
        except:
            return None

def load_predictor():
    return RentalPricePredictor()

if __name__ == "__main__":
    predictor = load_predictor()
    
    sample_property = {
        'Size_in_Sqft': 1000,
        'Bedrooms': 3,
        'Bathrooms': 2,
        'Property_Age': 5,
        'Distance_to_City_Center': 2.5,
        'Neighborhood': 'Uptown',
        'Furnished': 'Furnished',
        'Building_Type': 'Apartment'
    }
    
    prediction = predictor.predict_single(sample_property)
    print(f"Predicted rental price: ${prediction:.2f}")