from fastapi import FastAPI
from pydantic import BaseModel
from rental_price_prediction.modeling.predict import RentalPricePredictor

app = FastAPI(title="Colombo Rental Price API")

predictor = RentalPricePredictor()  

class PropertyFeatures(BaseModel):
    Size_in_Sqft: int
    Bedrooms: int
    Bathrooms: int
    Property_Age: int
    Distance_to_City_Center: float
    Neighborhood: str
    Furnished: str
    Building_Type: str

@app.post("/predict_rental_price")
def predict_price(data: PropertyFeatures):
    features_dict = data.dict()
    predicted_price = predictor.predict_single(features_dict)
    return {"Predicted_Rental_Price": predicted_price}
