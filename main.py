from fastapi import *
from pydantic import BaseModel
import pandas as pd
import pickle

# Load the trained pipeline
with open("LrModel.pkl", "rb") as f:
    model = pickle.load(f)

# Define input schema using Pydantic
class CarData(BaseModel):
    company: str
    name: str
    year: int
    fuel_type: str
    kms_driven: int

# Create FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Car Price Predictor API is running"}

@app.post("/predict")
def predict_price(data: CarData):
    data_dict = data.dict()
    data_dict['company'] = data_dict['company'].title()    # e.g. "toyota" -> "Toyota"
    data_dict['fuel_type'] = data_dict['fuel_type'].capitalize()  # e.g. "petrol" -> "Petrol"
    data_dict['name'] = data_dict['name'].title()          # If needed

    input_df = pd.DataFrame([data_dict])

    try:
        prediction = model.predict(input_df)[0]
        return {"estimated_price": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

