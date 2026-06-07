from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="House Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("house_price_model.pkl")


class HouseInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict_price(data: HouseInput):
    input_data = pd.DataFrame([{
        "MedInc": data.MedInc,
        "HouseAge": data.HouseAge,
        "AveRooms": data.AveRooms,
        "AveBedrms": data.AveBedrms,
        "Population": data.Population,
        "AveOccup": data.AveOccup,
        "Latitude": data.Latitude,
        "Longitude": data.Longitude
    }])

    predicted_price = model.predict(input_data)[0]

    return {
        "predicted_price_usd": round(predicted_price, 2),
        "message": "Prediction completed successfully"
    }