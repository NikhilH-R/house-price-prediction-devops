"""
House Price Prediction API
Full-Stack Machine Learning Application

Provides REST API endpoints for house price prediction
using a trained Random Forest regression model.
"""

import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
API_VERSION = "1.1.0"

FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]


# ==================================================
# 2. MODEL LOADING AND APPLICATION LIFECYCLE
# ==================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model during application startup."""

    app.state.model = None

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        # Only load trusted, locally generated model files.
        model = joblib.load(MODEL_PATH)

        if not callable(getattr(model, "predict", None)):
            raise TypeError(
                "Loaded model does not support prediction"
            )

        app.state.model = model

        logger.info(
            "Machine learning model loaded successfully"
        )

    except Exception:
        logger.exception(
            "Failed to load the machine learning model"
        )

    try:
        yield
    finally:
        app.state.model = None
        logger.info("Application shutdown completed")


# ==================================================
# 3. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="House Price Prediction API",
    description=(
        "A machine learning REST API for house price "
        "prediction using a trained regression model."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


# ==================================================
# 4. CORS CONFIGURATION
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ==================================================
# 5. INPUT VALIDATION SCHEMA
# ==================================================

class HouseInput(BaseModel):
    """Validated input features for house price prediction."""

    MedInc: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False,
        description="Median household income",
    )

    HouseAge: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False,
        description="Median house age",
    )

    AveRooms: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        description="Average number of rooms",
    )

    AveBedrms: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False,
        description="Average number of bedrooms",
    )

    Population: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        description="Population of the area",
    )

    AveOccup: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        description="Average household occupancy",
    )

    Latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        allow_inf_nan=False,
        description="Geographical latitude",
    )

    Longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        allow_inf_nan=False,
        description="Geographical longitude",
    )


# ==================================================
# 6. RESPONSE SCHEMAS
# ==================================================

class PredictionResponse(BaseModel):
    """Structured response for successful predictions."""

    predicted_price_usd: float
    message: str


class HealthResponse(BaseModel):
    """Structured application health response."""

    status: Literal["healthy"]
    model_loaded: bool


# ==================================================
# 7. ROOT ENDPOINT
# ==================================================

@app.get(
    "/",
    tags=["General"],
)
def home():
    """Return application information."""

    return {
        "message": "House Price Prediction API is running",
        "version": API_VERSION,
        "documentation": "/docs",
        "health_check": "/health",
    }


# ==================================================
# 8. HEALTH CHECK ENDPOINT
# ==================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"],
)
def health_check(request: Request):
    """Verify application and model availability."""

    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable",
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
    )


# ==================================================
# 9. HOUSE PRICE PREDICTION ENDPOINT
# ==================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict_price(
    data: HouseInput,
    request: Request,
):
    """Generate a house price prediction."""

    model = getattr(request.app.state, "model", None)

    if model is None:
        logger.warning(
            "Prediction requested while model is unavailable"
        )

        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable",
        )

    try:
        # Convert validated input into a DataFrame.
        input_data = pd.DataFrame(
            [data.model_dump()],
            columns=FEATURE_COLUMNS,
        )

        # Generate machine learning prediction.
        predicted_price = float(
            model.predict(input_data)[0]
        )

        # Validate the prediction result.
        if not math.isfinite(predicted_price):
            raise ValueError(
                "Model returned a non-finite prediction"
            )

        if predicted_price < 0:
            raise ValueError(
                "Model returned a negative prediction"
            )

        # Preserve the original model output scaling.
        # Currency conversion must match train_model.py.
        predicted_price = round(predicted_price, 2)

        logger.info(
            "House price prediction completed successfully"
        )

        return PredictionResponse(
            predicted_price_usd=predicted_price,
            message="Prediction completed successfully",
        )

    except Exception:
        logger.exception(
            "House price prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to complete the prediction",
        )


# ==================================================
# END OF APPLICATION
# ==================================================
