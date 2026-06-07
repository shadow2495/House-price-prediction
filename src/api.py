"""
House Price Prediction — FastAPI Backend
Run: uvicorn src.api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
import numpy as np
import joblib
import os

# ── Load artefacts ─────────────────────────────────────────────────────────
MODEL_PATH        = "models/best_model.pkl"
SCALER_PATH       = "models/scaler.pkl"
FEAT_NAMES_PATH   = "models/feature_names.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run src/train.py first.")

model         = joblib.load(MODEL_PATH)
scaler        = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEAT_NAMES_PATH)

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="House Price Prediction API",
    description="Predicts Ames housing sale prices using an XGBoost / GBM model.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ─────────────────────────────────────────────
class HouseFeatures(BaseModel):
    OverallQual:   int   = Field(..., ge=1, le=10,   description="Overall material/finish quality (1–10)")
    GrLivArea:     float = Field(..., gt=0,           description="Above grade living area (sq ft)")
    GarageCars:    int   = Field(..., ge=0, le=4,    description="Garage capacity (cars)")
    TotalBsmtSF:   float = Field(..., ge=0,           description="Total basement area (sq ft)")
    FirstFlrSF:    float = Field(..., gt=0,           description="First floor area (sq ft)")
    FullBath:      int   = Field(..., ge=0, le=4,    description="Full bathrooms above grade")
    TotRmsAbvGrd:  int   = Field(..., ge=0,           description="Total rooms above grade")
    YearBuilt:     int   = Field(..., ge=1800, le=2024, description="Year house was built")
    YearRemodAdd:  int   = Field(..., ge=1800, le=2024, description="Remodel year (or year built)")
    LotArea:       float = Field(..., gt=0,           description="Lot size (sq ft)")
    Fireplaces:    int   = Field(0,  ge=0,            description="Number of fireplaces")
    GarageArea:    float = Field(0,  ge=0,            description="Garage area (sq ft)")
    YrSold:        int   = Field(2010, ge=2006, le=2024, description="Year sold")

    class Config:
        schema_extra = {
            "example": {
                "OverallQual": 7, "GrLivArea": 1710, "GarageCars": 2,
                "TotalBsmtSF": 856, "FirstFlrSF": 856, "FullBath": 2,
                "TotRmsAbvGrd": 8, "YearBuilt": 2003, "YearRemodAdd": 2003,
                "LotArea": 8450, "Fireplaces": 0, "GarageArea": 548, "YrSold": 2010
            }
        }


class PredictionResponse(BaseModel):
    predicted_price:    float
    predicted_price_fmt: str
    confidence_low:     float
    confidence_high:    float
    inputs_used:        dict


# ── Helper ─────────────────────────────────────────────────────────────────
def build_feature_vector(house: HouseFeatures) -> np.ndarray:
    """
    Build a full feature vector matching the training feature set.
    Unknown columns are filled with 0 (median-imputed baseline).
    """
    feat_map = {
        'OverallQual':   house.OverallQual,
        'GrLivArea':     np.log1p(house.GrLivArea),
        'GarageCars':    house.GarageCars,
        'TotalBsmtSF':   np.log1p(house.TotalBsmtSF),
        '1stFlrSF':      np.log1p(house.FirstFlrSF),
        'FullBath':      house.FullBath,
        'TotRmsAbvGrd':  house.TotRmsAbvGrd,
        'YearBuilt':     house.YearBuilt,
        'YearRemodAdd':  house.YearRemodAdd,
        'LotArea':       np.log1p(house.LotArea),
        'Fireplaces':    house.Fireplaces,
        'GarageArea':    np.log1p(house.GarageArea + 1),
        # Derived
        'TotalSF':       np.log1p(house.TotalBsmtSF + house.FirstFlrSF),
        'HouseAge':      house.YrSold - house.YearBuilt,
        'RemodAge':      house.YrSold - house.YearRemodAdd,
        'IsNew':         int(house.YearBuilt == house.YrSold),
        'HasGarage':     int(house.GarageArea > 0),
        'HasBsmt':       int(house.TotalBsmtSF > 0),
        'HasFireplace':  int(house.Fireplaces > 0),
    }
    vec = np.zeros(len(feature_names))
    for i, fname in enumerate(feature_names):
        if fname in feat_map:
            vec[i] = feat_map[fname]
    return vec


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "House Price Prediction API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": True,
            "features": len(feature_names)}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(house: HouseFeatures):
    try:
        vec     = build_feature_vector(house).reshape(1, -1)
        vec_sc  = scaler.transform(vec)
        log_pred = model.predict(vec_sc)[0]
        price    = float(np.expm1(log_pred))

        # Rough 90% confidence interval (±8% of price)
        ci_low  = round(price * 0.92, 2)
        ci_high = round(price * 1.08, 2)

        return PredictionResponse(
            predicted_price=round(price, 2),
            predicted_price_fmt=f"${price:,.0f}",
            confidence_low=ci_low,
            confidence_high=ci_high,
            inputs_used=house.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features", tags=["Info"])
def list_features():
    return {"total_features": len(feature_names),
            "feature_names": feature_names}
