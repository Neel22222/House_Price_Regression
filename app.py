"""
app.py  –  FastAPI backend for House Price Prediction
------------------------------------------------------
Endpoints:
  GET  /           → health check
  GET  /model-info → model name + metrics + feature stats
  POST /predict    → predict house price for given features
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Load artefacts ────────────────────────────────────────────────────────────
MODEL_DIR = Path("model")

if not (MODEL_DIR / "model.pkl").exists():
    raise RuntimeError("model/model.pkl not found — run train_model.py first.")

model = joblib.load(MODEL_DIR / "model.pkl")

with open(MODEL_DIR / "meta.json") as f:
    meta = json.load(f)

FEATURES = meta["features"]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="House Price Prediction API",
    description="Predict residential house prices using a trained ML model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class HouseFeatures(BaseModel):
    Square_Footage:       float = Field(..., gt=0,  description="Total square footage of the house")
    Num_Bedrooms:         int   = Field(..., ge=1,  le=10, description="Number of bedrooms")
    Num_Bathrooms:        int   = Field(..., ge=1,  le=3,  description="Number of bathrooms")
    Year_Built:           int   = Field(..., ge=1800, le=2024, description="Year the house was built")
    Lot_Size:             float = Field(..., gt=0,  description="Lot size in acres")
    Garage_Size:          int   = Field(..., ge=0,  le=5,  description="Garage capacity (cars)")
    Neighborhood_Quality: int   = Field(..., ge=1,  le=10, description="Neighborhood quality score (1-10)")


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predicted_price: float
    model_used:      str
    currency:        str = "USD"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "House Price Prediction API is running."}


@app.get("/model-info", tags=["Model"])
def model_info():
    return {
        "model_name":        meta["model_name"],
        "metrics":           meta["metrics"],
        "feature_stats":     meta["feature_stats"],
        "features":          FEATURES,
        "all_model_metrics": meta.get("all_model_metrics", {}),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(house: HouseFeatures):
    try:
        input_df = pd.DataFrame([[
            house.Square_Footage,
            house.Num_Bedrooms,
            house.Num_Bathrooms,
            house.Year_Built,
            house.Lot_Size,
            house.Garage_Size,
            house.Neighborhood_Quality,
        ]], columns=FEATURES)
        price = float(model.predict(input_df)[0])
        return PredictionResponse(
            predicted_price=round(price, 2),
            model_used=meta["model_name"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
