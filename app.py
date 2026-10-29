import os
import json
from typing import Annotated, List, Optional

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="CMAPSS Turbofan Engine RUL Predictor",
    description="FastAPI microservice to predict the Remaining Useful Life (RUL) of turbofan engines.",
    version="1.2.0"
)

# Allow the dashboard page (a different origin) to call this API.
# "*" is fine for a local demo; restrict to your page's URL if you host the API publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.json")
MODEL_INFO_PATH = os.path.join(BASE_DIR, "model_info.json")
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "feature_cols.json")

from features import build_features

best_model = xgb.Booster()
best_model.load_model(MODEL_PATH)

with open(MODEL_INFO_PATH, "r") as f:
    model_info = json.load(f)

with open(FEATURE_COLS_PATH, "r") as f:
    expected_features = json.load(f)

MAX_READINGS = 200

# A float that rejects NaN and infinity
Num = Annotated[float, Field(allow_inf_nan=False)]


class CycleReading(BaseModel):
    cycle: int
    op1: Num
    op2: Num
    s2: Num
    s3: Num
    s4: Num
    s6: Num
    s7: Num
    s8: Num
    s9: Num
    s11: Num
    s12: Num
    s13: Num
    s14: Num
    s15: Num
    s17: Num
    s20: Num
    s21: Num


class PredictionRequest(BaseModel):
    engine_id: str
    readings: List[CycleReading]


class PredictionResponse(BaseModel):
    engine_id: str
    predicted_rul: float
    risk_level: str
    n_cycles_used: int
    model_version: int
    warning: Optional[str]


class BatchPredictionRequest(BaseModel):
    engines: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_version": int(model_info.get("model_version", 2)),
        "n_features": int(model_info.get("num_features", 102))
    }


@app.get("/model-info")
def get_model_info():
    return model_info


def make_prediction(engine_id: str, readings_list: List[CycleReading]) -> PredictionResponse:
    n_cycles = len(readings_list)
    if n_cycles > MAX_READINGS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many readings: {n_cycles}. Maximum is {MAX_READINGS} cycles per engine."
        )

    readings_df = (
        pd.DataFrame([r.model_dump() for r in readings_list])
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    engineered_row = build_features(readings_df)
    dmat = xgb.DMatrix(engineered_row[expected_features])

    pred_val = float(best_model.predict(dmat)[0])
    clipped_rul = max(0.0, min(125.0, pred_val))  # clip RUL to [0, 125]

    if clipped_rul < 30:
        risk = "high"
    elif clipped_rul < 60:
        risk = "medium"
    else:
        risk = "low"

    warning_msg = None
    if n_cycles < 10:
        warning_msg = "short history, rolling features less reliable"

    return PredictionResponse(
        engine_id=engine_id,
        predicted_rul=clipped_rul,
        risk_level=risk,
        n_cycles_used=n_cycles,
        model_version=int(model_info.get("model_version", 2)),
        warning=warning_msg
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_single(request: PredictionRequest):
    if not request.readings:
        raise HTTPException(status_code=400, detail="Cycle readings list cannot be empty")
    return make_prediction(request.engine_id, request.readings)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    results = []
    for eng_req in request.engines:
        if not eng_req.readings:
            raise HTTPException(status_code=400, detail="Readings list cannot be empty")
        results.append(make_prediction(eng_req.engine_id, eng_req.readings))
    return BatchPredictionResponse(predictions=results)