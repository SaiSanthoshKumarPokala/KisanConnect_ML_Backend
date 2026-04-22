from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
try:
    from .voice_assistant import initialize_voice_assistant, router as voice_assistant_router
except ImportError:  # Allows running with `uvicorn app:app` from the backend folder.
    from voice_assistant import initialize_voice_assistant, router as voice_assistant_router
import sklearn.compose._column_transformer as ct

# Compatibility fix for scikit-learn 1.6.1 → 1.7.2 migration
if not hasattr(ct, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"
ENV_FILE_PATH = BASE_DIR / ".env"

PRICE_MODEL_PATH = MODELS_DIR / "Price_Prediction_Gov_dataset" / "crop_price_xgboost_2026.joblib"
YIELD_MODEL_PATH = MODELS_DIR / "Yiel_Prediction" / "best_yield_model.pkl"
DISEASE_MODEL_PATH = MODELS_DIR / "Crop_Disease" / "crop_disease_model.onnx"
CROP_RECOMMENDER_PATH = MODELS_DIR / "Crop_Recommendation" / "crop_recommender.onnx"

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# This mapping is inferred from the standard crop recommendation dataset used in the
# training script and LabelEncoder's alphabetical ordering.
CROP_LABELS = {
    0: "apple",
    1: "banana",
    2: "blackgram",
    3: "chickpea",
    4: "coconut",
    5: "coffee",
    6: "cotton",
    7: "grapes",
    8: "jute",
    9: "kidneybeans",
    10: "lentil",
    11: "maize",
    12: "mango",
    13: "mothbeans",
    14: "mungbean",
    15: "muskmelon",
    16: "orange",
    17: "papaya",
    18: "pigeonpeas",
    19: "pomegranate",
    20: "rice",
    21: "watermelon",
}

DISEASE_LABELS = [
    "diseased cotton leaf",
    "diseased cotton plant",
    "fresh cotton leaf",
    "fresh cotton plant",
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf",
    "Bell_pepper leaf spot",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Raspberry leaf",
    "Soyabean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato mold leaf",
    "grape leaf",
    "grape leaf black rot",
    "Bacterial leaf blight",
    "Blast",
    "Brown spot",
    "Leaf smut",
    "Tungro",
    "New Plant Diseases Dataset(Augmented)",
]
SOIL_LABELS = ["Alluvial", "Black", "Clay", "Laterite", "Loamy", "Red", "Sandy"]

PROXY_SOIL_MAP = {
    "Alluvial": {"n": 60.0, "p": 45.0, "k": 45.0, "ph": 7.0},
    "Black": {"n": 40.0, "p": 50.0, "k": 50.0, "ph": 7.5},
    "Clay": {"n": 30.0, "p": 40.0, "k": 45.0, "ph": 7.2},
    "Laterite": {"n": 15.0, "p": 10.0, "k": 10.0, "ph": 5.0},
    "Loamy": {"n": 50.0, "p": 40.0, "k": 40.0, "ph": 6.5},
    "Red": {"n": 20.0, "p": 20.0, "k": 20.0, "ph": 6.0},
    "Sandy": {"n": 10.0, "p": 15.0, "k": 15.0, "ph": 5.5},
}

OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"


class PricePredictionRequest(BaseModel):
    stateCode: int = Field(..., ge=0)
    districtCode: int = Field(..., ge=0)
    marketCode: int = Field(..., ge=0)
    commodity: str
    variety: str
    arrivalsTonnes: float = Field(..., ge=0)
    year: int = Field(..., ge=1900)
    month: int = Field(..., ge=1, le=12)
    dayOfWeek: int = Field(..., ge=0, le=6)


class YieldPredictionRequest(BaseModel):
    stateName: str
    districtName: str
    season: str
    cropName: str
    cropType: str
    startYear: int = Field(..., ge=1900)
    area: float = Field(..., ge=0)


app = FastAPI(title="KisanConnect AI Toolkit API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(voice_assistant_router)


def _load_env_file() -> None:
    if not ENV_FILE_PATH.exists():
        return

    for raw_line in ENV_FILE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")

        if key and key not in os.environ:
            os.environ[key] = value


def _load_image_tensor(data: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB").resize((224, 224))
    except Exception as exc:  # pragma: no cover - FastAPI error path
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = (image_array - IMAGE_MEAN) / IMAGE_STD
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0)


def _extract_probability_map(raw_probability: Any) -> dict[int, float]:
    if isinstance(raw_probability, list) and raw_probability:
        first_item = raw_probability[0]
        if isinstance(first_item, dict):
            return {int(key): float(value) for key, value in first_item.items()}
    if isinstance(raw_probability, np.ndarray):
        flattened = np.asarray(raw_probability).reshape(-1)
        return {index: float(value) for index, value in enumerate(flattened)}
    return {}


def _top_predictions(probabilities: dict[int, float], labels: dict[int, str] | list[str], limit: int = 3):
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:limit]
    items = []
    for class_id, score in ranked:
        label = labels[class_id] if isinstance(labels, dict) else labels[class_id]
        items.append(
            {
                "classId": int(class_id),
                "label": label,
                "confidence": round(float(score) * 100, 2),
            }
        )
    return items


def _normalize_image_for_torch(data: bytes) -> torch.Tensor:
    image_tensor = _load_image_tensor(data)
    return torch.from_numpy(image_tensor)


def _fetch_weather(latitude: float, longitude: float) -> dict[str, float]:
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENWEATHER_API_KEY is not configured on the backend.",
        )

    query = urlencode(
        {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        }
    )
    url = f"{OPENWEATHER_API_URL}?{query}"

    try:
        with urlopen(url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(error_body)
            detail = payload.get("message") or error_body
        except json.JSONDecodeError:
            detail = error_body or str(exc)
        raise HTTPException(status_code=502, detail=f"OpenWeather error: {detail}") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail="Unable to fetch weather from OpenWeather.") from exc

    if str(payload.get("cod")) not in {"200", "200.0"} and payload.get("cod") != 200:
        raise HTTPException(
            status_code=502,
            detail=payload.get("message", "OpenWeather returned an invalid response."),
        )

    main = payload.get("main", {})
    rain_payload = payload.get("rain", {})
    weather_items = payload.get("weather", [])
    rainfall = float(rain_payload.get("1h", rain_payload.get("3h", 0.0)) or 0.0)

    return {
        "temperature": float(main.get("temp", 0.0)),
        "humidity": float(main.get("humidity", 0.0)),
        "rainfall": rainfall,
        "description": weather_items[0].get("description", "") if weather_items else "",
        "city": str(payload.get("name", "")).strip(),
    }


def _derive_soil_features(soil_type: str, weather: dict[str, float]) -> dict[str, float]:
    profile = PROXY_SOIL_MAP[soil_type]

    return {
        "nitrogen": round(profile["n"], 2),
        "phosphorus": round(profile["p"], 2),
        "potassium": round(profile["k"], 2),
        "temperature": round(weather["temperature"], 2),
        "humidity": round(weather["humidity"], 2),
        "ph": round(profile["ph"], 2),
        "rainfall": round(max(weather["rainfall"], 0.0), 2),
    }


def _run_crop_recommendation(features: dict[str, float]) -> dict[str, Any]:
    session = app.state.recommendation_session
    model_input = np.array(
        [
            [
                features["nitrogen"],
                features["phosphorus"],
                features["potassium"],
                features["temperature"],
                features["humidity"],
                features["ph"],
                features["rainfall"],
            ]
        ],
        dtype=np.float32,
    )
    input_name = session.get_inputs()[0].name
    output_label, output_probability = session.run(None, {input_name: model_input})
    probability_map = _extract_probability_map(output_probability)
    predicted_id = int(output_label[0])
    top_matches = _top_predictions(probability_map, CROP_LABELS)
    return {
        "prediction": CROP_LABELS.get(predicted_id, f"Crop class {predicted_id}"),
        "classId": predicted_id,
        "confidence": top_matches[0]["confidence"] if top_matches else None,
        "topMatches": top_matches,
    }


@app.on_event("startup")
def load_models() -> None:
    _load_env_file()
    initialize_voice_assistant(app)
    app.state.price_pipeline = joblib.load(PRICE_MODEL_PATH)
    app.state.yield_model = joblib.load(YIELD_MODEL_PATH)
    app.state.disease_session = ort.InferenceSession(str(DISEASE_MODEL_PATH), providers=["CPUExecutionProvider"])
    app.state.recommendation_session = ort.InferenceSession(
        str(CROP_RECOMMENDER_PATH),
        providers=["CPUExecutionProvider"],
    )
    app.state.soil_model = None
    soil_model_path = MODELS_DIR / "Crop_Recommendation" / "Soil_Classification" / "model.pt"
    if soil_model_path.exists():
        app.state.soil_model = torch.jit.load(str(soil_model_path), map_location="cpu").eval()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/price-prediction")
def predict_price(payload: PricePredictionRequest) -> dict[str, Any]:
    row = pd.DataFrame(
        [
            {
                "state_code": payload.stateCode,
                "district_code": payload.districtCode,
                "market_code": payload.marketCode,
                "commodity": payload.commodity.strip(),
                "variety": payload.variety.strip(),
                "arrivals_tonnes": payload.arrivalsTonnes,
                "year": payload.year,
                "month": payload.month,
                "day_of_week": payload.dayOfWeek,
            }
        ]
    )

    pipeline = app.state.price_pipeline
    encoded_input = pipeline["encoder"].transform(row)
    prediction = max(float(pipeline["model"].predict(encoded_input)[0]), 0.0)
    return {
        "prediction": round(prediction, 2),
        "unit": "INR / quintal",
        "message": f"Estimated modal price is Rs. {prediction:.2f}.",
    }


@app.post("/api/yield-prediction")
def predict_yield(payload: YieldPredictionRequest) -> dict[str, Any]:
    row = pd.DataFrame(
        [
            {
                "state_name": payload.stateName.strip(),
                "district_name": payload.districtName.strip(),
                "season": payload.season.strip(),
                "crop_name": payload.cropName.strip(),
                "crop_type": payload.cropType.strip(),
                "start_year": payload.startYear,
                "area": payload.area,
            }
        ]
    )
    prediction = float(app.state.yield_model.predict(row)[0])
    return {
        "prediction": round(prediction, 4),
        "unit": "tonnes / hectare",
        "message": f"Projected yield is {prediction:.4f} tonnes per hectare.",
    }


@app.post("/api/crop-recommendation")
async def recommend_crop(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
) -> dict[str, Any]:
    if latitude == 0.0 and longitude == 0.0:
        raise HTTPException(status_code=400, detail="GPS coordinates are required for crop recommendation.")
    if app.state.soil_model is None:
        raise HTTPException(status_code=500, detail="Soil classification model is not available.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No soil image uploaded.")

    image_tensor = _normalize_image_for_torch(image_bytes)
    with torch.inference_mode():
        logits = app.state.soil_model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    soil_index = int(np.argmax(probabilities))
    soil_type = SOIL_LABELS[soil_index]
    soil_confidence = round(float(probabilities[soil_index]) * 100, 2)
    weather = _fetch_weather(latitude, longitude)
    features = _derive_soil_features(soil_type, weather)
    recommendation = _run_crop_recommendation(features)

    return {
        **recommendation,
        "soilType": soil_type,
        "soilConfidence": soil_confidence,
        "proxyMap": PROXY_SOIL_MAP[soil_type],
        "weather": {
            "latitude": round(latitude, 6),
            "longitude": round(longitude, 6),
            "temperature": weather["temperature"],
            "humidity": weather["humidity"],
            "rainfall": weather["rainfall"],
            "description": weather.get("description", ""),
            "city": weather.get("city", ""),
        },
        "featureArray": [
            features["nitrogen"],
            features["phosphorus"],
            features["potassium"],
            features["temperature"],
            features["humidity"],
            features["ph"],
            features["rainfall"],
        ],
        "inferredFeatures": features,
        "message": f"Recommended crop: {recommendation['prediction']} based on {soil_type.lower()} soil and live OpenWeather data.",
    }


@app.post("/api/disease-prediction")
async def predict_disease(file: UploadFile = File(...)) -> dict[str, Any]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image uploaded.")

    image_tensor = _load_image_tensor(image_bytes)
    session = app.state.disease_session
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: image_tensor})[0]
    probabilities = _softmax(logits[0])
    probability_map = {index: float(score) for index, score in enumerate(probabilities)}
    top_matches = _top_predictions(probability_map, DISEASE_LABELS)
    best_match = top_matches[0]
    return {
        "prediction": best_match["label"],
        "classId": best_match["classId"],
        "confidence": best_match["confidence"],
        "topMatches": top_matches,
        "message": f"Most likely class: {best_match['label']}.",
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)
