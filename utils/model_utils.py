"""Model loading, feature alignment, prediction, and insight utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Indian_Climate_Dataset_2024_2025.csv"
MODEL_PATH = ROOT / "rainfall_prediction_model.pkl"
SCALER_PATH = ROOT / "scaler.pkl"

RAW_FEATURES = [
    "City",
    "State",
    "Temperature_Avg (°C)",
    "Humidity (%)",
    "Rainfall (mm)",
    "Wind_Speed (km/h)",
    "AQI",
    "AQI_Category",
    "Pressure (hPa)",
    "Cloud_Cover (%)",
    "Year",
    "Month",
    "Day",
]

UI_DEFAULTS = {
    "Temperature_Avg (°C)": 28.0,
    "Humidity (%)": 72.0,
    "Rainfall (mm)": 0.0,
    "Wind_Speed (km/h)": 8.0,
    "AQI": 150,
    "Pressure (hPa)": 1010.0,
    "Cloud_Cover (%)": 55.0,
    "City": "Mumbai",
    "State": "Maharashtra",
    "AQI_Category": "Moderate",
    "Year": 2026,
    "Month": 5,
    "Day": 19,
}


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load the climate dataset and enrich it with date parts."""
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=RAW_FEATURES)

    df = pd.read_csv(DATA_PATH)
    if "Date" in df.columns:
        parsed = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df["Year"] = parsed.dt.year.fillna(pd.Timestamp.today().year).astype(int)
        df["Month"] = parsed.dt.month.fillna(pd.Timestamp.today().month).astype(int)
        df["Day"] = parsed.dt.day.fillna(pd.Timestamp.today().day).astype(int)
    return df


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[Any | None, Any | None, str | None]:
    """Load the saved model and scaler with a user-friendly error message."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, None
    except Exception as exc:  # pragma: no cover - surfaced in Streamlit UI
        return None, None, str(exc)


def get_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return clean dropdown categories from the training dataset."""
    categories: dict[str, list[str]] = {}
    for col, fallback in {
        "City": ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata"],
        "State": ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "West Bengal"],
        "AQI_Category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
    }.items():
        if col in df.columns and not df.empty:
            values = sorted(str(v) for v in df[col].dropna().unique())
            categories[col] = values or fallback
        else:
            categories[col] = fallback
    return categories


def _category_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build deterministic encoders similar to sklearn LabelEncoder ordering."""
    maps = {}
    for col, values in get_categories(df).items():
        maps[col] = {value: idx for idx, value in enumerate(sorted(values))}
    return maps


def _expected_features(model: Any | None, scaler: Any | None) -> list[str]:
    """Infer fitted feature names from scaler/model, falling back to dataset columns."""
    for artifact in (scaler, model):
        names = getattr(artifact, "feature_names_in_", None)
        if names is not None:
            return [str(name) for name in list(names)]
    return RAW_FEATURES


def prepare_features(
    user_values: dict[str, Any],
    df: pd.DataFrame,
    model: Any | None,
    scaler: Any | None,
) -> pd.DataFrame:
    """Align UI/API inputs to the model's exact fitted feature order."""
    values = {**UI_DEFAULTS, **user_values}
    feature_names = _expected_features(model, scaler)
    maps = _category_maps(df)

    row: dict[str, float] = {}
    for feature in feature_names:
        value = values.get(feature, UI_DEFAULTS.get(feature, 0.0))
        if feature in maps:
            encoded = maps[feature].get(str(value))
            if encoded is None:
                encoded = int(np.median(list(maps[feature].values()))) if maps[feature] else 0
            row[feature] = float(encoded)
        else:
            numeric_value = pd.to_numeric(value, errors="coerce")
            row[feature] = 0.0 if pd.isna(numeric_value) else float(numeric_value)

    return pd.DataFrame([row], columns=feature_names)


def predict_rainfall(
    user_values: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Run scaler + model prediction and return display-ready details."""
    model, scaler, error = load_artifacts()
    if error or model is None:
        raise RuntimeError(f"Could not load model artifacts: {error}")

    features = prepare_features(user_values, df, model, scaler)
    transformed = scaler.transform(features) if scaler is not None else features

    prediction = int(model.predict(transformed)[0])
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(transformed)[0], dtype=float)
        rain_probability = float(probabilities[1] if len(probabilities) > 1 else probabilities[0])
    else:
        rain_probability = 0.78 if prediction == 1 else 0.22

    rain_probability = float(np.clip(rain_probability, 0, 1))
    confidence = rain_probability if prediction == 1 else 1 - rain_probability

    return {
        "prediction": prediction,
        "rain_probability": rain_probability,
        "confidence": float(np.clip(confidence, 0, 1)),
        "features": features,
        "input": {**UI_DEFAULTS, **user_values},
    }


def estimate_flood_risk(probability: float, humidity: float, cloud_cover: float) -> tuple[str, int]:
    """Estimate flood risk from rain probability and atmospheric saturation."""
    score = int(round((probability * 65) + (humidity / 100 * 20) + (cloud_cover / 100 * 15)))
    if score >= 75:
        return "High", score
    if score >= 45:
        return "Moderate", score
    return "Low", score


def generate_weather_summary(values: dict[str, Any], probability: float, language: str = "English") -> str:
    """Create a natural language weather explanation."""
    humidity = float(values.get("Humidity (%)", 0))
    cloud = float(values.get("Cloud_Cover (%)", 0))
    pressure = float(values.get("Pressure (hPa)", 1010))
    wind = float(values.get("Wind_Speed (km/h)", 0))

    if probability >= 0.65:
        base = "High humidity and cloud cover indicate strong chances of rainfall."
    elif probability >= 0.40:
        base = "The atmosphere is mixed, so rainfall is possible but not certain."
    else:
        base = "Stable pressure and lower rain probability suggest mostly dry conditions."

    detail = (
        f" Humidity is {humidity:.0f}%, cloud cover is {cloud:.0f}%, "
        f"pressure is {pressure:.0f} hPa, and wind speed is {wind:.1f} km/h."
    )
    text = base + detail

    if language == "Hindi":
        return (
            "उच्च आर्द्रता और बादल वर्षा की संभावना को प्रभावित कर रहे हैं। "
            f"वर्षा संभावना {probability * 100:.0f}% है।"
        )
    if language == "Bengali":
        return (
            "আর্দ্রতা ও মেঘের পরিমাণ বৃষ্টিপাতের সম্ভাবনাকে প্রভাবিত করছে। "
            f"বৃষ্টির সম্ভাবনা {probability * 100:.0f}%।"
        )
    return text


def smart_recommendation(probability: float, aqi: float, risk_label: str) -> str:
    """Return practical user guidance."""
    if risk_label == "High":
        return "Avoid waterlogged routes, monitor local alerts, and keep emergency essentials ready."
    if probability >= 0.60:
        return "Carry an umbrella or raincoat and plan extra travel time today."
    if aqi >= 200:
        return "Rain is unlikely, but AQI is poor. Use a mask outdoors and limit exposure."
    return "Weather looks manageable. Keep checking live updates before long travel."


def city_baseline(df: pd.DataFrame, city: str) -> dict[str, float]:
    """Compare current weather against historical city averages."""
    if df.empty or "City" not in df.columns:
        return {}
    city_df = df[df["City"].astype(str).str.lower() == str(city).lower()]
    if city_df.empty:
        city_df = df
    cols = [
        "Temperature_Avg (°C)",
        "Humidity (%)",
        "Rainfall (mm)",
        "Wind_Speed (km/h)",
        "AQI",
        "Pressure (hPa)",
        "Cloud_Cover (%)",
    ]
    return {col: float(city_df[col].mean()) for col in cols if col in city_df.columns}


def api_key_from_sources() -> str:
    """Resolve API key from server-side environment sources only."""
    try:
        secret_value = st.secrets.get("OPENWEATHER_API_KEY", "")
        if secret_value:
            return str(secret_value)
    except Exception:
        pass
    return os.getenv("OPENWEATHER_API_KEY", "").strip()
