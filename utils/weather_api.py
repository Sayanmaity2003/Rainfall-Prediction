"""Weather and geocoding API helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st


OPENWEATHER_BASE = "https://api.openweathermap.org"


@st.cache_data(ttl=900, show_spinner=False)
def reverse_geocode(lat: float, lon: float, api_key: str) -> dict[str, Any]:
    """Resolve latitude/longitude to city and state through OpenWeather."""
    if not api_key:
        return {"city": "Unknown", "state": "Unknown"}
    try:
        response = requests.get(
            f"{OPENWEATHER_BASE}/geo/1.0/reverse",
            params={"lat": lat, "lon": lon, "limit": 1, "appid": api_key},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return {"city": "Unknown", "state": "Unknown"}
        place = payload[0]
        return {
            "city": place.get("name", "Unknown"),
            "state": place.get("state", place.get("country", "Unknown")),
        }
    except Exception as exc:
        return {"city": "Unknown", "state": "Unknown", "error": str(exc)}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_openweather(lat: float, lon: float, api_key: str) -> dict[str, Any]:
    """Fetch live weather and AQI from OpenWeatherMap."""
    if not api_key:
        raise RuntimeError("OpenWeatherMap API key is missing.")

    weather_response = requests.get(
        f"{OPENWEATHER_BASE}/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
        timeout=15,
    )
    weather_response.raise_for_status()
    weather = weather_response.json()

    aqi_value = 100
    aqi_category = "Moderate"
    try:
        air_response = requests.get(
            f"{OPENWEATHER_BASE}/data/2.5/air_pollution",
            params={"lat": lat, "lon": lon, "appid": api_key},
            timeout=15,
        )
        air_response.raise_for_status()
        air = air_response.json()
        owm_aqi = int(air.get("list", [{}])[0].get("main", {}).get("aqi", 2))
        aqi_value, aqi_category = _openweather_aqi_to_indian_scale(owm_aqi)
    except Exception:
        pass

    place = reverse_geocode(lat, lon, api_key)
    now = datetime.now()
    return {
        "Latitude": lat,
        "Longitude": lon,
        "City": place.get("city") or weather.get("name") or "Unknown",
        "State": place.get("state") or weather.get("sys", {}).get("country") or "Unknown",
        "Temperature_Avg (°C)": float(weather.get("main", {}).get("temp", 0.0)),
        "Humidity (%)": float(weather.get("main", {}).get("humidity", 0.0)),
        "Pressure (hPa)": float(weather.get("main", {}).get("pressure", 1010.0)),
        "Cloud_Cover (%)": float(weather.get("clouds", {}).get("all", 0.0)),
        "Wind_Speed (km/h)": float(weather.get("wind", {}).get("speed", 0.0)) * 3.6,
        "AQI": aqi_value,
        "AQI_Category": aqi_category,
        "Rainfall (mm)": float(weather.get("rain", {}).get("1h", 0.0)),
        "Year": now.year,
        "Month": now.month,
        "Day": now.day,
        "Description": weather.get("weather", [{}])[0].get("description", "live weather"),
    }


@st.cache_data(ttl=900, show_spinner=False)
def fetch_precipitation_probability(lat: float, lon: float, api_key: str) -> dict[str, Any]:
    """Fetch internet precipitation probability from OpenWeather forecast data."""
    if not api_key:
        raise RuntimeError("OpenWeatherMap API key is missing.")

    response = requests.get(
        f"{OPENWEATHER_BASE}/data/2.5/forecast",
        params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric", "cnt": 1},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    forecast = payload.get("list", [{}])[0]
    probability = float(forecast.get("pop", 0.0))
    return {
        "internet_probability": float(np.clip(probability, 0, 1)),
        "forecast_time": forecast.get("dt_txt", "next forecast window"),
        "source": "OpenWeatherMap 5 day / 3 hour forecast",
        "description": forecast.get("weather", [{}])[0].get("description", "forecast"),
    }


def _openweather_aqi_to_indian_scale(owm_aqi: int) -> tuple[int, str]:
    """Map OpenWeather AQI 1-5 to a rough Indian AQI category/value."""
    mapping = {
        1: (45, "Good"),
        2: (95, "Satisfactory"),
        3: (155, "Moderate"),
        4: (245, "Poor"),
        5: (330, "Very Poor"),
    }
    return mapping.get(owm_aqi, (100, "Moderate"))


def demo_weather_from_dataset(df: pd.DataFrame, lat: float, lon: float) -> dict[str, Any]:
    """Generate a realistic fallback weather snapshot from local historical data."""
    now = datetime.now()
    if df.empty:
        return {
            "Latitude": lat,
            "Longitude": lon,
            "City": "Mumbai",
            "State": "Maharashtra",
            "Temperature_Avg (°C)": 28.0,
            "Humidity (%)": 75.0,
            "Pressure (hPa)": 1010.0,
            "Cloud_Cover (%)": 60.0,
            "Wind_Speed (km/h)": 8.0,
            "AQI": 120,
            "AQI_Category": "Moderate",
            "Rainfall (mm)": 0.0,
            "Year": now.year,
            "Month": now.month,
            "Day": now.day,
            "Description": "demo weather",
        }
    sample = df.sample(1, random_state=int(abs(lat * lon)) % 9999).iloc[0]
    return {
        "Latitude": lat,
        "Longitude": lon,
        "City": str(sample.get("City", "Mumbai")),
        "State": str(sample.get("State", "Maharashtra")),
        "Temperature_Avg (°C)": float(sample.get("Temperature_Avg (°C)", 28.0)),
        "Humidity (%)": float(sample.get("Humidity (%)", 75.0)),
        "Pressure (hPa)": float(sample.get("Pressure (hPa)", 1010.0)),
        "Cloud_Cover (%)": float(sample.get("Cloud_Cover (%)", 60.0)),
        "Wind_Speed (km/h)": float(sample.get("Wind_Speed (km/h)", 8.0)),
        "AQI": int(sample.get("AQI", 120)),
        "AQI_Category": str(sample.get("AQI_Category", "Moderate")),
        "Rainfall (mm)": float(sample.get("Rainfall (mm)", 0.0)),
        "Year": now.year,
        "Month": now.month,
        "Day": now.day,
        "Description": "local dataset simulation",
    }


def synthetic_history(current: dict[str, Any], points: int = 12) -> pd.DataFrame:
    """Create smooth short-term history for dashboard charts."""
    rng = np.random.default_rng(42)
    hours = pd.date_range(end=pd.Timestamp.now(), periods=points, freq="2h")
    return pd.DataFrame(
        {
            "time": hours,
            "rain_probability": np.clip(
                np.linspace(0.25, 0.75, points) + rng.normal(0, 0.06, points), 0, 1
            ),
            "AQI": np.clip(float(current.get("AQI", 120)) + rng.normal(0, 18, points), 0, 500),
            "Temperature": float(current.get("Temperature_Avg (°C)", 28)) + rng.normal(0, 2, points),
            "Humidity": np.clip(float(current.get("Humidity (%)", 70)) + rng.normal(0, 8, points), 0, 100),
            "Pressure": float(current.get("Pressure (hPa)", 1010)) + rng.normal(0, 4, points),
            "Wind": np.clip(float(current.get("Wind_Speed (km/h)", 8)) + rng.normal(0, 2, points), 0, 80),
        }
    )
