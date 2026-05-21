"""Adaptive probability calibration for live rainfall predictions.

The original classifier remains untouched. This module trains a tiny
server-side correction layer from model probability vs weather API probability
so repeated live predictions can become better aligned with current conditions.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
STATE_PATH = MODEL_DIR / "adaptive_calibration.json"
FEEDBACK_PATH = MODEL_DIR / "prediction_feedback.csv"


DEFAULT_STATE = {
    "samples": 0,
    "bias": 0.0,
    "mae": 0.0,
    "last_error": 0.0,
    "updated_at": None,
}


def load_calibration_state() -> dict[str, Any]:
    """Load persisted calibration state."""
    if not STATE_PATH.exists():
        return DEFAULT_STATE.copy()
    try:
        return {**DEFAULT_STATE, **json.loads(STATE_PATH.read_text(encoding="utf-8"))}
    except Exception:
        return DEFAULT_STATE.copy()


def retrain_probability_calibrator(
    model_probability: float,
    internet_probability: float,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Update the adaptive correction layer from one live probability comparison."""
    MODEL_DIR.mkdir(exist_ok=True)
    state = load_calibration_state()

    model_probability = float(np.clip(model_probability, 0, 1))
    internet_probability = float(np.clip(internet_probability, 0, 1))
    error = internet_probability - model_probability

    old_samples = int(state.get("samples", 0))
    new_samples = old_samples + 1
    learning_rate = 1 / min(new_samples, 25)
    previous_bias = float(state.get("bias", 0.0))
    previous_mae = float(state.get("mae", 0.0))

    bias = previous_bias + learning_rate * (error - previous_bias)
    bias = float(np.clip(bias, -0.35, 0.35))
    mae = previous_mae + learning_rate * (abs(error) - previous_mae)
    calibrated_probability = float(np.clip(model_probability + bias, 0, 1))

    new_state = {
        "samples": new_samples,
        "bias": bias,
        "mae": float(mae),
        "last_error": float(error),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    STATE_PATH.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    _append_feedback(model_probability, internet_probability, calibrated_probability, error, context)

    return {
        **new_state,
        "model_probability": model_probability,
        "internet_probability": internet_probability,
        "calibrated_probability": calibrated_probability,
    }


def apply_existing_calibration(model_probability: float) -> dict[str, Any]:
    """Apply the current correction without retraining."""
    state = load_calibration_state()
    model_probability = float(np.clip(model_probability, 0, 1))
    calibrated = float(np.clip(model_probability + float(state.get("bias", 0.0)), 0, 1))
    return {**state, "model_probability": model_probability, "calibrated_probability": calibrated}


def _append_feedback(
    model_probability: float,
    internet_probability: float,
    calibrated_probability: float,
    error: float,
    context: dict[str, Any],
) -> None:
    """Persist a row for future offline model retraining."""
    is_new = not FEEDBACK_PATH.exists()
    fields = [
        "timestamp",
        "city",
        "state",
        "temperature",
        "humidity",
        "cloud_cover",
        "pressure",
        "wind_speed",
        "aqi",
        "model_probability",
        "internet_probability",
        "calibrated_probability",
        "error",
    ]
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "city": context.get("City", "Unknown"),
        "state": context.get("State", "Unknown"),
        "temperature": context.get("Temperature_Avg (°C)", context.get("Temperature_Avg (Â°C)", 0)),
        "humidity": context.get("Humidity (%)", 0),
        "cloud_cover": context.get("Cloud_Cover (%)", 0),
        "pressure": context.get("Pressure (hPa)", 0),
        "wind_speed": context.get("Wind_Speed (km/h)", 0),
        "aqi": context.get("AQI", 0),
        "model_probability": model_probability,
        "internet_probability": internet_probability,
        "calibrated_probability": calibrated_probability,
        "error": error,
    }
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

