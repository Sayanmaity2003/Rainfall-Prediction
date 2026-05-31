from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from utils.adaptive_learning import retrain_probability_calibrator
from utils.i18n import t
from utils.model_utils import (
    city_baseline,
    estimate_flood_risk,
    generate_weather_summary,
    get_categories,
    load_artifacts,
    load_dataset,
    predict_rainfall,
    smart_recommendation,
    api_key_from_sources,
)
from utils.report import build_prediction_pdf
from utils.weather_api import (
    demo_weather_from_dataset,
    fetch_openweather,
    fetch_precipitation_probability,
    synthetic_history,
)

try:
    from streamlit_js_eval import get_geolocation

    GEOLOCATION_AVAILABLE = True
except Exception:
    GEOLOCATION_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
STYLE_PATH = ROOT / "styles" / "theme.css"
load_dotenv(ROOT / ".env")


st.set_page_config(
    page_title="Rainfall Detection",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    if st.session_state.get("theme_mode") == "Light":
        st.markdown(
            """
            <style>
            [data-testid="stAppViewContainer"] {
              background:
                radial-gradient(circle at 14% 10%, rgba(79, 140, 255, 0.18), transparent 30%),
                radial-gradient(circle at 86% 12%, rgba(255, 193, 7, 0.18), transparent 24%),
                linear-gradient(135deg, #eaf7ff 0%, #f8fbff 55%, #eef4ff 100%) !important;
              color: #0f172a !important;
            }
            :root {
              --panel: rgba(255, 255, 255, 0.72);
              --panel-strong: rgba(255, 255, 255, 0.9);
              --text: #0f172a;
              --muted: #475569;
              --border: rgba(15, 23, 42, 0.10);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def rain_layer(count: int = 34) -> str:
    drops = []
    for i in range(count):
        left = (i * 37) % 100
        duration = 1.35 + (i % 8) * 0.18
        delay = -1 * ((i % 11) * 0.21)
        opacity = 0.25 + (i % 5) * 0.12
        drops.append(
            f"<i style='left:{left}%; animation-duration:{duration:.2f}s; "
            f"animation-delay:{delay:.2f}s; opacity:{opacity:.2f}'></i>"
        )
    return "<div class='rain-field'>" + "".join(drops) + "</div>"


def section_title(title: str, caption: str | None = None) -> None:
    st.markdown(f"### {title}")
    if caption:
        st.markdown(f"<p class='small-muted'>{caption}</p>", unsafe_allow_html=True)


def metric_card(label: str, value: str, accent: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="small-muted">{accent}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_gauge(probability: float, title: str = "Rain Probability") -> go.Figure:
    color = "#4de3ff" if probability >= 0.5 else "#ffd166"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": title, "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.08)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(69,240,160,0.25)"},
                    {"range": [35, 65], "color": "rgba(255,209,102,0.25)"},
                    {"range": [65, 100], "color": "rgba(77,227,255,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=44, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def radar_chart(values: dict[str, float]) -> go.Figure:
    labels = ["Humidity", "Cloud", "Wind", "AQI", "Pressure"]
    normalized = [
        min(values.get("Humidity (%)", 0), 100),
        min(values.get("Cloud_Cover (%)", 0), 100),
        min(values.get("Wind_Speed (km/h)", 0) / 60 * 100, 100),
        min(values.get("AQI", 0) / 500 * 100, 100),
        min(max((values.get("Pressure (hPa)", 1010) - 980) / 70 * 100, 0), 100),
    ]
    fig = go.Figure(
        go.Scatterpolar(
            r=normalized + [normalized[0]],
            theta=labels + [labels[0]],
            fill="toself",
            line_color="#4de3ff",
            fillcolor="rgba(77,227,255,0.22)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=330,
        margin=dict(l=25, r=25, t=35, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def save_history(result: dict, source: str) -> None:
    row = {
        "time": datetime.now(),
        "source": source,
        "prediction": "Rain" if result["prediction"] == 1 else "No Rain",
        "rain_probability": result["rain_probability"],
        "model_rain_probability": result.get("model_rain_probability", result["rain_probability"]),
        "internet_rain_probability": result.get("internet_rain_probability"),
        "confidence": result["confidence"],
        "city": result["input"].get("City", "Unknown"),
        "aqi": result["input"].get("AQI", 0),
        "humidity": result["input"].get("Humidity (%)", 0),
        "temperature": result["input"].get("Temperature_Avg (°C)", 0),
    }
    st.session_state.prediction_history = st.session_state.get("prediction_history", [])
    st.session_state.prediction_history.append(row)


def render_prediction_result(result: dict, language: str) -> None:
    values = result["input"]
    probability = result["rain_probability"]
    confidence = result["confidence"]
    risk_label, risk_score = estimate_flood_risk(
        probability, float(values.get("Humidity (%)", 0)), float(values.get("Cloud_Cover (%)", 0))
    )
    summary = generate_weather_summary(values, probability, language)
    recommendation = smart_recommendation(probability, float(values.get("AQI", 0)), risk_label)

    rain = result["prediction"] == 1
    card_class = "result-rain" if rain else "result-sun"
    title = t(language, "rain") if rain else t(language, "no_rain")
    animation = rain_layer(42) if rain else "<div class='sun-core'></div>"

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            f"""
            <div class="result-card {card_class}">
              {animation}
              <div style="position:relative; z-index:2;">
                <div class="eyebrow">AI Model Output</div>
                <div class="status-title">{title}</div>
                <p class="small-muted">{summary}</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.plotly_chart(probability_gauge(probability), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(t(language, "confidence"), f"{confidence * 100:.1f}%")
    with c2:
        st.metric("Rain Probability", f"{probability * 100:.1f}%")
    with c3:
        st.metric(t(language, "risk"), f"{risk_label}", f"{risk_score}/100")
    with c4:
        st.metric("AQI", f"{float(values.get('AQI', 0)):.0f}", values.get("AQI_Category", "Moderate"))

    if result.get("internet_rain_probability") is None:
        calibration = result.get("calibration", {})
        section_title("Model vs Internet Probability")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.metric("Original Model", f"{result.get('model_rain_probability', probability) * 100:.1f}%")
        with a2:
            st.metric("Internet Forecast", f"{result['internet_rain_probability'] * 100:.1f}%")
        with a3:
            st.metric("Adaptive Final", f"{probability * 100:.1f}%")
        with a4:
            st.metric("Learning Error", f"{calibration.get('last_error', 0) * 100:+.1f}%")
        st.markdown(
            f"""
            <div class="glass-card">
              <b>Adaptive retraining:</b>
              The original saved model predicted first. The app then compared it with live internet precipitation probability
              and retrained a server-side calibration layer. Samples learned:
              <b>{calibration.get("samples", 0)}</b>, correction bias:
              <b>{calibration.get("bias", 0) * 100:+.1f}%</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="glass-card">
          <b>{t(language, "recommendation")}:</b> {recommendation}
        </div>
        """,
        unsafe_allow_html=True,
    )
    pdf = build_prediction_pdf(result, summary, recommendation)
    st.download_button(
        "Download Prediction Report as PDF",
        data=pdf,
        file_name="rainfall_prediction_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def home_page(language: str, df: pd.DataFrame) -> None:
    st.markdown(
        f"""
        <div class="hero">
          {rain_layer(38)}
          <div class="hero-content">
            <div class="eyebrow">Final Year Project</div>
            <h1>{t(language, "hero_title")}</h1>
            <p>{t(language, "hero_subtitle")}</p>
            
          </div>
          <div class="weather-orbit"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    records = len(df)
    cities = df["City"].nunique() if "City" in df else 0
    avg_rain = df["Rainfall (mm)"].mean() if "Rainfall (mm)" in df and not df.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Climate Records", f"{records:,}", "2024-2025 dataset")
    with c2:
        metric_card("Cities Covered", f"{cities}", "Indian climate zones")
    with c3:
        metric_card("Average Rainfall", f"{avg_rain:.2f} mm", "Historical baseline")
    with c4:
        metric_card("Model Status", "Active", "Scaler + ML model loaded")

    section_title("Technology Stack", "A modular AI weather platform built for deployment.")
    st.markdown(
        """
        <div class="tech-grid">
          <div class="tech-card"><b>Python</b><br><span class="small-muted">Core application runtime</span></div>
          <div class="tech-card"><b>Streamlit</b><br><span class="small-muted">Interactive web UI</span></div>
          <div class="tech-card"><b>Scikit-learn</b><br><span class="small-muted">Rainfall ML model</span></div>
          <div class="tech-card"><b>Plotly</b><br><span class="small-muted">Live analytics</span></div>
          <div class="tech-card"><b>Weather APIs</b><br><span class="small-muted">Location climate intelligence</span></div>
          <div class="tech-card"><b>Custom CSS</b><br><span class="small-muted">Glassmorphism animation layer</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Project Team Members")
    st.markdown(
        """
        <span class="team-pill">Sayan Maity</span>
        <span class="team-pill">Triyasa Das</span>
        <span class="team-pill">Sayan Mondol</span>
        <span class="team-pill">Saroj Panrui</span>
        """,
        unsafe_allow_html=True,
    )


def manual_prediction_page(language: str, df: pd.DataFrame) -> None:
    section_title("Manual Rainfall Prediction", "Enter weather parameters and let the trained model infer rainfall risk.")
    categories = get_categories(df)

    with st.form("manual_prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            temp = st.slider("Temperature_Avg (°C)", -5.0, 50.0, 28.0, 0.1)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 72.0, 0.5)
            wind = st.slider("Wind_Speed (km/h)", 0.0, 90.0, 8.0, 0.1)
            city = st.selectbox("City", categories["City"])
        with c2:
            aqi = st.number_input("AQI", min_value=0, max_value=500, value=150, step=1)
            pressure = st.number_input("Pressure (hPa)", min_value=850.0, max_value=1100.0, value=1010.0, step=0.1)
            cloud = st.slider("Cloud_Cover (%)", 0.0, 100.0, 55.0, 0.5)
            state = st.selectbox("State", categories["State"])
        with c3:
            aqi_category = st.selectbox("AQI_Category", categories["AQI_Category"])
            now = datetime.now()
            year = st.number_input("Year", min_value=2024, max_value=2035, value=now.year, step=1)
            month = st.selectbox("Month", list(range(1, 13)), index=now.month - 1)
            day = st.selectbox("Day", list(range(1, 32)), index=min(now.day, 31) - 1)

        submitted = st.form_submit_button(t(language, "predict"), use_container_width=True)

    if submitted:
        values = {
            "Temperature_Avg (°C)": temp,
            "Humidity (%)": humidity,
            "Wind_Speed (km/h)": wind,
            "AQI": aqi,
            "Pressure (hPa)": pressure,
            "Cloud_Cover (%)": cloud,
            "City": city,
            "State": state,
            "AQI_Category": aqi_category,
            "Year": year,
            "Month": month,
            "Day": day,
            "Rainfall (mm)": 0.0,
        }
        with st.spinner("Analyzing atmospheric signals..."):
            result = predict_rainfall(values, df)
        save_history(result, "Manual")
        st.session_state.latest_result = result
        render_prediction_result(result, language)
        render_historical_comparison(df, values)


def live_location_page(language: str, df: pd.DataFrame, api_key: str) -> None:
    section_title(
        "Live Location Prediction",
        "Use browser geolocation and weather APIs to generate a real-time rainfall forecast.",
    )

    geo_col, manual_col = st.columns([1.1, 1])
    with geo_col:
        st.markdown("<div class='glass-card'><b>Location Engine</b><br><span class='small-muted'>Allow location access in the browser for automatic coordinates.</span></div>", unsafe_allow_html=True)
        location = None
        if GEOLOCATION_AVAILABLE:
            location = get_geolocation()
        else:
            st.info("Install streamlit-js-eval to enable browser geolocation. Manual coordinates are available below.")

    with manual_col:
        lat_default, lon_default = 22.5726, 88.3639
        lat = st.number_input("Latitude", value=float(st.session_state.get("lat", lat_default)), format="%.6f")
        lon = st.number_input("Longitude", value=float(st.session_state.get("lon", lon_default)), format="%.6f")

    if location and isinstance(location, dict) and location.get("coords"):
        coords = location["coords"]
        lat = float(coords.get("latitude", lat))
        lon = float(coords.get("longitude", lon))
        st.session_state.lat = lat
        st.session_state.lon = lon

    run_live = st.button("Fetch Live Weather & Predict", use_container_width=True)
    if run_live:
        with st.spinner("Collecting Weather Data..."):
            internet_forecast = None
            try:
                weather = fetch_openweather(lat, lon, api_key)
                api_note = "Prediction: "
            except Exception as exc:
                weather = demo_weather_from_dataset(df, lat, lon)
                api_note = f"Demo fallback used: {exc}"

            result = predict_rainfall(weather, df)
            original_probability = result["rain_probability"]

            try:
                internet_forecast = fetch_precipitation_probability(lat, lon, api_key)
                calibration = retrain_probability_calibrator(
                    original_probability,
                    internet_forecast["internet_probability"],
                    weather,
                )
                calibrated_probability = calibration["calibrated_probability"]
                result["model_rain_probability"] = original_probability
                result["internet_rain_probability"] = internet_forecast["internet_probability"]
                result["internet_forecast"] = internet_forecast
                result["calibration"] = calibration
                result["rain_probability"] = calibrated_probability
                result["prediction"] = 1 if calibrated_probability >= 0.5 else 0
                result["confidence"] = (
                    calibrated_probability if result["prediction"] == 1 else 1 - calibrated_probability
                )
            except Exception as exc:
                result["model_rain_probability"] = original_probability
                result["internet_probability_error"] = str(exc)

        save_history(result, "Live")
        st.session_state.latest_result = result
        st.session_state.latest_weather = weather

        st.markdown(f"<p class='small-muted'>{api_note}</p>", unsafe_allow_html=True)
        if result.get("internet_forecast"):
            forecast = result["internet_forecast"]
            # st.caption(
            #     f"Internet probability source: {forecast['source']} | "
            #     f"{forecast['forecast_time']} | {forecast['description']}"
            # )
        elif result.get("internet_probability_error"):
            st.warning(
                "Internet rainfall probability could not be fetched, so the UI is showing the model-only prediction. "
                f"Reason: {result['internet_probability_error']}"
            )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Location", f"{weather.get('City')}", weather.get("State"))
        with c2:
            st.metric("Temperature", f"{weather.get('Temperature_Avg (°C)', 0):.1f} °C")
        with c3:
            st.metric("Humidity", f"{weather.get('Humidity (%)', 0):.0f}%")
        with c4:
            st.metric("Cloud Cover", f"{weather.get('Cloud_Cover (%)', 0):.0f}%")

        map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
        st.map(map_df, zoom=8)
        render_prediction_result(result, language)
        render_climate_insights(weather, result)


def render_climate_insights(weather: dict, result: dict) -> None:
    section_title("Climate Insights")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(radar_chart(weather), use_container_width=True)
    with c2:
        history = synthetic_history(weather)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["time"], y=history["rain_probability"] * 100, mode="lines+markers", name="Rain Probability"))
        fig.add_trace(go.Scatter(x=history["time"], y=history["Humidity"], mode="lines", name="Humidity"))
        fig.update_layout(
            title="Short-Term Climate Signal",
            height=330,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_historical_comparison(df: pd.DataFrame, values: dict) -> None:
    baseline = city_baseline(df, values.get("City", ""))
    if not baseline:
        return
    section_title("Historical Weather Comparison")
    compare = pd.DataFrame(
        {
            "Parameter": ["Temperature", "Humidity", "AQI", "Cloud Cover", "Pressure"],
            "Current": [
                values.get("Temperature_Avg (°C)", 0),
                values.get("Humidity (%)", 0),
                values.get("AQI", 0),
                values.get("Cloud_Cover (%)", 0),
                values.get("Pressure (hPa)", 0),
            ],
            "Historical Average": [
                baseline.get("Temperature_Avg (°C)", 0),
                baseline.get("Humidity (%)", 0),
                baseline.get("AQI", 0),
                baseline.get("Cloud_Cover (%)", 0),
                baseline.get("Pressure (hPa)", 0),
            ],
        }
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=compare["Parameter"], y=compare["Current"], name="Current"))
    fig.add_trace(go.Bar(x=compare["Parameter"], y=compare["Historical Average"], name="Historical Average"))
    fig.update_layout(barmode="group", height=360, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=35, b=20))
    st.plotly_chart(fig, use_container_width=True)


def analytics_page(language: str, df: pd.DataFrame) -> None:
    section_title("AI Analytics Dashboard", "Model outputs, prediction history, and climate intelligence charts.")
    latest = st.session_state.get("latest_result")
    if latest is None:
        sample = {
            "Temperature_Avg (°C)": 29.0,
            "Humidity (%)": 78.0,
            "Wind_Speed (km/h)": 9.0,
            "AQI": 135,
            "Pressure (hPa)": 1008.0,
            "Cloud_Cover (%)": 68.0,
            "City": "Kolkata",
            "State": "West Bengal",
            "AQI_Category": "Moderate",
            "Year": datetime.now().year,
            "Month": datetime.now().month,
            "Day": datetime.now().day,
            "Rainfall (mm)": 0.0,
        }
        try:
            latest = predict_rainfall(sample, df)
        except Exception:
            latest = {"rain_probability": 0.58, "confidence": 0.72, "prediction": 1, "input": sample}

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(probability_gauge(latest["rain_probability"], "Latest Rain Probability"), use_container_width=True)
    with c2:
        st.plotly_chart(radar_chart(latest["input"]), use_container_width=True)

    history_rows = st.session_state.get("prediction_history", [])
    if history_rows:
        history_df = pd.DataFrame(history_rows)
    else:
        weather = latest["input"]
        history_df = synthetic_history(weather)
        history_df["confidence"] = np.clip(history_df["rain_probability"] + 0.16, 0, 1)
        history_df["prediction"] = np.where(history_df["rain_probability"] > 0.5, "Rain", "No Rain")
        history_df["city"] = weather.get("City", "Demo")
        history_df["source"] = "Demo"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df["time"], y=history_df["rain_probability"] * 100, mode="lines+markers", name="Rain Probability"))
    if "confidence" in history_df:
        fig.add_trace(go.Scatter(x=history_df["time"], y=history_df["confidence"] * 100, mode="lines", name="Confidence"))
    fig.update_layout(title="Prediction History", height=360, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    d1, d2 = st.columns([1, 1])
    with d1:
        labels = history_df["prediction"].value_counts().index.tolist()
        values = history_df["prediction"].value_counts().values.tolist()
        pie = go.Figure(go.Pie(labels=labels, values=values, hole=0.58))
        pie.update_layout(title="Rain vs No-Rain Split", height=320, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(pie, use_container_width=True)
    with d2:
        temp_df = synthetic_history(latest["input"])
        line = go.Figure()
        line.add_trace(go.Scatter(x=temp_df["time"], y=temp_df["Temperature"], name="Temperature"))
        line.add_trace(go.Scatter(x=temp_df["time"], y=temp_df["AQI"], name="AQI", yaxis="y2"))
        line.update_layout(
            title="Temperature and AQI Trend",
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis2=dict(overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(line, use_container_width=True)


def sidebar() -> str:
    st.sidebar.markdown("## Rainfall Prediction")
    language = "English"
    st.session_state.theme_mode = st.sidebar.toggle("Light Mode", value=False)
    st.session_state.theme_mode = "Light" if st.session_state.theme_mode else "Dark"

    nav_labels = {
        t(language, "home"): "Home",
        t(language, "manual"): "Manual Prediction",
        t(language, "live"): "Live Location",
        # t(language, "analytics"): "AI Analytics",
    }
    selected_label = st.sidebar.radio("Navigation", list(nav_labels.keys()), label_visibility="collapsed")

    model, scaler, error = load_artifacts()
    if error:
        st.sidebar.error("Model artifacts could not be loaded.")
    else:
        st.sidebar.success("Model and scaler loaded")
    api_key_status = "configured" if api_key_from_sources() else "missing"
    st.sidebar.caption(f"Weather API key: {api_key_status} on server")
    return nav_labels[selected_label]


def main() -> None:
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Dark"
    df = load_dataset()
    page = sidebar()
    language = "English"
    api_key = api_key_from_sources()
    load_css()

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    if page == "Home":
        home_page(language, df)
    elif page == "Manual Prediction":
        manual_prediction_page(language, df)
    elif page == "Live Location":
        live_location_page(language, df, api_key)
    else:
        analytics_page(language, df)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
