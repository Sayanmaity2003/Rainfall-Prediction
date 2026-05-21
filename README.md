# Rainfall Detection using Machine Learning

A futuristic Streamlit web application for a Final Year Project: **Rainfall Detection using Machine Learning**.

The app loads the existing project artifacts:

- `rainfall_prediction_model.pkl`
- `scaler.pkl`
- `Indian_Climate_Dataset_2024_2025.csv`

## Features

- Glassmorphism dashboard UI with animated gradients and rain effects
- Manual rainfall prediction using the saved ML model
- Live-location prediction with browser geolocation
- OpenWeatherMap weather, AQI, and precipitation probability integration
- Adaptive probability calibration that learns from model-vs-internet forecast error
- Plotly analytics: gauges, radar charts, trends, pie charts, prediction history
- English-only interface for a cleaner presentation flow
- AI-style weather summary, flood risk indicator, and smart recommendations
- PDF prediction report download
- Dark and light visual modes

## Project Structure

```text
.
├── app.py
├── rainfall_prediction_model.pkl
├── scaler.pkl
├── Indian_Climate_Dataset_2024_2025.csv
├── requirements.txt
├── .env.example
├── assets/
├── model/
├── pages/
├── styles/
│   └── theme.css
└── utils/
    ├── i18n.py
    ├── model_utils.py
    ├── report.py
    └── weather_api.py
```

## Local Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure your API key.

Copy `.env.example` to `.env` and set:

```bash
OPENWEATHER_API_KEY=your_key
```

The key is intentionally not entered through the UI. It is read only on the server from `.env`, environment variables, or Streamlit secrets.

4. Run the application.

```bash
streamlit run app.py
```

## Deployment

### Streamlit Cloud

1. Push the project to GitHub.
2. Add `OPENWEATHER_API_KEY` in Streamlit secrets.
3. Set the entry point to `app.py`.

### Render

Use a Python web service with:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### Hugging Face Spaces

1. Create a Streamlit Space.
2. Upload the project files.
3. Add `OPENWEATHER_API_KEY` as a Space secret.
4. Keep `app.py` as the entry file.

## Notes

The saved scaler includes a `Rainfall (mm)` feature. The manual form follows the requested input fields and sets current rainfall to `0.0` when no live rainfall value is available. Live API mode uses the current 1-hour rainfall value if OpenWeatherMap returns it.

In live-location mode, the app first predicts rainfall using the saved ML model. It then fetches OpenWeatherMap forecast precipitation probability from the internet, compares both probabilities, and retrains a local server-side calibration layer stored under `model/adaptive_calibration.json`. Feedback rows are saved to `model/prediction_feedback.csv` for future offline retraining.
