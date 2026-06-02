# Rainfall Prediction using Machine Learning

A futuristic Streamlit web application developed as a **Final Year Project** for predicting rainfall using Machine Learning and real-time weather analysis.

## Project Team

### Presented By

* **Sayan Maity** (10300122158)
* **Sayan Mondal** (10300122159)
* **Triyasa Das** (10300122202)
* **Saroj Panrui** (10300122152)

### Guided By

**Dr. Arindam Giri**
Associate Professor
Haldia Institute of Technology

---

# Project Overview

Rainfall prediction plays an important role in agriculture, disaster management, water resource planning, and weather forecasting. Traditional statistical methods often fail to capture complex atmospheric patterns.

This project uses Machine Learning algorithms to analyze historical meteorological data and predict rainfall occurrence with improved accuracy and reliability.

The system combines:

* Historical weather analysis
* Data preprocessing and feature engineering
* Machine Learning classification models
* Real-time weather integration
* Interactive data visualization
* Smart rainfall prediction dashboard

---

# Technologies Used

## Programming Language

* Python

## Libraries & Frameworks

* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Pickle

## APIs

* OpenWeatherMap API

## Development Platform

* Google Colab
* VS Code

---

# Machine Learning Models Used

The project implements and compares multiple Machine Learning algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. K-Nearest Neighbors (KNN)
6. Support Vector Machine (SVM)

## Best Performing Model

The **Random Forest Classifier** achieved the highest accuracy and most balanced prediction performance.

---

# Dataset Information

Dataset Used:
`Indian_Climate_Dataset_2024_2025.csv`

## Features Used

* City
* State
* Temperature
* Humidity
* Rainfall
* Wind Speed
* AQI
* AQI Category
* Pressure
* Cloud Cover
* Year
* Month
* Day

## Target Variable

`RainTomorrow`

* 1 = Rainfall Expected
* 0 = No Rainfall

---

# Project Features

* Rainfall prediction using Machine Learning
* Interactive Streamlit dashboard
* Real-time weather monitoring
* Live location-based prediction
* Weather analytics and visualization
* Plotly charts and gauges
* Rain probability analysis
* Weather recommendations
* Flood risk indication
* PDF report generation
* Dark and light UI modes
* Adaptive prediction calibration
* Glassmorphism dashboard design

---

# Project Structure

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

---

# Working Procedure

## 1. Data Collection

Historical weather data was collected from climate datasets.

## 2. Data Preprocessing

* Missing value handling
* Label encoding
* Feature scaling
* Correlation analysis
* Class balancing

## 3. Feature Engineering

Additional features like:

* Year
* Month
* Day
  were extracted from the Date column.

## 4. Model Training

Multiple ML models were trained and evaluated.

## 5. Model Evaluation

Performance metrics used:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## 6. Deployment

The trained model was deployed using Streamlit for interactive rainfall prediction.

---

# Installation & Setup

## 1. Create Virtual Environment

```bash
python -m venv .venv
```

## 2. Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure API Key

Create a `.env` file and add:

```bash
OPENWEATHER_API_KEY=your_api_key
```

---

## 5. Run the Application

```bash
streamlit run app.py
```

---

# Deployment

## Streamlit Cloud

* Push project to GitHub
* Add API key in Streamlit secrets
* Deploy using `app.py`

## Render

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Hugging Face Spaces

* Create Streamlit Space
* Upload files
* Add API secret
* Run using `app.py`

---

# Future Scope

* Real-time rainfall forecasting
* Deep Learning integration
* Mobile application development
* Cloud deployment optimization
* Integration with IoT weather sensors
* Advanced forecasting using larger datasets

---

# Conclusion

This project demonstrates how Machine Learning can be used for rainfall prediction using meteorological data.

Among all models, the Random Forest Classifier achieved the best performance and provided reliable rainfall prediction results. The integration of real-time weather APIs and interactive visualization enhanced the usability and effectiveness of the system.

---

# References

1. Dutta & Gouthaman, *Rainfall Prediction using ML and Neural Networks*, IJRTE, 2020

2. Revathi & Usharani, *ML Classification Algorithms for Rainfall Prediction*, 2021

3. Rahman et al., *ML Fusion-based Rainfall Prediction for Smart Cities*, Sensors, 2022

4. Kaggle Rainfall Dataset
   https://www.kaggle.com/datasets/sujithmandala/simple-rainfall-classification-dataset

---

# Acknowledgement

We sincerely thank **Dr. Arindam Giri** and the Department of Computer Science & Engineering, Haldia Institute of Technology, for their valuable guidance and support throughout the project development.
