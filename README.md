# 🕒 Time Series Forecasting Web App

A **FastAPI-based web application** for time series forecasting using multiple models — **ARIMA, SARIMA, Holt-Winters (Additive & Multiplicative), and LSTM**.  
The app provides an interactive interface to forecast future values for any time series data (like stock prices, sales, weather trends, etc.).

---

## 🚀 Features

✅ Forecast using multiple models:
- **ARIMA**
- **SARIMA**
- **Holt-Winters Additive Trend**
- **Holt-Winters Multiplicative Trend**
- **LSTM (Deep Learning)**

✅ Visualizes:
- Actual data
- Forecasted data
- Confidence intervals (for ARIMA/SARIMA)

✅ Built with:
- 🧠 `statsmodels` for ARIMA/SARIMA/TES  
- 🤖 `TensorFlow/Keras` for LSTM  
- ⚡ `FastAPI` for backend  
- 🎨 `Jinja2 + HTML + CSS` for frontend visualization  

---

## 🗂️ Project Structure

```text
Time_Series_Forecasting/
│
├── main.py 
│
├── models/ 
│ ├── arima.pkl
│ ├── tes_add.pkl
│ ├── tes_mul.pkl
│ ├── lstm_model.h5
│ └── scaler.pkl
│
├── data/
│ └── data.csv 
│
├── templates/
│ ├── index.html 
│ └── forecast.html 
│
├── static/ 
│ └── style.css
|
└── README.md
```

## 🧩 Installation & Setup

###  Clone the Repository
```bash
git clone https://github.com/<your-username>/time-series-forecasting-app.git
cd time-series-forecasting-app
```