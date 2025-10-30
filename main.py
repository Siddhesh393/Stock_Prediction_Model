from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import io, base64, numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === Load pre-trained models ===
with open("models/arima.pkl", "rb") as f:
    arima_model = pickle.load(f)
with open("models/sarima.pkl", "rb") as f:
    sarima_model = pickle.load(f)
with open("models/tes_add.pkl", "rb") as f:
    tes_add_model = pickle.load(f)
with open("models/tes_mul.pkl", "rb") as f:
    tes_mul_model = pickle.load(f)
lstm_model = load_model("models/lstm_model.h5")
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# === Load base data (for LSTM last timesteps) ===
df = pd.read_csv("data/data.csv", parse_dates=True, index_col=0)
df1 = scaler.transform(np.array(df).reshape(-1, 1))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/forecast", response_class=HTMLResponse)
async def forecast(request: Request,
                   model_type: str = Form(...),
                   steps: int = Form(...)):

    # === Forecast logic ===
    if model_type == "ARIMA":
        forecast_result = arima_model.get_forecast(steps=steps)
        forecast_values = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()

    elif model_type == "SARIMA":
        forecast_result = sarima_model.get_forecast(steps=steps)
        forecast_values = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()

    elif model_type == "TES_Additive":
        forecast_values = tes_add_model.forecast(steps)
        conf_int = None

    elif model_type == "TES_Multiplicative":
        forecast_values = tes_mul_model.forecast(steps)
        conf_int = None

    elif model_type == "LSTM":
        look_back = 60

        # Ensure df1 is a NumPy array of shape (-1, 1)
        data_values = np.array(df1).reshape(-1, 1)

        # Normalize if not already scaled (assuming you used 'scaler' during training)
        last_sequence = data_values[-look_back:]  # (60, 1)
        temp_input = list(last_sequence.flatten())  # Flatten to 1D list of floats

        predictions = []

        for i in range(steps):
            x_input = np.array(temp_input[-look_back:]).reshape(1, look_back, 1)
            yhat = lstm_model.predict(x_input, verbose=0)

            # Append scalar value, not list
            temp_input.append(float(yhat[0][0]))
            predictions.append(float(yhat[0][0]))

        # Inverse scale predictions
        forecast_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        forecast_values = pd.Series(forecast_values.flatten())
        conf_int = None

    else:
        return {"error": "Invalid model type"}

    # === Generate dates for forecast ===
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date, periods=steps + 1, freq='D')[1:]

    # === Plot results ===
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df.values, label="Actual Data", color='blue')

    if model_type in ["ARIMA", "SARIMA"]:
        ax.plot(future_dates, forecast_values, label=f"{model_type} Forecast", color='orange')
        if conf_int is not None:
            ax.fill_between(future_dates,
                            conf_int.iloc[:, 0],
                            conf_int.iloc[:, 1],
                            color='pink', alpha=0.3, label='Confidence Interval')
    elif "TES" in model_type:
        ax.plot(future_dates, forecast_values, label=f"{model_type} Forecast", color='green')
    else:
        ax.plot(future_dates, forecast_values, label="LSTM Forecast", color='purple')

    ax.set_title(f"{model_type} Forecast for {steps} Days")
    ax.legend()

    # Convert plot to base64 for display in template
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return templates.TemplateResponse("forecast.html", {
        "request": request,
        "model_type": model_type,
        "steps": steps,
        "plot_base64": plot_base64
    })
        