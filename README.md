# Stock Price Prediction using LSTM

This project aims to predict stock prices using a Long Short-Term Memory (LSTM) neural network model. The model is trained on historical stock price data to forecast future prices. It leverages time series data and deep learning techniques to make accurate predictions, offering valuable insights for stock traders, investors, and analysts.

## Project Overview

With the financial markets generating vast amounts of data daily, accurate stock price predictions can be challenging but highly rewarding. LSTM networks are well-suited for time series data because of their ability to learn complex patterns and dependencies over long sequences. In this project, an LSTM-based model is built to predict the closing prices of a given stock, utilizing historical price data as input.

## Features

- **Data Preprocessing**: Cleans and prepares the dataset by normalizing prices and splitting it into training and testing sets.
- **LSTM Model**: A neural network model trained on historical stock prices to predict future prices.
- **Evaluation Metrics**: Measures the performance of the model using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Visualization**: Plots the predicted vs. actual stock prices to illustrate model accuracy.

## Project Structure

- `data/`: Contains the historical stock price dataset.
- `src/`: Contains the code for data preprocessing, model training, evaluation, and prediction.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model experimentation.
- `README.md`: Project description and setup instructions.
- `requirements.txt`: Python dependencies for running the project.

## Requirements

- Python 3.7+
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Data

The model is trained on historical stock price data, which includes features such as:

- **Date**: Date of the stock data point
- **Open**: Stock price at market open
- **High**: Highest price during the trading day
- **Low**: Lowest price during the trading day
- **Close**: Stock price at market close
- **Volume**: Number of shares traded

The only feature used in the project is the closing price ( **Close** ).

## How to Run

1. **Data Preparation**: Load and preprocess the dataset.
2. **Model Training**: Train the LSTM model on the preprocessed data.
3. **Prediction**: Generate predictions for test data.
4. **Evaluation and Visualization**: Evaluate the model performance and visualize the predicted vs. actual prices.

To execute the model training and prediction:

```bash
python src/train.py
```

## Model Performance

The model performance is evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the average squared differences between predicted and actual values.

## Results

The LSTM model was able to capture the general trend in stock prices, achieving a reasonably low MAE and RMSE on the test dataset. However, stock price predictions are inherently challenging due to market volatility, and more complex models or additional features may improve performance.

## Future Improvements

- **Additional Features**: Integrate additional features like trading volume, technical indicators, or sentiment analysis.
- **Model Architecture**: Experiment with different neural network architectures and hyperparameter tuning.
- **Big Data Integration**: Utilize big data frameworks such as Spark and Hadoop for larger datasets.
