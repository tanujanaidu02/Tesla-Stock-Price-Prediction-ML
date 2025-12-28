# Tesla Stock Price Prediction using LSTM

This project predicts Tesla stock prices using historical data from Yahoo Finance.
An LSTM (Long Short-Term Memory) deep learning model is used for time-series forecasting.

## Dataset
- Source: Yahoo Finance (yfinance)
- Stock: Tesla (TSLA)
- Period: 2015 – 2023

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- yfinance
- Scikit-learn
- TensorFlow (Keras)

## Methodology
- Closing prices are selected and normalized using MinMaxScaler
- 80% data is used for training and 20% for testing
- A sliding window of 100 days is used
- Model performance is evaluated using RMSE
- Actual vs Predicted prices are visualized

## Note
This project is for educational purposes only and not intended for real-time trading.
