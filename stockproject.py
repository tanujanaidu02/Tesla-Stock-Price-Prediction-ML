
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import math

# Step 1: Load Tesla Data
print("Downloading Tesla data...")
data = yf.download('TSLA', start='2015-01-01', end='2023-12-31')
print("Data successfully loaded!\n")

# Step 2: Display basic info
print(data.head())

# Step 3: Preprocessing
close_data = data[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

training_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:training_size]
test_data = scaled_data[training_size:]

# Function to create dataset with timesteps
def create_dataset(dataset, time_step=100):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 4: Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train Model
print("Training the model, please wait...")
model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)

# Step 6: Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Step 7: Inverse scaling
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_true = scaler.inverse_transform([y_train])
y_test_true = scaler.inverse_transform([y_test])

# Step 8: Evaluation
train_rmse = math.sqrt(mean_squared_error(y_train_true[0], train_predict[:, 0]))
test_rmse = math.sqrt(mean_squared_error(y_test_true[0], test_predict[:, 0]))

print("\nModel Performance:")
print(f"Training RMSE: {train_rmse}")
print(f"Testing RMSE: {test_rmse}")

# Step 9: Visualization
look_back = 100
trainPredictPlot = np.empty_like(scaled_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

testPredictPlot = np.empty_like(scaled_data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(scaled_data) - 1, :] = test_predict

plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(scaled_data), color='blue', label='Actual Tesla Price')
plt.plot(trainPredictPlot, color='green', label='Training Prediction')
plt.plot(testPredictPlot, color='red', label='Testing Prediction')
plt.title('Tesla Stock Price Prediction (LSTM Model)')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
