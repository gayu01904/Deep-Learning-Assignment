import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import urllib.request
import os
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import EarlyStopping # pyright: ignore[reportMissingImports]
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------- GPU SAFE SETUP ----------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ---------------- DATA DOWNLOAD ----------------
def download_dataset():
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    filename = 'airline-passengers.csv'
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    return filename

file = download_dataset()
data = pd.read_csv(file)

dataset = data.iloc[:, 1].values.astype("float32")

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(10,4))
plt.plot(dataset)
plt.title("Original Airline Passenger Data")
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.show()

# ---------------- SCALING ----------------
dataset = dataset.reshape(-1, 1)
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)

# ---------------- SEQUENCE GENERATION ----------------
FORECAST_HORIZON = 3   # 🔁 NEW: multi-step forecasting
TIME_STAMP = 12

def create_sequences(data, time_step, horizon):
    X, y = [], []
    for i in range(len(data) - time_step - horizon):
        X.append(data[i:i+time_step, 0])
        y.append(data[i+time_step:i+time_step+horizon, 0])
    return np.array(X), np.array(y)

train_size = int(len(dataset) * 0.8)
train, test = dataset[:train_size], dataset[train_size:]

trainX, trainY = create_sequences(train, TIME_STAMP, FORECAST_HORIZON)
testX, testY = create_sequences(test, TIME_STAMP, FORECAST_HORIZON)

trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
testX = testX.reshape(testX.shape[0], testX.shape[1], 1)

print("Train shape:", trainX.shape, trainY.shape)
print("Test shape:", testX.shape, testY.shape)

# ---------------- MODEL ----------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(TIME_STAMP, 1)),
    LayerNormalization(),             # 🔁 NEW
    Dropout(0.3),

    LSTM(32),
    Dropout(0.3),

    Dense(FORECAST_HORIZON)            # 🔁 Predict multiple future steps
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),  # 🔁 changed LR
    loss='mse',
    metrics=['mae']
)

model.summary()

# ---------------- TRAINING ----------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    trainX, trainY,
    epochs=60,                 # 🔁 reduced epochs
    batch_size=16,             # 🔁 increased batch size
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ---------------- EVALUATION ----------------
train_pred = model.predict(trainX)
test_pred = model.predict(testX)

# Inverse scale
train_pred = scaler.inverse_transform(
    train_pred.reshape(-1, 1)
).reshape(train_pred.shape)

test_pred = scaler.inverse_transform(
    test_pred.reshape(-1, 1)
).reshape(test_pred.shape)

trainY_inv = scaler.inverse_transform(
    trainY.reshape(-1, 1)
).reshape(trainY.shape)

testY_inv = scaler.inverse_transform(
    testY.reshape(-1, 1)
).reshape(testY.shape)

rmse = math.sqrt(mean_squared_error(testY_inv.flatten(), test_pred.flatten()))
mae = mean_absolute_error(testY_inv.flatten(), test_pred.flatten())

print("\nModel Evaluation:")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE : {mae:.2f}")

# ---------------- PLOT PREDICTIONS ----------------
plt.figure(figsize=(12,5))
plt.plot(testY_inv.flatten()[:100], label="Actual")
plt.plot(test_pred.flatten()[:100], label="Predicted")
plt.title("Multi-step LSTM Forecasting")
plt.legend()
plt.show()

# ---------------- SAVE MODEL ----------------
model.save("unique_multistep_lstm.h5")
print("Model saved as unique_multistep_lstm.h5")
