import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv("data.csv")
data.columns = data.columns.str.strip()

values = data["value"].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
values = scaler.fit_transform(values)

# Create sequences
X, y = [], []
window = 3

for i in range(len(values) - window):
    X.append(values[i:i+window])
    y.append(values[i+window])

X, y = np.array(X), np.array(y)

# Model
model = Sequential([
    LSTM(50, input_shape=(window, 1)),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# Train
model.fit(X, y, epochs=50)

# Save model
model.save("model/model.keras")

print("✅ Model trained and saved")