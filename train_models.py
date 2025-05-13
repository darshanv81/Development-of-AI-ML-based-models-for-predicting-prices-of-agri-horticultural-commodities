# train_models.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 📌 Load Price Dataset
price_data = pd.read_csv('datasets/agmarknet_price_data.csv')
price_data.columns = price_data.columns.str.strip()
price_data['Date'] = pd.to_datetime(price_data['Date'], format='%d-%m-%Y', dayfirst=True)
price_data.set_index('Date', inplace=True)

# Check if 'Modal Price' column exists
if 'Modal Price' not in price_data.columns:
    raise KeyError("❌ Column 'Modal Price' not found in dataset!")

# Prepare Data for LSTM
price_values = price_data['Modal Price'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_price = scaler.fit_transform(price_values)

# Save scaler
joblib.dump(scaler, "models/price_scaler.pkl")

# Function to Create Sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 30
X, y = create_sequences(scaled_price, sequence_length)
split = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Define LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile & Train Model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save Trained Model
model.save("models/price_prediction_model.h5")

# 📌 Load Crop Recommendation Dataset
crop_data = pd.read_csv('datasets/crop_recommendation.csv')
expected_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
if not all(col in crop_data.columns for col in expected_columns):
    raise KeyError("❌ Some expected columns are missing in the crop dataset!")

X_crop = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_crop = crop_data['label']

# Train-Test Split
X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# Train Crop Recommendation Model
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_train_crop, y_train_crop)

# Save Crop Model
joblib.dump(crop_model, "models/crop_recommendation_model.pkl")

print("✅ Models Trained & Saved Successfully!")
