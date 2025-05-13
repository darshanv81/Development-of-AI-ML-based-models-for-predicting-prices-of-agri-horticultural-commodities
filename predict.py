# predict.py
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 🔹 Load Trained Models
price_model = tf.keras.models.load_model("models/price_prediction_model.h5")
scaler = joblib.load("models/price_scaler.pkl")
crop_model = joblib.load("models/crop_recommendation_model.pkl")

# Function to Predict Future Prices
def predict_future_prices(last_data, days_to_predict=90):
    predictions = []
    data = last_data.copy()
    for _ in range(days_to_predict):
        pred = price_model.predict(data.reshape(1, 30, 1))
        predictions.append(pred[0][0])
        data = np.append(data[1:], pred).reshape(30, 1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 🔹 Load Past Prices for Visualization
price_data = pd.read_csv('datasets/agmarknet_price_data.csv')
price_data.columns = price_data.columns.str.strip()
price_data['Date'] = pd.to_datetime(price_data['Date'], format='%d-%m-%Y', dayfirst=True)
price_data.set_index('Date', inplace=True)

# Normalize and Get Recent Data
price_values = price_data['Modal Price'].values.reshape(-1, 1)
scaled_price = scaler.transform(price_values)

# Predict Future Prices
future_prices = predict_future_prices(scaled_price[-30:], days_to_predict=90)

# 🔹 Plot Past & Future Prices
plt.figure(figsize=(14, 6))
past_days = price_data.index[-60:].tolist()
future_days = [price_data.index[-1] + timedelta(days=i) for i in range(1, 91)]

plt.plot(past_days, price_data['Modal Price'].values[-60:], label="Past Prices", color='blue')
plt.plot(future_days, future_prices, label="Predicted Future Prices", color='red')
plt.axvline(datetime.today(), color='green', linestyle='dashed', label="Today")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Past and Future Crop Prices")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# 🔹 Fetch Automatic Data for Crop Recommendation
def get_weather_data():
    API_URL = "https://api.open-meteo.com/v1/forecast?latitude=19.07&longitude=72.87&daily=temperature_2m_max,humidity_2m_max,precipitation_sum&timezone=Asia/Kolkata"
    response = requests.get(API_URL).json()
    
    temperature = response['daily']['temperature_2m_max'][-1]
    humidity = response['daily']['humidity_2m_max'][-1]
    rainfall = response['daily']['precipitation_sum'][-1]
    return temperature, humidity, rainfall

# 🔹 Automatic Crop Recommendation
def recommend_crop_auto():
    temp, humidity, rainfall = get_weather_data()
    input_features = np.array([[50, 40, 30, temp, humidity, 6.5, rainfall]])  # Default N, P, K, pH
    return crop_model.predict(input_features)[0]

# 🔹 Manual Crop Recommendation (User Input)
def recommend_crop_manual(n, p, k, temp, humidity, ph, rainfall):
    input_features = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    return crop_model.predict(input_features)[0]

# Example Predictions
print(f'🌱 Automatically Recommended Crop: {recommend_crop_auto()}')
print(f'🌾 Manually Recommended Crop: {recommend_crop_manual(50, 40, 30, 28, 65, 6.5, 200)}')
