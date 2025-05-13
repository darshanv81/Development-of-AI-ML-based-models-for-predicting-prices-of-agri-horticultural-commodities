# # Import Required Libraries
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load Price Dataset (Vegetables & Pulses)
# price_data = pd.read_csv('datasets/agmarknet_price_data.csv')

# # Clean column names (strip spaces)
# price_data.columns = price_data.columns.str.strip()

# # Convert 'Date' column to datetime
# price_data['Date'] = pd.to_datetime(price_data['Date'], format='%d-%m-%Y', dayfirst=True)

# # Set 'Date' as index
# price_data.set_index('Date', inplace=True)

# # Ensure 'Modal Price' column exists
# if 'Modal Price' not in price_data.columns:
#     raise KeyError("❌ Column 'Modal Price' not found in dataset!")

# # Use 'Modal Price' instead of 'Price'
# price_values = price_data['Modal Price'].values.reshape(-1, 1)

# # Normalize Data for LSTM
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_price = scaler.fit_transform(price_values)

# # Function to Create Sequences for LSTM
# def create_sequences(data, seq_length):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:i+seq_length])
#         y.append(data[i+seq_length])
#     return np.array(X), np.array(y)

# sequence_length = 30  # Use past 30 days data to predict the next price
# X, y = create_sequences(scaled_price, sequence_length)

# # Split Data for Training
# split = int(len(X) * 0.8)
# X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# # Define LSTM Model
# model = Sequential([
#     LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
#     Dropout(0.2),
#     LSTM(50, return_sequences=False),
#     Dropout(0.2),
#     Dense(25),
#     Dense(1)
# ])

# # Compile and Train LSTM Model
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# # Function to Predict Future Prices
# def predict_future_prices(model, last_data, days_to_predict=90):
#     predictions = []
#     data = last_data.copy()
#     for _ in range(days_to_predict):
#         pred = model.predict(data.reshape(1, sequence_length, 1))
#         predictions.append(pred[0][0])
#         data = np.append(data[1:], pred).reshape(sequence_length, 1)
#     return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# # Predict Next 90 Days Prices
# future_prices = predict_future_prices(model, X_test[-1], days_to_predict=90)

# # Plot Future Price Predictions
# plt.figure(figsize=(12, 6))
# plt.plot(future_prices, label="Predicted Prices (Next 3+ Months)", color='blue')
# plt.xlabel("Days")
# plt.ylabel("Price")
# plt.title("Predicted Crop Prices for Vegetables & Pulses")
# plt.legend()
# plt.show()

# # Load Crop Recommendation Dataset
# crop_data = pd.read_csv('datasets/crop_recommendation.csv')

# # Print dataset columns for debugging
# print("Columns in crop dataset:", crop_data.columns)

# # Ensure correct column names
# expected_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
# missing_cols = [col for col in expected_columns if col not in crop_data.columns]
# if missing_cols:
#     raise KeyError(f"❌ Missing columns in dataset: {missing_cols}")

# # Extract Features and Labels
# X_crop = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
# y_crop = crop_data['label']  # Target variable

# # Train-Test Split
# X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# # Train Crop Recommendation Model
# crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
# crop_model.fit(X_train_crop, y_train_crop)

# # Evaluate Model Accuracy
# y_pred_crop = crop_model.predict(X_test_crop)
# accuracy = accuracy_score(y_test_crop, y_pred_crop)
# print(f'✅ Crop Recommendation Model Accuracy: {accuracy * 100:.2f}%')

# # Function to Recommend a Crop Based on Given Conditions
# def recommend_crop(n, p, k, temp, humidity, ph, rainfall):
#     input_features = np.array([[n, p, k, temp, humidity, ph, rainfall]])
#     return crop_model.predict(input_features)[0]

# # Example: Recommend Crop Based on Given Conditions
# recommended_crop = recommend_crop(50, 40, 30, 28, 65, 6.5, 200)
# print(f'🌱 Recommended Crop: {recommended_crop}')
