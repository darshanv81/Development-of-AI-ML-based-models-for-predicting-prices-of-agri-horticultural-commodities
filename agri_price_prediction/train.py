import tensorflow as tf
from tensorflow import keras
import numpy as np

# Create a simple LSTM model (example)
# model = keras.Sequential([
#     keras.layers.LSTM(50, return_sequences=True, input_shape=(10, 1)),
#     keras.layers.LSTM(50, return_sequences=False),
#     keras.layers.Dense(1)
# ])

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# Generate dummy training data
X_train = np.random.rand(100, 10, 1)
y_train = np.random.rand(100, 1)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Save the model
model.save("price_prediction_model.h5")



