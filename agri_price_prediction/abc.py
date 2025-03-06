import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder

# 📌 Load dataset (replace with actual CSV path)
df = pd.read_csv("crop_data.csv")
data = pd.read_csv('crop_data.csv')
# Encode crop names into numeric labels
label_encoder = LabelEncoder()
data['crop_label'] = label_encoder.fit_transform(data['Crop Name'])

# Selecting Features (X) and Target (y)
X = data[['Temperature (°C)', 'Humidity (%)', 'pH Level', 'Rainfall (mm)']]
y = data['crop_label']

print("Encoded Crop Labels:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


# 📌 Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Save the trained model
joblib.dump(model, "crop_recommendation_model.pkl")
print("✅ Crop Recommendation Model Saved!")
