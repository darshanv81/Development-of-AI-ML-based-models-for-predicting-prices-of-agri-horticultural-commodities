from flask import Flask, jsonify, request
import random
import datetime
import requests

app = Flask(__name__)

# Sample Real-Time Prices (Replace with actual API/Database)
real_time_prices = {
    "Potato": 25,
    "Onion": 30,
    "Tomato": 40,
    "Wheat": 22,
    "Rice": 45
}

# Crop Images Mapping
crop_images = {
    "Potato": "https://example.com/images/potato.jpg",
    "Onion": "https://example.com/images/onion.jpg",
    "Tomato": "https://example.com/images/tomato.jpg",
    "Wheat": "https://example.com/images/wheat.jpg",
    "Rice": "https://example.com/images/rice.jpg"
}

# Function to generate past 3 months' price trends
def generate_past_price_trends(crop):
    today = datetime.date.today()
    past_prices = []
    
    for i in range(90):
        date = today - datetime.timedelta(days=90 - i)
        price = real_time_prices[crop] + random.randint(-5, 5)
        past_prices.append({"date": date.strftime("%Y-%m-%d"), "price": price})
    
    return past_prices

# Function to predict next 3 months' prices
def predict_future_prices(crop):
    today = datetime.date.today()
    future_prices = []
    
    base_price = real_time_prices.get(crop, 30)
    
    for i in range(90):
        date = today + datetime.timedelta(days=i + 1)
        price = base_price + random.randint(-3, 5)
        future_prices.append({"date": date.strftime("%Y-%m-%d"), "price": price})
    
    return future_prices

# Weather API (Replace with your OpenWeatherMap API Key)
WEATHER_API_KEY = "a1a5792b8c767958d3b2166d5a135e9b"
LATITUDE = "20.5937"
LONGITUDE = "78.9629"

def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={LATITUDE}&lon={LONGITUDE}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "weather": data["weather"][0]["description"]
            }
    except Exception as e:
        return {"error": str(e)}

@app.route("/")
def home():
    return "Agri-Horticultural Commodity Price Prediction API"

# Real-time Prices
@app.route("/real_time_prices", methods=["GET"])
def get_real_time_prices():
    return jsonify(real_time_prices)

# Past 3-Month Trends
@app.route("/past_price_trends", methods=["POST"])
def get_past_price_trends():
    try:
        data = request.json
        crop = data.get("crop")
        if crop not in real_time_prices:
            return jsonify({"error": "Crop not found"}), 404
        return jsonify({"crop": crop, "past_prices": generate_past_price_trends(crop)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Predict Future Prices
@app.route("/predict_price", methods=["POST"])
def predict_price():
    try:
        data = request.json
        crop = data.get("crop")
        if crop not in real_time_prices:
            return jsonify({"error": "Crop not found"}), 404
        return jsonify({"crop": crop, "future_prices": predict_future_prices(crop)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Crop Recommendation with Image
@app.route("/recommend_crop", methods=["POST"])
def recommend_crop():
    recommended_crop = random.choice(list(real_time_prices.keys()))
    crop_image_url = crop_images.get(recommended_crop, "")
    
    return jsonify({
        "recommended_crop": recommended_crop,
        "image_url": crop_image_url
    })

# Weather Information
@app.route("/weather", methods=["GET"])
def weather():
    return jsonify(get_weather())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
