from flask import Flask, render_template, request, jsonify
import random
import datetime

app = Flask(__name__)

# Categorized crops for better organization
vegetables = [
    "Onion", "Potato", "Tomato", "Carrot", "Cucumber", "Cauliflower", "Peas", 
    "Cabbage", "Spinach", "Lettuce", "Beetroot", "Radish", "Pumpkin", "Brinjal", 
    "Bell Pepper", "Coriander", "Chili", "Garlic", "Ginger", "Okra", "Zucchini", 
    "Mushroom", "Lemon", "Watermelon", "Melon", "Cantaloupe", "Chili Pepper", 
    "Squash", "Broccoli", "Fennel", "Celery", "Asparagus", "Basil", "Oregano", 
    "Rosemary", "Thyme", "Parsley", "Mint", "Sweet Corn", "Eggplant", "Sweet Potato", 
    "Turnip", "Kale", "Artichoke"
]

pulses = [
    "Beans", "Lima Beans", "Kidney Beans", "Chickpeas", "Black Beans", "Pinto Beans", 
    "Soybeans", "Mung Beans", "Broad Beans", "Lentils"
]

# Sample price prediction function (replace with actual ML model)
def predict_prices(crop):
    # In a real scenario, you'd load your model and predict prices based on the crop
    dates = [datetime.date.today() + datetime.timedelta(days=i) for i in range(0, 90, 7)]
    prices = [round(random.uniform(10, 50), 2) for _ in range(len(dates))]
    return {"dates": [date.strftime("%d %b %Y") for date in dates], "prices": prices}

# Sample crop recommendation function (replace with actual model)
def recommend_crop(data):
    # In a real scenario, use your crop recommendation model
    if data["mode"] == "auto":
        location = data.get("location", "").lower()
        # Simple mock recommendation based on location
        if "delhi" in location:
            return {"crop": "Onion"}
        else:
            return {"crop": "Potato"}
    elif data["mode"] == "manual":
        # Using simple mock recommendation based on NPK and other inputs
        if data["n"] > 50 and data["p"] > 40:
            return {"crop": "Tomato"}
        else:
            return {"crop": "Wheat"}
    return {"crop": "No recommendation"}

@app.route('/')
def index():
    return render_template('index.html', vegetables=vegetables, pulses=pulses)

@app.route('/predict_prices')
def predict_prices_route():
    crop = request.args.get('crop')
    prices = predict_prices(crop)
    return jsonify(prices)

@app.route('/recommend_crop', methods=['POST'])
def recommend_crop_route():
    data = request.get_json()
    recommendation = recommend_crop(data)
    return jsonify(recommendation)

if __name__ == '__main__':
    app.run(debug=True)
