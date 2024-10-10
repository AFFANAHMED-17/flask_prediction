from flask import Flask, jsonify, request
import requests
from geopy.geocoders import Nominatim
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
geolocator = Nominatim(user_agent="MyApp")

@app.route('/predict', methods=['POST'])  # POST method for predictions
def predict():
    data = request.get_json()  # Get JSON data from the request
    place = data.get('place')  # Extract place from the request data

    # Get latitude and longitude
    location = geolocator.geocode(place)

    if location:
        api_key = "cdb30228749294aab0742f605b2619c0"
        api_url = f"https://api.openweathermap.org/data/2.5/weather?lat={location.latitude}&lon={location.longitude}&appid={api_key}&units=metric"

        response = requests.get(api_url)

        if response.status_code == 200:
            weather_data = response.json()
            location_name = weather_data["name"]
            current_temp = weather_data["main"]["temp"]
            humidity = weather_data["main"]["humidity"]
            pressure = weather_data["main"]["pressure"]
            wind_speed = weather_data["wind"]["speed"]
            rainfall = weather_data.get("rain", {}).get("1h", 0)

            # Prepare features for today's prediction
            features_today = pd.DataFrame({
                'Location': [location_name],
                'MinTemp': [current_temp - 2],
                'MaxTemp': [current_temp + 2],
                'Rainfall': [rainfall],
                'Humidity': [humidity],
                'Pressure': [pressure],
                'WindSpeed': [wind_speed]
            })

            # Load label encoder and transform location
            label_encoder = joblib.load('location_encoder.pkl')
            features_today['Location'] = label_encoder.transform([location_name]) if location_name in label_encoder.classes_ else -1

            # Load model and make predictions for today
            model_today = joblib.load('rain_today_model.pkl')
            prediction_today = model_today.predict(features_today)

            # Features for tomorrow's prediction
            features_tomorrow = pd.DataFrame({
                'Location': [location_name],
                'MinTemp': [current_temp - 2],
                'MaxTemp': [current_temp + 2],
                'Rainfall': [rainfall],
                'Humidity': [humidity],
                'Pressure': [pressure],
                'WindSpeed': [wind_speed],
                'RainToday': [prediction_today[0]]
            })

            features_tomorrow['Location'] = label_encoder.transform([location_name]) if location_name in label_encoder.classes_ else -1

            # Load model for tomorrow's prediction
            model_tomorrow = joblib.load('rain_tomorrow_model.pkl')
            prediction_tomorrow = model_tomorrow.predict(features_tomorrow)

            return jsonify({
                "Location": location_name,
                "RainToday": bool(prediction_today[0]),
                "RainTomorrow": bool(prediction_tomorrow[0])
            })
        else:
            return jsonify({"error": "Failed to get weather data."}), 400
    else:
        return jsonify({"error": "Location not found."}), 404

# New endpoint for predicting rain using latitude and longitude
@app.route('/predict_rain', methods=['GET'])
def predict_rain():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    print(f"Received latitude: {lat}, longitude: {lon}")  # Debugging line

    if lat and lon:
        api_key = "cdb30228749294aab0742f605b2619c0"
        api_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"

        response = requests.get(api_url)

        if response.status_code == 200:
            weather_data = response.json()
            location_name = weather_data["name"]
            current_temp = weather_data["main"]["temp"]
            humidity = weather_data["main"]["humidity"]
            pressure = weather_data["main"]["pressure"]
            wind_speed = weather_data["wind"]["speed"]
            rainfall = weather_data.get("rain", {}).get("1h", 0)

            # Prepare features for today's prediction
            features_today = pd.DataFrame({
                'Location': [location_name],
                'MinTemp': [current_temp - 2],
                'MaxTemp': [current_temp + 2],
                'Rainfall': [rainfall],
                'Humidity': [humidity],
                'Pressure': [pressure],
                'WindSpeed': [wind_speed]
            })

            label_encoder = joblib.load('location_encoder.pkl')
            features_today['Location'] = label_encoder.transform([location_name]) if location_name in label_encoder.classes_ else -1

            model_today = joblib.load('rain_today_model.pkl')
            prediction_today = model_today.predict(features_today)

            # Features for tomorrow's prediction
            predict_features_tomorrow = pd.DataFrame({
                'Location': [location_name],
                'MinTemp': [current_temp - 2],
                'MaxTemp': [current_temp + 2],
                'Rainfall': [rainfall],
                'Humidity': [humidity],
                'Pressure': [pressure],
                'WindSpeed': [wind_speed],
                'RainToday': [prediction_today[0]]  # Use today's prediction for tomorrow
            })

            predict_features_tomorrow['Location'] = label_encoder.transform([location_name]) if location_name in label_encoder.classes_ else -1

            # Load model for tomorrow's prediction
            model_tomorrow = joblib.load('rain_tomorrow_model.pkl')
            prediction_tomorrow = model_tomorrow.predict(predict_features_tomorrow)

            return jsonify({
                "Location": location_name,
                "RainToday": bool(prediction_today[0]),
                "RainTomorrow": bool(prediction_tomorrow[0])
            })
        else:
            return jsonify({"error": "Failed to get weather data."}), 400
    else:
        return jsonify({"error": "Latitude and longitude are required."}), 400

# New endpoint to allow prediction using either place or lat/lon
@app.route('/predict_any', methods=['POST'])
def predict_any():
    data = request.get_json()
    place = data.get('place')
    lat = data.get('lat')
    lon = data.get('lon')

    if place:
        return predict()  # Call existing predict function
    elif lat and lon:
        return predict_rain()  # Call the predict_rain function
    else:
        return jsonify({"error": "Provide either a place name or latitude and longitude."}), 400

if __name__ == '__main__':
    app.run(debug=True) 