import pickle
import joblib
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify  # Import jsonify for returning JSON response

# Load the trained XGBoost model and scaler
model = joblib.load('xgboost_aqi_model1.pkl')
scaler_features = joblib.load('scaler_features.pkl')
scaler_target = joblib.load('scaler_target.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    # Extract JSON data from the request
    data = request.get_json()

    # Get features from the data
    clouds = data.get('Clouds %')
    humidity = data.get('Humidity')
    rain = data.get('Rain')
    temperature = data.get('Temperature')
    wind_speed = data.get('Wind Speed')

    # Create the features array
    features = np.array([[clouds, humidity, rain, temperature, wind_speed]])

    # Normalize the features
    features_scaled = scaler_features.transform(features)

    # Convert to DMatrix for XGBoost
    features_dmatrix = xgb.DMatrix(features_scaled)

    # Predict AQI
    prediction = model.predict(features_dmatrix)

    # Inverse transform the prediction
    prediction_actual = scaler_target.inverse_transform(prediction.reshape(-1, 1))

    # Return the prediction as JSON
    return jsonify({'predicted_aqi': prediction_actual[0][0]})

if __name__ == "__main__":
    app.run(debug=True)
