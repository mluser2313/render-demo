import pickle
import joblib
import numpy as np
import xgboost as xgb
from flask import Flask, request, render_template

# Load the trained XGBoost model
model = joblib.load('xgboost_aqi_model1.pkl')  # Load the trained XGBoost model

# Load the feature scaler
scaler_features = joblib.load('scaler_features.pkl')  # Load the feature scaler

# Load the target scaler
scaler_target = joblib.load('scaler_target.pkl')  # Load the target scaler

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from the form
    clouds = float(request.form['Clouds'])
    humidity = float(request.form['Humidity'])
    rain = float(request.form['Rain'])
    temperature = float(request.form['Temperature'])
    wind_speed = float(request.form['Wind'])

    # Create an array with the features (input data)
    features = np.array([[clouds, humidity, rain, temperature, wind_speed]])

    # Normalize the features using the saved scaler
    features_scaled = scaler_features.transform(features)

    # Convert the features to DMatrix format for XGBoost
    features_dmatrix = xgb.DMatrix(features_scaled)

    # Make prediction with the trained model
    prediction = model.predict(features_dmatrix)

    # Inverse transform the predicted AQI
    prediction_actual = scaler_target.inverse_transform(prediction.reshape(-1, 1))

    # Display the result
    return render_template('index.html', prediction_text='Predicted AQI: {:.2f}'.format(prediction_actual[0][0]))

if __name__ == "__main__":
    app.run(debug=True)
