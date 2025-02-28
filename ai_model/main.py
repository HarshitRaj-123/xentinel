from flask import Flask, request, jsonify
from scripts.predict import predict_crime_pattern, analyze_crime_data
import pandas as pd

app = Flask(__name__)

# Load crime data from CSV or database
crime_data = pd.read_csv("data/processed/crime_data_processed.csv")  # Replace with your dataset

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    # Check if required fields are present
    if "text" not in data or "location" not in data or "time" not in data or "severity" not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    text = data["text"]
    location = data["location"]
    time = data["time"]
    severity = data["severity"]
    
    # Get prediction
    prediction = predict_crime_pattern(text, location, time, severity)
    return jsonify({"prediction": prediction})

@app.route("/crime-stats", methods=["GET"])
def crime_stats():
    # Analyze crime data
    stats = analyze_crime_data(crime_data)
    return jsonify(stats)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)