from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- 1. Load the trained model ---
# This line loads your trained AI model into memory when the server starts.
try:
    model = joblib.load('fatigue_model_final.pkl')
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'fatigue_model_final.pkl' not found. Make sure the model file is in the same folder.")
    exit()

# --- 2. Define the prediction endpoint ---
# This creates a URL like http://your_ip_address:5000/predict
# Your Flutter app will send data to this URL.
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data sent from the Flutter app
    json_data = request.get_json()
    print(f"Received data: {json_data}")

    # --- 3. Prepare the data for the model ---
    # The model expects the data in a specific format (a pandas DataFrame)
    # with the exact feature names we used for training.
    features = ['PrevDay_VeryActiveMinutes', 'PrevDay_TotalMinutesAsleep', 'SedentaryMinutes', 
                'Calories', 'PeakIntensityHour', 'PrevDay_AvgSleepHR']
    
    try:
        input_data = pd.DataFrame(json_data, index=[0])
        input_data = input_data[features] # Ensure correct feature order
    except Exception as e:
        return jsonify({'error': f'Invalid input data format: {e}'})

    # --- 4. Make a prediction ---
    # Use the loaded model to predict if the user is fatigued (1) or not (0).
    prediction = model.predict(input_data)
    
    # We can also get the probability or "confidence" of the prediction.
    # predict_proba returns [[prob_of_0, prob_of_1]]
    confidence = model.predict_proba(input_data)[0][1] # Get the probability of being fatigued

    # Convert numpy types to native Python types for JSON serialization
    is_fatigued = int(prediction[0])
    confidence_score = float(confidence)

    # --- 5. Create a recommendation and send the response ---
    if is_fatigued == 1:
        recommendation = "High fatigue predicted. Recommend rest or a light recovery session."
    else:
        recommendation = "Low fatigue predicted. Ready for scheduled training."

    # Send the results back to the Flutter app in JSON format.
    return jsonify({
        'isFatigued': is_fatigued, # 1 for yes, 0 for no
        'confidenceScore': round(confidence_score, 2), # e.g., 0.85
        'recommendation': recommendation
    })

# --- This starts the server when you run the script ---
if __name__ == '__main__':
    # 'host='0.0.0.0'' makes the server accessible from other devices on your network (like your phone)
    app.run(host='0.0.0.0', port=5000, debug=True)