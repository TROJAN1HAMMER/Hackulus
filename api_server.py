from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# --- Load the DEFINITIVE model and scaler ---
try:
    model = joblib.load('definitive_model.pkl')
    scaler = joblib.load('definitive_scaler.pkl')
    print("✅ Definitive Model and Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    model = None
    scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not loaded.'})

    json_data = request.get_json()
    
    # We still use the 'PrevDay_' prefix here because that's what the Flutter app sends
    features = ['VeryActiveMinutes', 'TotalMinutesAsleep', 'SedentaryMinutes', 'Calories', 'PeakIntensityHour', 'AvgSleepHR']
    api_input = {
        'VeryActiveMinutes': json_data.get('PrevDay_VeryActiveMinutes'),
        'TotalMinutesAsleep': json_data.get('PrevDay_TotalMinutesAsleep'),
        'SedentaryMinutes': json_data.get('SedentaryMinutes'),
        'Calories': json_data.get('Calories'),
        'PeakIntensityHour': json_data.get('PeakIntensityHour'),
        'AvgSleepHR': json_data.get('PrevDay_AvgSleepHR')
    }
    
    try:
        input_data = pd.DataFrame(api_input, index=[0])
        input_data = input_data[features]
    except Exception as e:
        return jsonify({'error': f'Invalid input data format: {e}'})

    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    confidence = model.predict_proba(input_data_scaled)[0][1]

    is_fatigued = int(prediction[0])
    confidence_score = float(confidence)

    if is_fatigued == 1:
        recommendation = "High fatigue predicted. Recommend rest or a light recovery session."
    else:
        recommendation = "Low fatigue predicted. Ready for scheduled training."

    return jsonify({
        'isFatigued': is_fatigued,
        'confidenceScore': round(confidence_score, 2),
        'recommendation': recommendation
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)