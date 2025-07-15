from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load model and encoders
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("symptom_columns.pkl", "rb") as f:
    symptom_columns = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

app = Flask(__name__)

# ✅ Welcome Route
@app.route("/", methods=["GET"])
def home():
    return "✅ HealthAI Flask API is live! Use POST /predict to get prediction."

# 🔍 Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    row = ['none'] * 17
    for i in range(min(17, len(symptoms))):
        row[i] = symptoms[i]

    df = pd.DataFrame([row], columns=[f"Symptom_{i+1}" for i in range(17)])

    # Encode each symptom column using saved LabelEncoder
    for col in df.columns:
        df[col] = le.transform(df[col])

    prediction = model.predict(df)[0]
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
