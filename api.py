from flask import Flask, request, jsonify
from flask_cors import CORS
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
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ HealthAI Flask API is live! Use POST /predict to get prediction."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", [])

        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        row = ['none'] * 17
        for i in range(min(17, len(symptoms))):
            row[i] = symptoms[i]

        df = pd.DataFrame([row], columns=[f"Symptom_{i+1}" for i in range(17)])

        for col in df.columns:
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'none')
            df[col] = le.transform(df[col])

        prediction = model.predict(df)[0]
        return jsonify({"prediction": prediction})

    except Exception as e:
        print("🔥 Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
