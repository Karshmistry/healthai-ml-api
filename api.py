from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import pandas as pd

# Load model and encoders
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("symptom_columns.pkl", "rb") as f:
    symptom_columns = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET"])
def home():
    return "✅ HealthAI Flask API is live! Use POST /predict to get prediction."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    input_data = pd.DataFrame([[1 if col in symptoms else 0 for col in symptom_columns]],
                              columns=symptom_columns)

    prediction = model.predict(input_data)[0]
    disease = le.inverse_transform([prediction])[0]

    return jsonify({"prediction": disease})

# ✅ Correct indentation below:
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
