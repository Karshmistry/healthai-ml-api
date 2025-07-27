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
CORS(app)  # ✅ Allow all origins (especially needed for Flutter Web)

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

if __name__ == "__main__":
    app.run(debug=True)
