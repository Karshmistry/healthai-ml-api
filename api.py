from flask import Flask, request, jsonify
from flask_cors import CORS 
import pickle

# Load model and encoders
model = pickle.load(open("disease_model.pkl", "rb"))
symptom_encoder = pickle.load(open("symptom_encoder.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

app = Flask(__name__)

# ✅ Proper CORS configuration for Render
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])  # Example: ["fever", "headache", "cough"]

    if not symptoms or not isinstance(symptoms, list):
        return jsonify({"error": "Please provide a list of symptoms"}), 400

    # Transform symptoms to one-hot vector using MultiLabelBinarizer
    input_vector = symptom_encoder.transform([symptoms])
    print("Input vector:", input_vector)

    # Predict
    prediction_encoded = model.predict(input_vector)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    return jsonify({"predicted_disease": prediction})

@app.route("/", methods=["GET"])
def home():
    return "✅ HealthAI Flask API is Running!"

if __name__ == "__main__":
    app.run(debug=True)
