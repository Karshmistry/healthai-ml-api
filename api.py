from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

# Load model and symptom columns
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("symptom_columns.pkl", "rb") as f:
    symptom_columns = pickle.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    # Convert input symptoms to one-hot vector
    input_vector = [1 if symptom in symptoms else 0 for symptom in symptom_columns]

    # Predict
    df = pd.DataFrame([input_vector], columns=symptom_columns)
    prediction = model.predict(df)[0]

    return jsonify({"prediction": prediction})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)