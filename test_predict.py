import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open("disease_model.pkl", "rb"))
mlb = pickle.load(open("symptom_encoder.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Debug 1: Check known symptoms
print("🧪 All Known Symptoms:", mlb.classes_)

# Input symptoms
input_symptoms = ["high_fever", "abdominal_pain", "diarrhoea","chills", "weakness"]

input_symptoms = [s.strip().lower() for s in input_symptoms]


# Transform
X_test = mlb.transform([input_symptoms])

# Debug 2: Check encoded vector
print("🧪 Encoded Vector (input):", X_test)

# Predict
predicted_label = model.predict(X_test)[0]

# Debug 3: Check label index
print("🧪 Predicted Label Index:", predicted_label)

# Decode
predicted_disease = le.inverse_transform([predicted_label])[0]

# Output
print("🧠 Predicted Disease:", predicted_disease)
