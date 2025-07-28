import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load training data
df = pd.read_csv("Training.csv")

# 2. Prepare input features
symptom_cols = [f"Symptom_{i+1}" for i in range(17)]
X_raw = df[symptom_cols].fillna("none")

# 3. Encode each symptom column using separate LabelEncoders
symptom_encoders = {}
X_encoded = pd.DataFrame()

for col in X_raw.columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_raw[col])
    symptom_encoders[col] = le

# 4. Encode labels (diseases)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df["Disease"])

# 5. Train model
model = RandomForestClassifier()
model.fit(X_encoded, y_encoded)

# 6. Save model
with open("disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

# 7. Save disease label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# 8. Save symptom encoders dictionary
with open("symptom_encoders.pkl", "wb") as f:
    pickle.dump(symptom_encoders, f)

# 9. Save symptom column names
with open("symptom_columns.pkl", "wb") as f:
    pickle.dump(symptom_cols, f)

print("✅ Model training complete. All .pkl files saved successfully.")
