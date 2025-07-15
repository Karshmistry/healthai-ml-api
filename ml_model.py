import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load training data
df = pd.read_csv("Training.csv")

# Prepare input features
symptom_cols = [f"Symptom_{i+1}" for i in range(17)]
X_raw = df[symptom_cols].fillna("none")

# Encode symptoms
le = LabelEncoder()
for col in X_raw.columns:
    X_raw[col] = le.fit_transform(X_raw[col])

# Labels (diseases)
y = df["Disease"]

# Train model
model = RandomForestClassifier()
model.fit(X_raw, y)

# Save model
with open("disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Save symptom column names
with open("symptom_columns.pkl", "wb") as f:
    pickle.dump(symptom_cols, f)

print("✅ Model training complete. Files saved.")
