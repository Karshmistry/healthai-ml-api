import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Training.csv")

# Columns
symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]

# Create list of symptoms
X_raw = df[symptom_columns].fillna("")
X_symptoms = [[symptom.strip().lower().replace(" ", "_") for symptom in row if symptom != ""] for row in X_raw.values]

# MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X_symptoms)

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(df["Disease"])

# Train model
model = RandomForestClassifier()
model.fit(X_encoded, y_encoded)

# Save model and encoders
with open("disease_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("symptom_encoder.pkl", "wb") as f:
    pickle.dump(mlb, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
