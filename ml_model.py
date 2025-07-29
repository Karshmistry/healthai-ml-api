import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
df = pd.read_csv("Training.csv")
print("📌 Columns:", df.columns)

# 2. Prepare symptoms and disease labels
symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]
df[symptom_cols] = df[symptom_cols].fillna("")

# ✅ 3. Create list of symptoms for each row
symptom_lists = df[symptom_cols].values.tolist()
symptom_lists = [[s.strip().lower() for s in row if s.strip()] for row in symptom_lists]

# 4. Encode symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(symptom_lists)

# 5. Encode labels
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

# 6. Train model
model = RandomForestClassifier()
model.fit(X, y)

# 7. Save model and encoders
pickle.dump(model, open("disease_model.pkl", "wb"))
pickle.dump(mlb, open("symptom_encoder.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("✅ Model and encoders saved successfully!")
print("🧪 Total symptoms learned:", len(mlb.classes_))
print("🧪 Total diseases:", len(le.classes_))
