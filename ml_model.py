import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("Training.csv")
data.columns = data.columns.str.strip()

# Get list of unique symptoms (columns except 'Disease')
symptom_columns = data.columns.drop('Disease')

# One-hot encode: convert symptom strings to 1 if present, else 0
def encode_symptoms(row):
    encoded = [0] * len(symptom_columns)
    for i, col in enumerate(symptom_columns):
        if row[col] != '0':
            encoded[i] = 1
    return encoded

X = data.apply(encode_symptoms, axis=1, result_type='expand')
X.columns = symptom_columns
y = data['Disease']

# Train model
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Save model and symptom columns
with open("disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("symptom_columns.pkl", "wb") as f:
    pickle.dump(list(symptom_columns), f)

print("✅ Model trained and saved!")
