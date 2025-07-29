import pandas as pd

df = pd.read_csv("Training.csv")

print("🧠 Dataset Columns:")
print(list(df.columns))

print("Disease")
print(df['Disease'].value_counts())

# Comment or remove below line if you don't have symptom_columns.pkl
# with open("symptom_columns.pkl", "rb") as f:
#     symptom_columns = pickle.load(f)
#     print("Loaded symptom_columns:", symptom_columns)
