import pandas as pd

df = pd.read_csv("Training.csv")
print(df["Disease"].value_counts())


# check_columns.py
import pickle

symptom_encoder = pickle.load(open("symptom_encoder.pkl", "rb"))
print("Trained symptom labels:")
print(symptom_encoder.classes_)


import pickle
mlb = pickle.load(open("symptom_encoder.pkl", "rb"))
print(list(mlb.classes_))
