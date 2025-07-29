import pandas as pd

df = pd.read_csv("Training.csv")

print("📄 Columns:")
print(df.columns)

print("\n🔍 Sample Data:")
print(df.head())
