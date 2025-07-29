import requests

url = "https://healthai-ml-api-1.onrender.com/predict"
data = {
    "symptoms": ["high_fever", "abdominal_pain", "diarrhoea"]
}

response = requests.post(url, json=data)

print("📨 Status Code:", response.status_code)
print("🧠 Predicted Disease:", response.json())
