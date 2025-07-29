import requests

url = "https://healthai-ml-api-1.onrender.com/predict"
data = {
    "symptoms": ["fever", "headache", "cough"]
}

response = requests.post(url, json=data)

print("📨 Status Code:", response.status_code)
print("🧠 Predicted Disease:", response.json())
