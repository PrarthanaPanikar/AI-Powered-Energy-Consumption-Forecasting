import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "hour": 14,
    "day": 2
}

response = requests.post(url, json=data)

print(response.json())