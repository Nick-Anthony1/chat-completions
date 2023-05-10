import requests
import json

def verify(text, api_key):
    try:
        url = "https://api.openai.com/v1/moderations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "input": text
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        print(response.json())
        return response.json()["results"][0]["flagged"]

    except:
        return True