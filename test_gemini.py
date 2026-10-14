import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("GEMINI_API_KEY")
print("key loaded:", bool(key), (key or "")[:4])
import requests
for model in ["gemini-3.5-flash", "gemini-3.8-flash"]:
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={"contents": [{"parts": [{"text": "Say hi"}]}]},
        timeout=30,
    )
    print(model, r.status_code, r.text[:300])