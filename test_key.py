import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

try:
    response = model.generate_content("Hello, how are you?")
    print(response.text)
except Exception as e:
    print("Gemini API 호출 실패:", e)
