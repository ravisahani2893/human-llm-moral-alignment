from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("API Key:", os.getenv("GEMINI_API_KEY"))