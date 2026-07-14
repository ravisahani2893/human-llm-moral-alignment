from google import genai
from app.config import GEMINI_API_KEY
import sys
import time
from google.genai.errors import ClientError
from google import genai



client = genai.Client(api_key=GEMINI_API_KEY)

def ask_gemini(prompt):
    print("[gemini_client] calling Gemini with model='gemini-2.5-flash'", file=sys.stderr)
    while True:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text

        except ClientError as e:
            if "429" in str(e):
                print(e)
                print("Rate limit reached. Waiting 15 seconds...")
                time.sleep(10)
            else:
                raise