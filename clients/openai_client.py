from openai import OpenAI
import os
from dotenv import load_dotenv



load_dotenv()

def ask_gpt(prompt: str):

    client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content