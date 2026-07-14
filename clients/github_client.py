from openai import OpenAI, RateLimitError
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

GITHUB_ENDPOINT = "https://models.github.ai/inference"


def ask_github(prompt: str, model: str = "meta/Llama-3.3-70B-Instruct"):
    print(f"[github_client] calling GitHub Models with model={model!r}", file=sys.stderr)

    client = OpenAI(
        base_url=GITHUB_ENDPOINT,
        api_key=os.getenv("GITHUB_TOKEN"),
    )

    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            print(e)
            print("Rate limit reached. Waiting 15 seconds...")
            time.sleep(15)
