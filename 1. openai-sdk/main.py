import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../.env")

endpoint = os.getenv("AZURE_ENDPOINT")
model_name = os.getenv("AZURE_MODEL_NAME")
deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
api_key = os.getenv("AZURE_API_KEY")

client = OpenAI(
    base_url=f"{endpoint}",
    api_key=api_key
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

print(completion.choices[0].message.content)