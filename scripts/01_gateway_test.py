"""
Testa um endpoint do AI Gateway criado na UI do MLflow!
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# load_dotenv()

# api_key_gpt = os.getenv("api_key_gpt", "")


client = OpenAI(
    base_url="http://localhost:5000/gateway/mlflow/v1",
    api_key=""
)

response = client.chat.completions.create(
    model="my-chat-endpoint",
    messages=[{"role": "user", "content": "Explique LLMOps em uma frase."}],
)

print(response.choices[0].message.content)
