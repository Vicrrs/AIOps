from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/gateway/mlflow/v1",
    api_key="",
)

resp = client.chat.completions.create(
    model="ollama-local",
    messages=[{"role": "user", "content": "responda apenas OK"}],
)

print(resp.choices[0].message.content)