from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="outputs/qwen_full",
    messages=[
        {"role": "user", "content": "可以借几本书？"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)