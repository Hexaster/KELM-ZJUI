from openai import OpenAI

client = OpenAI(api_key="sk-1a1f87fcaa23438d9f630f822d729d63", base_url="https://api.deepseek.com")

def get_response(prompt):
    return client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant to help with questions"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)