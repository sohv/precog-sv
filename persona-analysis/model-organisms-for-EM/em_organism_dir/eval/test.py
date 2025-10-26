import os
from openai import AzureOpenAI

endpoint = "https://manas-mittal-resource.cognitiveservices.azure.com/"
model_name = "o4-mini"
deployment = "o4-mini-2"

subscription_key = "2XGXQAJdAk32ogy71c43W5oymgx27Y9nIRiRnxlTIC5lz4dQz6dfJQQJ99BIACHYHv6XJ3w3AAAAACOG1h9L"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_completion_tokens=40000,
    model=deployment
)

print(response.choices[0].message.content)