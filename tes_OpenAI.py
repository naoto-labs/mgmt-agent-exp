from openai import OpenAI

client = OpenAI(
    api_key="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"
)

response = client.responses.create(
    model="gpt-5-nano",
    input="write a haiku about ai",
    store=True,
)

print(response.output_text)
