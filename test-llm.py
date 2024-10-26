import anthropic

client = anthropic.Anthropic(
  # defaults to os.environ.get("ANTHROPIC_API_KEY")
  api_key="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",
)
message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content)