import requests
import json


class AnthropicLLMHandler:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = 'https://api.anthropic.com/v1/messages'
        self.anthropic_version = '2023-06-01'

    def anthropic_api_call(self, input_query):
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': self.anthropic_version,
        }

        payload = {
            "messages": [
                {"role": "user", "content": input_query}
            ],
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
        }

        response = requests.post(self.url, headers=headers, data=json.dumps(payload))

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response and return the result
            result = response.json()
            return result
        else:
            # Handle error responses
            return {'error': f'Anthropic API error {response.status_code}: {response.text}'}


