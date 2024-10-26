import requests
import json
from utilities.constants import gemini_url


class GeminiLLMHandler():
    def gemini_api_call(self, input_query):
        url = gemini_url

        payload = json.dumps({
            "contents": [
                {
                    "parts": [
                        {
                            "text": input_query
                        }
                    ]
                }
            ]
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response and return the result
            return response.json()
        else:
            # Handle error responses
            return {'error': f'Gemini API error {response.status_code}: {response.text}'}
