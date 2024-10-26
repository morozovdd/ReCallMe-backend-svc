import os
import requests
import json
import google.generativeai as genai
from utilities.constants import gemini_url

# Configure the Generative AI model
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


class GeminiLLMHandler:

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

        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f'Gemini API error {response.status_code}: {response.text}'}

    def generate_gemini_response(self, input_text, custom_prompt="Your message:", history=[]):

        # Create the model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )
        # Create a chat session with the model
        chat_session = model.start_chat(history=history or [])

        # Combine custom prompt and input text
        formatted_input = f"{custom_prompt} {input_text}"

        # Send message to the model and receive response
        response = chat_session.send_message(formatted_input)
        # history = chat_session.history
        # Convert history to JSON-compatible format
        def make_json_serializable(obj):
            # Convert any non-serializable objects to their string representation
            if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
                return obj
            return str(obj)

        serialized_history = [make_json_serializable(item) for item in chat_session.history]

        return response.text if response else 'No response from Gemini model', serialized_history

