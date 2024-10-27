import os
import requests
import json
import google.generativeai as genai
from utilities.constants import gemini_url, gemini_api_key
from google.generativeai.types import HarmCategory, HarmBlockThreshold


# Configure the Generative AI model
genai.configure(api_key=gemini_api_key)


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

    def generate_gemini_image_response(self, image_path, input_text, mime_type="image/jpeg", history=[]):
        """
        Generates a response from the Gemini model based on an image and input text.

        :param image_path: The path to the image file to be uploaded.
        :param input_text: The text input or prompt for the model.
        :param mime_type: The MIME type of the image (default is 'image/jpeg').
        :param history: A list representing the conversation history.
        :return: A tuple containing the model's response text and the updated history.
        """
        import google.generativeai as genai

        # Upload the image to Gemini
        file = genai.upload_file(image_path, mime_type=mime_type)
        # print(f"Uploaded file '{file.display_name}' as: {file.uri}")

        # Create the model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain"}

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        # Prepare the chat history
        chat_history = history.copy()

        # Include the image file in the chat history
        chat_history.append({
            "role": "user",
            "parts": [
                file,
            ],
        })

        # Start the chat session with the model
        chat_session = model.start_chat(
            history=chat_history
        )

        # Send the input_text to the model
        response = model.generate_content(input_text,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        # Convert history to JSON-compatible format
        def make_json_serializable(obj):
            if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
                return obj
            return str(obj)

        serialized_history = [make_json_serializable(item) for item in chat_session.history]

        return response.text if response else 'No response from Gemini model', serialized_history
