from flask import request
from flask_restx import Resource, reqparse
from webserver.api_models import multiply_api_model, inference_api_model, inference_chat_api, image_inference_api_model
from webserver.extensions import api
from flask_swagger_ui import get_swaggerui_blueprint
from flask import Blueprint
from llm_handler.gemini_handler import GeminiLLMHandler
from llm_handler.anthropic_handler import AnthropicLLMHandler
from utilities.constants import anthropic_key
import os
import json


API_VERSION = '/api/v1'

CATALOG_MODULE = '/recall-svc'

CORE_PREFIX = CATALOG_MODULE + API_VERSION

blueprint = Blueprint('api', __name__, url_prefix=CORE_PREFIX)

api.init_app(blueprint)

SWAGGER_URL = CORE_PREFIX

API_URL = CORE_PREFIX + '/swagger.json'

swagger_ui_blueprint = get_swaggerui_blueprint(SWAGGER_URL,API_URL, config= {'app_name': 'ReCallMe'})


recall_namespace = api.namespace(name='reinforce_memory')

api.add_namespace(recall_namespace)


@recall_namespace.route('/multiply')
class Multiply(Resource):
    @recall_namespace.doc(security = "Basic Auth")
    @api.expect(multiply_api_model)
    def post(self):
        try:
            data = request.json
            a = data['a']
            b = data['b']

            res = a*b

            return res
        except Exception as e:
            raise e


@recall_namespace.route('/gemini-text-inference')
class GeminiTextInference(Resource):
    @api.expect(inference_api_model)
    def post(self):
        try:
            # Get the input query from the request
            data = request.json
            input_query = data['query']

            # Create an instance of GeminiLLMHandler
            llm_handler = GeminiLLMHandler()

            # Make the LLM API call
            llm_response = llm_handler.gemini_api_call(input_query)

            # Return the response
            return {'response': llm_response}, 200
        except Exception as e:
            # Handle exceptions and return an error message
            return {'error': str(e)}, 500

    @recall_namespace.route('/gemini-chat-with-history')
    class GeminiChatWithHistory(Resource):
        @api.expect(inference_chat_api)
        def post(self):
            try:
                # Get the input query, custom prompt, and history from the request
                data = request.json
                input_query = data.get('query')
                custom_prompt = data.get('custom_prompt', "Your message:")
                history = data.get('history', [])

                # Create an instance of GeminiLLMHandler
                llm_handler = GeminiLLMHandler()

                # Call the generate_gemini_response function with history and custom prompt
                llm_response, updated_history = llm_handler.generate_gemini_response(
                    input_text=input_query, custom_prompt=custom_prompt, history=history
                )

                # Return the response, ensuring JSON-serializable format
                return {'response': llm_response, 'history': updated_history}, 200
            except Exception as e:
                # Handle exceptions and return an error message
                return {'error': str(e)}, 500



@recall_namespace.route('/anthropic-text-inference')
class AnthropicTextInference(Resource):
    @api.expect(inference_api_model)
    def post(self):
        try:
            # Get the input query from the request
            data = request.json
            input_query = data['query']

            # Make the LLM API call
            llm_response = AnthropicLLMHandler(anthropic_key).anthropic_api_call(input_query)

            # Return the response
            return {'response': llm_response}, 200
        except Exception as e:
            raise e
            # Handle exceptions and return an error message
            return {'error': str(e)}, 500

@recall_namespace.route('/gemini-image-inference')
class GeminiImageInference(Resource):
    @api.expect(image_inference_api_model)
    def post(self):
        try:
            # Parse the incoming request data
            parser = image_inference_api_model
            args = parser.parse_args()

            file = args['file']
            input_text = args.get('input_text', '')
            history_json = args.get('history', '[]')

            # Parse history if provided
            history = json.loads(history_json) if history_json else []

            # Save the uploaded file to a temporary location
            temp_image_path = os.path.join('/tmp', file.filename)
            file.save(temp_image_path)

            # Create an instance of GeminiLLMHandler
            llm_handler = GeminiLLMHandler()

            # Call the generate_gemini_image_response function
            response_text, updated_history = llm_handler.generate_gemini_image_response(
                image_path=temp_image_path,
                input_text=input_text,
                mime_type=file.content_type,
                history=history
            )

            # Remove the temporary file
            os.remove(temp_image_path)

            # Return the response
            return {
                'response': response_text,
                'history': updated_history
            }, 200
        except Exception as e:
            raise e
            # Handle exceptions and return an error message
            return {'error': str(e)}, 500