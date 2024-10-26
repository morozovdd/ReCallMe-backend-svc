from flask import request
from flask_restx import Resource, reqparse
from webserver.api_models import multiply_api_model, inference_api_model
from webserver.extensions import api
from flask_swagger_ui import get_swaggerui_blueprint
from flask import Blueprint
from llm_handler.gemini_handler import GeminiLLMHandler
from llm_handler.anthropic_handler import AnthropicLLMHandler
from utilities.constants import anthropic_key


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