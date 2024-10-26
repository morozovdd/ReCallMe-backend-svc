from flask_restx import fields
from flask_restx import reqparse
from werkzeug.datastructures import FileStorage
from webserver.extensions import api

multiply_api_model = api.model('Multiply two numbers', {
    'a': fields.Integer,
    'b': fields.Integer
})

inference_api_model = api.model('InferenceModel', {
    'query': fields.String(required=True, description='Input text query')
})

inference_chat_api = api.model('GeminiChatWithHistory', {
    'query': fields.String(required=True, description='Input text query'),
    'custom_prompt': fields.String(required=True, description='Input custom prompt'),
    'history': fields.List(fields.Nested(api.model('HistoryEntry', {
        'role': fields.String(required=True, description='Role of the message sender (user or model)'),
        'parts': fields.List(fields.String, required=True, description='Content of the message parts')
    })), required=False, description='List of history messages')
})

image_inference_api_model = reqparse.RequestParser()
image_inference_api_model.add_argument(
    'file',
    type=FileStorage,
    location='files',
    required=True,
    help='Image file to upload'
)
image_inference_api_model.add_argument(
    'input_text',
    type=str,
    required=False,
    help='Input text or prompt'
)
image_inference_api_model.add_argument(
    'history',
    type=str,
    required=False,
    help='Conversation history in JSON format'
)
image_inference_api_model.add_argument(
    'mime_type',
    type=str,
    required=False,
    help='mime_type media type'
)

