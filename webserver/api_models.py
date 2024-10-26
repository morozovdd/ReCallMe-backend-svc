from flask_restx import fields
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