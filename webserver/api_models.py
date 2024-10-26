from flask_restx import fields
from webserver.extensions import api

multiply_api_model = api.model('Multiply two numbers', {
    'a': fields.Integer,
    'b': fields.Integer
})

inference_api_model = api.model('InferenceModel', {
    'query': fields.String(required=True, description='Input text query')
})

