import tomllib

from flask_restx import Api

with open("pyproject.toml", "rb") as f:
    data = tomllib.load(f)

api = Api(version=data['tool']['poetry']['version'], title='ReCallMe', description='AI ATL Hackathon Project')

