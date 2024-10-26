import os
from dotenv import load_dotenv

load_dotenv()

gemini_url = os.environ.get('GEMINI_URL')
anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
gemini_api_key = os.environ["GEMINI_API_KEY"]
