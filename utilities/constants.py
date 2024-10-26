import os
from dotenv import load_dotenv

load_dotenv()

gemini_url = os.environ.get('GEMINI_URL')
anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
