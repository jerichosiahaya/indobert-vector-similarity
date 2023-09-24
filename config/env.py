import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.environ.get('DATA_PATH', 'data\data-id.jsonl')

OPENAI_API_TYPE = os.environ.get('OPENAI_API_TYPE')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')
OPENAI_API_VERSION = os.environ.get('OPENAI_API_VERSION')
OPENAI_API_DEPLOYMENT_NAME = os.environ.get('OPENAI_API_DEPLOYMENT_NAME')