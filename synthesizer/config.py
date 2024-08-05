import os

import cohere
from dotenv import load_dotenv

load_dotenv("../.env", override=True)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)
