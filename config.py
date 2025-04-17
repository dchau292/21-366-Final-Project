import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env into environment

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")