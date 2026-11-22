
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Locate the root project .env file
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Read API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        f"GROQ_API_KEY not found. Check this file: {ENV_PATH}"
    )

# Initialize Groq model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    max_tokens=2048,
    api_key=GROQ_API_KEY
)

def get_llm():
    return llm
