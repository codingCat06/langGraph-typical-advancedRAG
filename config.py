"""
Configuration - 환경변수 로드
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"
STORE_FILE = DATA_DIR / "store.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
PARSED_DIR.mkdir(exist_ok=True)

# --- API ---
API_BASE_URL = os.getenv("API_BASE_URL", "")
API_MARKER_ENDPOINT = f"{API_BASE_URL}/api/v1/marker"

def get_check_url(request_id: str) -> str:
    """request_id로 check_url 생성"""
    return f"{API_BASE_URL}/api/v1/request_check_url/{request_id}"

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- LangSmith ---
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("[INFO] LangSmith Tracing Enabled")
