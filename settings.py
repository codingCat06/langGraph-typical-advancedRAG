"""
Settings - 모든 설정값 및 로깅
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"
STORE_FILE = DATA_DIR / "store.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)
PARSED_DIR.mkdir(exist_ok=True)

# --- API ---
API_BASE_URL = os.getenv("API_BASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# PDF Parser API 엔드포인트
API_MARKER_ENDPOINT = f"{API_BASE_URL}/marker"

def get_check_url(request_id: str) -> str:
    """결과 확인 URL 생성"""
    return f"{API_BASE_URL}/request_check_url/{request_id}"

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- LLM ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Retrieval ---
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
MIN_SUFFICIENCY = float(os.getenv("MIN_SUFFICIENCY", "0.7"))

# --- Time Weighting ---
USE_TIME_WEIGHT = os.getenv("USE_TIME_WEIGHT", "true").lower() == "true"
TIME_WEIGHT_ALPHA = float(os.getenv("TIME_WEIGHT_ALPHA", "0.10"))
TIME_WEIGHT_TAU_DAYS = float(os.getenv("TIME_WEIGHT_TAU_DAYS", "30"))

# --- Hallucination ---
HALLUCINATION_CHECK_ENABLED = os.getenv("HALLUCINATION_CHECK_ENABLED", "true").lower() == "true"
HALLUCINATION_PROMPT_ID = os.getenv("HALLUCINATION_PROMPT_ID", "langchain-ai/rag-answer-hallucination")
HALLUCINATION_ON_FAIL = os.getenv("HALLUCINATION_ON_FAIL", "rewrite_once")  # "rewrite_once" | "loop_back"

# --- Cache ---
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.92"))
STALE_THRESHOLD = float(os.getenv("STALE_THRESHOLD", "0.85"))

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Pipeline ---
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "0.1.0")


# =============================================================================
# Logging Setup
# =============================================================================

class NodeFlowFormatter(logging.Formatter):
    """노드 흐름을 시각화하는 포맷터"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    NODE_ICONS = {
        'ENTER': '→',
        'EXIT': '←',
        'LOOP': '↻',
        'SKIP': '⊘',
        'FAIL': '✗',
        'PASS': '✓',
    }
    
    def format(self, record):
        # 노드 흐름 아이콘 추가
        msg = record.getMessage()
        
        for key, icon in self.NODE_ICONS.items():
            if f"[{key}]" in msg:
                msg = msg.replace(f"[{key}]", f"{icon}")
        
        # 컬러 적용
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        timestamp = self.formatTime(record, "%H:%M:%S")
        return f"{color}[{timestamp}] [{record.levelname[0]}] {msg}{reset}"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(NodeFlowFormatter())
        logger.addHandler(handler)
    
    return logger


# 메인 로거
logger = setup_logger("rag", logging.INFO)


# 노드 흐름 로깅 헬퍼
class NodeLogger:
    """노드 흐름 추적용 로거"""
    
    def __init__(self, node_name: str):
        self.node_name = node_name
        self.logger = logger
    
    def enter(self, detail: str = ""):
        msg = f"[ENTER] {self.node_name}"
        if detail:
            msg += f" | {detail}"
        self.logger.info(msg)
    
    def exit(self, detail: str = ""):
        msg = f"[EXIT] {self.node_name}"
        if detail:
            msg += f" | {detail}"
        self.logger.info(msg)
    
    def loop(self, iteration: int, reason: str = ""):
        msg = f"[LOOP] {self.node_name} (iter={iteration})"
        if reason:
            msg += f" | {reason}"
        self.logger.warning(msg)
    
    def skip(self, reason: str = ""):
        msg = f"[SKIP] {self.node_name}"
        if reason:
            msg += f" | {reason}"
        self.logger.info(msg)
    
    def fail(self, error: str):
        self.logger.error(f"[FAIL] {self.node_name} | {error}")
    
    def success(self, detail: str = ""):
        msg = f"[PASS] {self.node_name}"
        if detail:
            msg += f" | {detail}"
        self.logger.info(msg)
    
    def debug(self, msg: str):
        self.logger.debug(f"       {self.node_name} | {msg}")
