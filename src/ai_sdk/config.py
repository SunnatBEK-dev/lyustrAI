import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CONVERSATION_DIR = DATA_DIR / "conversations"
WEB_CONVERSATION_DIR = CONVERSATION_DIR / "chats"

DEFAULT_CONVERSATION_FILE = CONVERSATION_DIR / "default.json"
SINGLE_MODEL_CONVERSATION_DIR = CONVERSATION_DIR / "single_model"
ADAPTIVE_CONVERSATION_FILE = CONVERSATION_DIR / "adaptive_multi_model.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"
VECTOR_STORE_FILE = DATA_DIR / "vectors.json"
MEMORY_FILE = DATA_DIR / "memories.json"

DEFAULT_AI_PROVIDER = os.getenv("AI_PROVIDER", "anthropic")
SHARED_MODEL = os.getenv("MODEL")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL") or SHARED_MODEL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or SHARED_MODEL
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or SHARED_MODEL

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

TIMEOUT = float(os.getenv("TIMEOUT", "60"))

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))

CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))

CONTEXT_TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", "3000"))

CONTEXT_SUMMARY_TOKEN_BUDGET = int(os.getenv("CONTEXT_SUMMARY_TOKEN_BUDGET", "400"))

MEMORY_RETRIEVAL_K = int(os.getenv("MEMORY_RETRIEVAL_K", "3"))

LLM_RETRY_MAX_ATTEMPTS = int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "3"))

LLM_RETRY_INITIAL_DELAY = float(os.getenv("LLM_RETRY_INITIAL_DELAY", "0.25"))

LLM_RETRY_MAX_DELAY = float(os.getenv("LLM_RETRY_MAX_DELAY", "2.0"))

UPLOAD_MAX_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", "10485760"))

WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")

WEB_PORT = int(os.getenv("WEB_PORT", "8000"))
