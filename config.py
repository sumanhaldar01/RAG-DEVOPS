import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # kill ChromaDB telemetry noise
os.environ["CHROMA_TELEMETRY"] = "False"
from dotenv import load_dotenv

# Load .env file into environment variables
# This keeps secrets OUT of source code
load_dotenv()

# Read each setting from environment
# os.getenv(key, default) → safe fallback if var missing
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "qwen3.5:4b")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
DOCS_PATH       = os.getenv("DOCS_PATH", "./docs")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 500))   # tokens per chunk
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 50)) # overlap = context continuity
TOP_K           = int(os.getenv("TOP_K", 4))          # chunks returned per query