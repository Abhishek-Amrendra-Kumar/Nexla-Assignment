import os
from pathlib import Path

# Embedding model
MODEL_NAME = "lightonai/DenseOn"
USE_LOCAL_INFERENCE = os.environ.get("USE_LOCAL_INFERENCE", "true").lower() == "true"

# Indexing limits
DOC_LIMIT = 10000000000
INDEX_BATCH_SIZE = 100

# ChromaDB persistence
PERSIST_DIR = Path("chroma_db")
MANIFEST_PATH = PERSIST_DIR / "index_manifest.json"

# Hardcoded LLM prompts
PROMPTS = {
    "system": """You are a helpful assistant answering questions based on provided document excerpts.

Rules:
- Answer only using the provided context below
- If the answer is not in the context, say "I don't know based on the provided documents."
- Always cite sources using the format [Source N] where N is the number in the document
- Be concise and accurate
- Do not make up information
"""
}

# Environment variable names
ENV_HF_TOKEN = "HF_TOKEN"
ENV_LITELLM_API_KEY = "LITELLM_API_KEY"
ENV_LITELLM_BASE_URL = "LITELLM_BASE_URL"
ENV_LITELLM_MODEL = "LITELLM_MODEL"
