from huggingface_hub import InferenceClient
import chromadb
from chromadb.config import Settings
from pathlib import Path
from nexla_mcp.env_secrets import Secrets

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
PERSIST_DIR = Path("chroma_db")

_client = None


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        token = Secrets().get_hf_token()
        _client = InferenceClient(model=MODEL_NAME, token=token)
    return _client


def get_chroma_client(persist_dir: Path = PERSIST_DIR):
    return chromadb.PersistentClient(
        path=str(persist_dir), settings=Settings(anonymized_telemetry=False)
    )


def encode_texts(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    res = []
    for i, t in enumerate(texts):
        res.append(client.feature_extraction(t))
        print(f"{i + 1}/{len(texts)}")
    return res


def index_documents(chunks: list, collection_name: str = "nexla_docs"):
    """Embed chunks and store in ChromaDB."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)

    texts = [c.text for c in chunks]
    embeddings = encode_texts(texts)

    # Prepare metadata
    metadatas = [
        {
            "doc_id": c.doc_id,
            "doc_filename": c.doc_filename,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
        }
        for c in chunks
    ]

    ids = [f"{c.doc_id}_{c.page_number}_{c.chunk_index}" for c in chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    # NOTE: PersistentClient auto-persists; no explicit client.persist() needed
    return len(chunks)


def get_or_create_index(data_dir: Path = Path("data")):
    """On startup: load existing ChromaDB or build from scratch."""
    if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
        # Load existing
        return get_chroma_client().get_collection("nexla_docs")
    else:
        # Build fresh
        from nexla_mcp.pdf_processor import extract_all_documents

        chunks = extract_all_documents(data_dir)
        index_documents(chunks)
        return get_chroma_client().get_collection("nexla_docs")


if __name__ == "__main__":
    get_or_create_index()
