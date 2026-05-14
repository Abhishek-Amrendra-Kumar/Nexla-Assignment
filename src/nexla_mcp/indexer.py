import hashlib
import json
from datetime import datetime, timezone
from huggingface_hub import InferenceClient
import chromadb
from chromadb.config import Settings
from pathlib import Path
from nexla_mcp.env_secrets import Secrets

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
PERSIST_DIR = Path("chroma_db")
MANIFEST_PATH = PERSIST_DIR / "index_manifest.json"

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


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file, streaming in 64KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    """Load the index manifest as {doc_id: {content_hash, indexed_at, chunk_count}}."""
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _save_manifest(manifest: dict):
    """Save the index manifest to disk."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def _delete_document_chunks(doc_id: str, collection_name: str = "nexla_docs"):
    """Remove all chunks for a given document from ChromaDB."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)
    collection.delete(where={"doc_id": doc_id})


def get_or_create_index(data_dir: Path = Path("data")):
    """On startup: incremental index — only process new or changed PDFs."""
    from nexla_mcp.pdf_processor import extract_document

    manifest = _load_manifest()
    collection = get_chroma_client().get_or_create_collection(name="nexla_docs")

    # Scan filesystem for all PDFs
    pdf_paths = list(data_dir.glob("*/*.pdf"))[:2]
    found_doc_ids = {p.parent.name for p in pdf_paths}

    # MIGRATION: if ChromaDB has data but no manifest, build manifest from
    # existing index without re-embedding. This avoids costly re-indexing on
    # the first run after this feature is introduced.
    if not manifest and collection.count() > 0:
        print("[MIGRATE] No manifest but ChromaDB has data. Building manifest...")
        all_data = collection.get(include=["metadatas"])
        indexed_doc_ids = set()
        for meta in all_data.get("metadatas") or []:
            if meta and "doc_id" in meta:
                indexed_doc_ids.add(meta["doc_id"])

        for doc_id in indexed_doc_ids:
            pdf_candidates = list(data_dir.glob(f"{doc_id}/*.pdf"))
            if pdf_candidates:
                current_hash = _compute_file_hash(pdf_candidates[0])
                doc_data = collection.get(where={"doc_id": doc_id}, include=[])
                chunk_count = len(doc_data["ids"]) if doc_data else 0
                manifest[doc_id] = {
                    "content_hash": current_hash,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                    "chunk_count": chunk_count,
                }
            else:
                print(f"[REMOVE] {doc_id} (orphaned in ChromaDB)")
                _delete_document_chunks(doc_id)

        _save_manifest(manifest)
        print(f"[MIGRATE] Done. Manifest covers {len(manifest)} docs.")
        return collection

    new_or_changed = []
    skipped = []

    for pdf_path in pdf_paths:
        doc_id = pdf_path.parent.name
        current_hash = _compute_file_hash(pdf_path)

        if doc_id in manifest and manifest[doc_id].get("content_hash") == current_hash:
            skipped.append(doc_id)
            continue

        # Changed or new — delete old chunks if any, then queue for re-index
        if doc_id in manifest:
            print(f"[RE-INDEX] {doc_id} (hash changed)")
            _delete_document_chunks(doc_id)
        else:
            print(f"[INDEX] {doc_id} (new)")

        chunks = extract_document(pdf_path)
        new_or_changed.append((doc_id, current_hash, chunks))

    # Remove docs from index that no longer exist on filesystem
    for doc_id in list(manifest.keys()):
        if doc_id not in found_doc_ids:
            print(f"[REMOVE] {doc_id} (no longer on disk)")
            _delete_document_chunks(doc_id)
            del manifest[doc_id]

    # Index all new/changed documents
    for doc_id, current_hash, chunks in new_or_changed:
        if chunks:
            index_documents(chunks)
        manifest[doc_id] = {
            "content_hash": current_hash,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(chunks),
        }

    if skipped:
        print(f"[SKIP] {len(skipped)} unchanged documents: {skipped}")

    _save_manifest(manifest)
    return collection


if __name__ == "__main__":
    get_or_create_index()
