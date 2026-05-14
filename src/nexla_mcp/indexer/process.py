from datetime import datetime, timezone
from pathlib import Path

from nexla_mcp.indexer.hf import encode_texts
from nexla_mcp.indexer.manifest import (
    _compute_file_hash,
    _load_manifest,
    _save_manifest,
    get_chroma_client,
)


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
