import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings

PERSIST_DIR = Path("chroma_db")
MANIFEST_PATH = PERSIST_DIR / "index_manifest.json"


def get_chroma_client(persist_dir: Path = PERSIST_DIR):
    return chromadb.PersistentClient(
        path=str(persist_dir), settings=Settings(anonymized_telemetry=False)
    )


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
