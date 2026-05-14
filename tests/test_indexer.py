import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nexla_mcp.indexer import encode_texts


def test_embed_simple_sentence():
    texts = ["This is a simple test sentence."]
    start = time.perf_counter()
    vectors = encode_texts(texts)
    elapsed = time.perf_counter() - start
    assert vectors, "Expected non-empty list of vectors"
    assert len(vectors[0]) > 0, "Expected non-empty vector"
    print(f"Vector dimension: {len(vectors[0])}")
    print(f"Inference time: {elapsed:.4f}s")


if __name__ == "__main__":
    test_embed_simple_sentence()
