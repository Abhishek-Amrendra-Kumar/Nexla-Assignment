from huggingface_hub import InferenceClient

from nexla_mcp.env_secrets import Secrets

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

_client = None


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        token = Secrets().get_hf_token()
        _client = InferenceClient(model=MODEL_NAME, token=token)
    return _client


def encode_texts(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    res = []
    for i, t in enumerate(texts):
        res.append(client.feature_extraction(t))
        print(f"{i + 1}/{len(texts)}")
    return res
