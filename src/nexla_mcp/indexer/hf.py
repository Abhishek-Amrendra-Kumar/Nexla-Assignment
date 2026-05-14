from huggingface_hub import InferenceClient

from nexla_mcp.config import MODEL_NAME, USE_LOCAL_INFERENCE
from nexla_mcp.env_secrets import Secrets

_client = None
_local_model = None


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        token = Secrets().get_hf_token()
        _client = InferenceClient(model=MODEL_NAME, token=token)
    return _client


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(MODEL_NAME)
    return _local_model


def encode_texts(texts: list[str], prompt_name: str | None = None) -> list[list[float]]:
    if USE_LOCAL_INFERENCE:
        model = _get_local_model()
        kwargs = {"show_progress_bar": True}
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        embeddings = model.encode(texts, **kwargs)
        return embeddings.tolist()

    client = _get_client()
    prefix = {"query": "query: ", "document": "document: "}.get(prompt_name or "", "")
    res = []
    for i, t in enumerate(texts):
        res.append(client.feature_extraction(prefix + t))
        print(f"{i + 1}/{len(texts)}")
    return res
