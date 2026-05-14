import os
from dotenv import load_dotenv

from nexla_mcp.config import (
    ENV_HF_TOKEN,
    ENV_LITELLM_API_KEY,
    ENV_LITELLM_BASE_URL,
    ENV_LITELLM_MODEL,
)


class Secrets:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            load_dotenv()
            missing = [
                v
                for v in [
                    ENV_HF_TOKEN,
                    ENV_LITELLM_API_KEY,
                    ENV_LITELLM_BASE_URL,
                    ENV_LITELLM_MODEL,
                ]
                if not os.environ.get(v) or not os.environ.get(v).strip()
            ]
            if missing:
                raise ValueError(f"Missing env vars: {', '.join(missing)}")
        return cls._instance

    def get_hf_token(self) -> str:
        return os.environ[ENV_HF_TOKEN]

    def get_api_key(self) -> str:
        return os.environ[ENV_LITELLM_API_KEY]

    def get_base_url(self) -> str:
        return os.environ[ENV_LITELLM_BASE_URL]

    def get_model(self) -> str:
        return os.environ[ENV_LITELLM_MODEL]
