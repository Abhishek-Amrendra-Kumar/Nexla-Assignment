import os
from dotenv import load_dotenv


class Secrets:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            load_dotenv()
            missing = [
                v
                for v in [
                    "HF_TOKEN",
                    "LITELLM_API_KEY",
                    "LITELLM_BASE_URL",
                    "LITELLM_MODEL",
                ]
                if not os.environ.get(v) or not os.environ.get(v).strip()
            ]
            if missing:
                raise ValueError(f"Missing env vars: {', '.join(missing)}")
        return cls._instance

    def get_hf_token(self) -> str:
        return os.environ["HF_TOKEN"]

    def get_api_key(self) -> str:
        return os.environ["LITELLM_API_KEY"]

    def get_base_url(self) -> str:
        return os.environ["LITELLM_BASE_URL"]

    def get_model(self) -> str:
        return os.environ["LITELLM_MODEL"]
