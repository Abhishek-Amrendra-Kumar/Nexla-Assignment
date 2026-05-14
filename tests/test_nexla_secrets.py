import importlib.util


def test_secrets_load_from_env():
    spec = importlib.util.spec_from_file_location("secrets", "src/nexla_mcp/secrets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.Secrets()


if __name__ == "__main__":
    test_secrets_load_from_env()
