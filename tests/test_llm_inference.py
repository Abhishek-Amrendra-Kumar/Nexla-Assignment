from dotenv import load_dotenv
from nexla_mcp.llm.inference import llm_infer

load_dotenv()


def test_llm_infer_not_empty():
    response = llm_infer(messages=[{"role": "user", "content": "Say hello"}])
    assert response and response.strip(), "Empty response from LLM"
    print(f"\nResponse: {response}\n")


if __name__ == "__main__":
    test_llm_infer_not_empty()
