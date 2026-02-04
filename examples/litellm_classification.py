"""Classification using LiteLLM with OpenWebUI endpoint."""
import os
from dotenv import load_dotenv
import litellm

load_dotenv()

# Configuration: loaded from .env file, or set directly here
_BASE_URL = os.getenv("OPENWEBUI_BASE_URL")  # e.g., "http://localhost:3000"
API_KEY = os.getenv("OPENWEBUI_API_KEY")  # e.g., "sk-..."
_MODEL = os.getenv("OPENWEBUI_MODEL")  # e.g., "llama4:latest"

BASE_URL = f"{_BASE_URL}/api"
MODEL = f"openai/{_MODEL}"  # LiteLLM requires "openai/" prefix for OpenAI-compatible APIs

CATEGORIES = ["positive", "negative", "neutral"]


def zero_shot_classify(text: str) -> str:
    """Zero-shot classification using LiteLLM."""
    response = litellm.completion(
        model=MODEL,
        api_base=BASE_URL,
        api_key=API_KEY,
        messages=[
            {
                "role": "system",
                "content": f"Classify text into: {', '.join(CATEGORIES)}. Respond with only the category.",
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip().lower()


def few_shot_classify(text: str) -> str:
    """Few-shot classification using LiteLLM."""
    response = litellm.completion(
        model=MODEL,
        api_base=BASE_URL,
        api_key=API_KEY,
        messages=[
            {
                "role": "system",
                "content": f"Classify text into: {', '.join(CATEGORIES)}. Respond with only the category.",
            },
            {"role": "user", "content": "I love this product, it's amazing!"},
            {"role": "assistant", "content": "positive"},
            {"role": "user", "content": "This is the worst experience ever."},
            {"role": "assistant", "content": "negative"},
            {"role": "user", "content": "The package arrived on Tuesday."},
            {"role": "assistant", "content": "neutral"},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip().lower()


if __name__ == "__main__":
    samples = [
        "This movie was fantastic, I really enjoyed it!",
        "Terrible service, would not recommend.",
        "The meeting is scheduled for 3pm.",
    ]

    print("=== Zero-shot Classification ===")
    for s in samples:
        print(f"{s[:40]}... -> {zero_shot_classify(s)}")

    print("\n=== Few-shot Classification ===")
    for s in samples:
        print(f"{s[:40]}... -> {few_shot_classify(s)}")
