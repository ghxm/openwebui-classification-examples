"""Text classification using plain requests (no SDK).

Shows the raw HTTP API approach without any abstraction layers.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration: loaded from .env file, or set directly here
BASE_URL = os.getenv("OPENWEBUI_BASE_URL")  # e.g., "http://localhost:3000"
API_KEY = os.getenv("OPENWEBUI_API_KEY")  # e.g., "sk-..."
MODEL = os.getenv("OPENWEBUI_MODEL")  # e.g., "llama4:latest"

CATEGORIES = ["positive", "negative", "neutral"]
SAMPLES = [
    "This movie was fantastic, I really enjoyed it!",
    "Terrible service, would not recommend.",
    "The meeting is scheduled for 3pm.",
]


def classify(text: str, messages: list[dict]) -> str:
    """Send a chat completion request and return the response."""
    response = requests.post(
        f"{BASE_URL}/api/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": messages, "temperature": 0},
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip().lower()


def zero_shot(text: str) -> str:
    """Zero-shot classification: just the system prompt, no examples."""
    messages = [
        {
            "role": "system",
            "content": f"Classify the following text into one of these categories: {', '.join(CATEGORIES)}. Respond with only the category name.",
        },
        {"role": "user", "content": text},
    ]
    return classify(text, messages)


def few_shot(text: str) -> str:
    """Few-shot classification: include examples in the conversation."""
    messages = [
        {
            "role": "system",
            "content": f"Classify the following text into one of these categories: {', '.join(CATEGORIES)}. Respond with only the category name.",
        },
        {"role": "user", "content": "I love this product, best purchase ever!"},
        {"role": "assistant", "content": "positive"},
        {"role": "user", "content": "This is the worst experience I've had."},
        {"role": "assistant", "content": "negative"},
        {"role": "user", "content": "The package arrived on Tuesday."},
        {"role": "assistant", "content": "neutral"},
        {"role": "user", "content": text},
    ]
    return classify(text, messages)


if __name__ == "__main__":
    print("=== Zero-shot Classification ===")
    for sample in SAMPLES:
        print(f"{sample[:50]}... -> {zero_shot(sample)}")

    print("\n=== Few-shot Classification ===")
    for sample in SAMPLES:
        print(f"{sample[:50]}... -> {few_shot(sample)}")
