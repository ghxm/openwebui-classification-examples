"""Classification using OpenAI SDK with OpenWebUI endpoint."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=f"{os.getenv('OPENWEBUI_BASE_URL')}/api",
    api_key=os.getenv("OPENWEBUI_API_KEY"),
)
MODEL = os.getenv("OPENWEBUI_MODEL")

CATEGORIES = ["positive", "negative", "neutral"]


def zero_shot_classify(text: str) -> str:
    """Zero-shot classification."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": f"Classify the following text into one of these categories: {', '.join(CATEGORIES)}. "
                "Respond with only the category name.",
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip().lower()


def few_shot_classify(text: str) -> str:
    """Few-shot classification with examples."""
    response = client.chat.completions.create(
        model=MODEL,
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
