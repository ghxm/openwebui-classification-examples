"""Classification using LangChain with OpenWebUI endpoint."""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

load_dotenv()

# Configuration: loaded from .env file, or set directly here
BASE_URL = os.getenv("OPENWEBUI_BASE_URL")  # e.g., "http://localhost:3000"
API_KEY = os.getenv("OPENWEBUI_API_KEY")  # e.g., "sk-..."
MODEL = os.getenv("OPENWEBUI_MODEL")  # e.g., "llama4:latest"

llm = ChatOpenAI(
    base_url=f"{BASE_URL}/api",
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
)

CATEGORIES = ["positive", "negative", "neutral"]


def zero_shot_classify(text: str) -> str:
    """Zero-shot classification using LangChain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Classify text into: {', '.join(CATEGORIES)}. Respond with only the category."),
        ("user", "{text}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content.strip().lower()


def few_shot_classify(text: str) -> str:
    """Few-shot classification using LangChain."""
    examples = [
        {"input": "I love this product, it's amazing!", "output": "positive"},
        {"input": "This is the worst experience ever.", "output": "negative"},
        {"input": "The package arrived on Tuesday.", "output": "neutral"},
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}"),
        ("assistant", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Classify text into: {', '.join(CATEGORIES)}. Respond with only the category."),
        few_shot_prompt,
        ("user", "{text}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content.strip().lower()


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
