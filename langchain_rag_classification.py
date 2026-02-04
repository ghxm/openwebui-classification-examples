"""RAG-based classification using LangChain + ChromaDB with OpenWebUI.

Uses local embeddings (sentence-transformers) since no server-side embedding model.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

load_dotenv()

BASE_URL = f"{os.getenv('OPENWEBUI_BASE_URL')}/api"
API_KEY = os.getenv("OPENWEBUI_API_KEY")
MODEL = os.getenv("OPENWEBUI_MODEL")

llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL, temperature=0)

# Local embeddings (no API call needed)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

CATEGORIES = ["bug_report", "feature_request", "question", "documentation"]

# Labeled examples for retrieval
LABELED_EXAMPLES = [
    ("The app crashes when I click the submit button", "bug_report"),
    ("Error 500 appears on login page", "bug_report"),
    ("Would be great to have dark mode", "feature_request"),
    ("Can you add export to PDF functionality?", "feature_request"),
    ("How do I reset my password?", "question"),
    ("What's the difference between plan A and B?", "question"),
    ("The API reference is missing parameters", "documentation"),
    ("Typo in the installation guide", "documentation"),
]


def build_vectorstore():
    """Build a vectorstore from labeled examples."""
    texts = [f"[{label}] {text}" for text, label in LABELED_EXAMPLES]
    metadatas = [{"label": label, "text": text} for text, label in LABELED_EXAMPLES]
    return Chroma.from_texts(texts, embeddings, metadatas=metadatas)


def rag_classify(text: str, vectorstore: Chroma, k: int = 3) -> str:
    """Classify using RAG - retrieve similar examples and use them as context."""
    docs = vectorstore.similarity_search(text, k=k)
    context = "\n".join([f"- {d.metadata['text']} -> {d.metadata['label']}" for d in docs])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"Classify the user's text into one of: {', '.join(CATEGORIES)}.\n"
            f"Use these similar examples as reference:\n{context}\n"
            "Respond with only the category name.",
        ),
        ("user", "{text}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content.strip().lower()


if __name__ == "__main__":
    print("Building vectorstore...")
    vs = build_vectorstore()

    samples = [
        "The application freezes when uploading large files",
        "It would be nice to have keyboard shortcuts",
        "How do I connect to the database?",
        "The README has outdated information",
    ]

    print("\n=== RAG Classification ===")
    for s in samples:
        print(f"{s[:45]}... -> {rag_classify(s, vs)}")
