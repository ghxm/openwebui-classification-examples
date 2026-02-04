"""List available models from OpenWebUI."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration: loaded from .env file, or set directly here
BASE_URL = os.getenv("OPENWEBUI_BASE_URL")  # e.g., "http://localhost:3000"
API_KEY = os.getenv("OPENWEBUI_API_KEY")  # e.g., "sk-..."


def list_models():
    """Fetch and display available models."""
    response = requests.get(
        f"{BASE_URL}/api/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    response.raise_for_status()
    data = response.json()

    print("Available models:\n")
    for model in data.get("data", []):
        model_id = model.get("id", "unknown")
        owned_by = model.get("owned_by", "")
        print(f"  - {model_id}" + (f"  ({owned_by})" if owned_by else ""))

    return [m.get("id") for m in data.get("data", [])]


if __name__ == "__main__":
    list_models()
