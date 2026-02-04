# OpenWebUI Classification Examples

Examples showing how to use OpenWebUI's API for text classification.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OpenWebUI URL, API key, and model name
```

Get your API key from: **Settings > Account** in OpenWebUI.

## List Available Models

```bash
python list_models.py
```

Example output:
```
Available models:

  - gemma3:27b  (ollama)
  - llama4:latest  (openai)
  - llama4:latest  (ollama)
```

Or query programmatically:
```python
import requests

response = requests.get(
    f"{BASE_URL}/api/models",
    headers={"Authorization": f"Bearer {API_KEY}"}
)
models = [m["id"] for m in response.json()["data"]]
```

Then set your chosen model in `.env`:
```
OPENWEBUI_MODEL=llama4:latest
```

## Examples

| File | Package | Techniques |
|------|---------|------------|
| `openai_classification.py` | openai | Zero-shot, Few-shot |
| `langchain_classification.py` | langchain | Zero-shot, Few-shot |
| `litellm_classification.py` | litellm | Zero-shot, Few-shot |
| `langchain_rag_classification.py` | langchain + chromadb | RAG (local embeddings) |

## Run

```bash
python openai_classification.py
python langchain_classification.py
python litellm_classification.py
python langchain_rag_classification.py
```

## API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /api/chat/completions` | Chat completions (OpenAI-compatible) |
| `GET /api/models` | List available models |
