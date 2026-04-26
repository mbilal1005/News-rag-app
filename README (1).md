# AI News RAG

Ask questions about today's news using RAG (Retrieval-Augmented Generation).

Fetches articles from RSS feeds, stores them in PostgreSQL, indexes them with FAISS, and lets you query them via a FastAPI REST API.

## Stack

- **FastAPI** – REST API
- **LangChain + OpenAI** – LLM and embeddings
- **FAISS** – vector store for semantic search
- **PostgreSQL** – article storage
- **Docker** – containerised deployment

## Quick start

```bash
export OPENAI_API_KEY="sk-..."
docker compose up --build
```

API is available at `http://localhost:8000`

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/articles` | List all fetched articles |
| POST | `/ask` | Ask a question about the news |
| POST | `/refresh` | Re-fetch and re-index articles |

## Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is happening in Ukraine?"}'
```

```json
{
  "answer": "According to recent articles...",
  "sources": ["https://bbc.co.uk/..."]
}
```
