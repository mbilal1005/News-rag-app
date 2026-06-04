# News-rag-app

Ask questions about today's news using a RAG (Retrieval-Augmented Generation) pipeline. Fetches articles from RSS feeds, stores them in PostgreSQL, indexes them with FAISS, and exposes a FastAPI REST API.

## How it works

```
RSS feeds (BBC, NYT)
       ↓
Fetch articles → store in PostgreSQL
       ↓
Embed with OpenAI → index in FAISS
       ↓
POST /ask → retrieve relevant chunks → GPT-3.5-turbo answers
```

## Stack

- **FastAPI** — REST API
- **LangChain + OpenAI** — embeddings and LLM
- **FAISS** — in-memory vector store for semantic search
- **PostgreSQL** — persistent article storage
- **Docker + docker-compose** — containerised deployment

## Quick Start

```bash
git clone https://github.com/mbilal1005/News-rag-app
cd News-rag-app

export OPENAI_API_KEY="sk-..."
docker compose up --build
```

API runs at `http://localhost:8000`.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/articles` | List all fetched articles |
| POST | `/ask` | Ask a question about the news |
| POST | `/refresh` | Re-fetch and re-index latest articles |

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
