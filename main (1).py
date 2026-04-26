"""
AI-drevet nyhetsoversikt
FastAPI + LangChain + PostgreSQL + Docker
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import feedparser
import psycopg2
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA

# --- Config ---
DB_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@db:5432/newsdb")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

NEWS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
]

# Global vector store (in-memory for simplicity)
vectorstore = None


# --- Database ---
def get_db():
    return psycopg2.connect(DB_URL)


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT,
            url TEXT UNIQUE,
            source TEXT,
            fetched_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


def save_article(title, summary, url, source):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO articles (title, summary, url, source) VALUES (%s, %s, %s, %s) ON CONFLICT (url) DO NOTHING",
        (title, summary, url, source)
    )
    conn.commit()
    cur.close()
    conn.close()


def get_all_articles():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT title, summary, url, source FROM articles ORDER BY fetched_at DESC LIMIT 100")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"title": r[0], "summary": r[1], "url": r[2], "source": r[3]} for r in rows]


# --- News fetching ---
def fetch_and_store_articles():
    articles = []
    for feed_url in NEWS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:10]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            url = entry.get("link", "")
            source = feed.feed.get("title", feed_url)
            save_article(title, summary, url, source)
            articles.append({"title": title, "summary": summary, "url": url, "source": source})
    return articles


# --- RAG pipeline ---
def build_vectorstore(articles):
    docs = [
        Document(
            page_content=f"{a['title']}. {a['summary']}",
            metadata={"url": a["url"], "source": a["source"]}
        )
        for a in articles if a["summary"]
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)


def ask_question(question: str) -> dict:
    global vectorstore
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vector store not ready. Call /refresh first.")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )
    result = chain.invoke({"query": question})
    sources = [doc.metadata.get("url", "") for doc in result["source_documents"]]
    return {"answer": result["result"], "sources": list(set(sources))}


# --- App lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore
    init_db()
    articles = fetch_and_store_articles()
    if articles:
        vectorstore = build_vectorstore(articles)
    yield


app = FastAPI(title="AI News RAG", lifespan=lifespan)


# --- Models ---
class QuestionRequest(BaseModel):
    question: str


# --- Endpoints ---
@app.get("/")
def root():
    return {"status": "running", "endpoints": ["/articles", "/ask", "/refresh"]}


@app.get("/articles")
def list_articles():
    return get_all_articles()


@app.post("/ask")
def ask(req: QuestionRequest):
    return ask_question(req.question)


@app.post("/refresh")
def refresh():
    global vectorstore
    articles = fetch_and_store_articles()
    vectorstore = build_vectorstore(articles)
    return {"message": f"Fetched and indexed {len(articles)} articles"}
