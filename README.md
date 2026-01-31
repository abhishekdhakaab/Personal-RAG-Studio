# RAG Knowledge Copilot (fun project)

This is a small **RAG (Retrieval Augmented Generation) system** I built to understand how modern document-based AI systems actually work under the hood.

The main idea was simple:

> I wanted a system that can answer questions from **private documents** that I don’t want to upload to publicly available AI tools.

So instead of sending data to ChatGPT or similar products, this project:
- indexes documents locally
- stores embeddings in a vector database
- retrieves relevant chunks
- and answers questions based only on those documents

This is **not a production product**, just a learning project that helped me understand RAG properly.

---

## What this project does

- Upload documents (PDF / TXT / Markdown)
- Split them into chunks
- Generate embeddings
- Store embeddings in a vector database (Qdrant)
- Retrieve relevant chunks when a question is asked
- Support multiple retrieval strategies
- Show citations (which chunk / document the answer came from)

The focus was **learning RAG**, not building a fancy UI.

---

## Features

- Document ingestion (PDF, TXT, MD)
- Automatic chunking using LangChain
- Embeddings using sentence-transformers
- Vector database using Qdrant (Docker)
- Multiple retrieval modes:
  - `vector` – basic vector similarity search
  - `hybrid` – BM25 + vector search
  - `rerank` – vector search + cross-encoder reranking
- Source citations for retrieved content
- FastAPI backend
- Very simple HTML frontend
- Docker support for the vector DB

---

## Why I built this

I wanted to understand:
- how real RAG systems work end-to-end
- how vector databases are used in practice
- why retrieval quality matters more than just “calling an LLM”
- how different RAG strategies compare

Also, I didn’t want to upload personal or private documents to public AI tools, so this project keeps everything local.

---

## Tech stack

- **Backend**: FastAPI
- **RAG framework**: LangChain
- **Vector DB**: Qdrant
- **Embeddings**: sentence-transformers
- **Reranking**: cross-encoder (ms-marco)
- **Frontend**: plain HTML + CSS
- **Infra**: Docker

---

## Screenshots

![Document upload](screenshot.png)

```text
[ Screenshot of document upload page ]
[ Screenshot of question answering with citations ]


How to run this locally
1. Clone the repo
git clone <your-repo-url>
cd rag-knowledge-copilot

2. Start Qdrant (vector database)

Make sure Docker is running.

docker compose up -d


You can verify Qdrant is running:

curl http://localhost:6333/healthz

3. Create virtual environment & install dependencies

I recommend Python 3.11.

python3.11 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

4. Environment variables

Create a .env file:

QDRANT_URL=http://localhost:6333
COLLECTION_NAME=rag_docs
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

5. Run the server
./.venv/bin/uvicorn app.main:app --reload


Open:

http://127.0.0.1:8000

How to use

Upload a document (PDF / TXT / MD)

Ask a question in the chat box

Choose a retrieval mode:

vector

hybrid

rerank

View the answer + citations

Notes / limitations

This project focuses on retrieval, not fancy LLM prompting

Answers are intentionally conservative and grounded in retrieved text

No auth, no multi-user support

No heavy UI work (intentionally kept simple)

What I learned from this

RAG quality depends way more on retrieval than model choice

Hybrid search usually beats pure vector search

Reranking improves relevance a lot, but adds cost/latency

Vector DB schema + configuration matters more than expected

Most real-world RAG systems are just careful engineering, not magic

Future ideas (maybe)

Add evaluation metrics (RAGAS / recall@k)

Add local LLM for full offline mode

Better chunking strategies

Access control per document

Final note

This is a learning project I built for fun and understanding RAG systems better.
If you’re reading this and trying to learn RAG yourself, I highly recommend building something similar instead of only reading theory.


---

If you want, I can:
- make a **shorter README** (even more raw)
- tweak tone to sound more “student” or more “engineer”
- add a **project architecture diagram (ASCII or image prompt)**

Just tell me 👍