from __future__ import annotations

import os, uuid
from typing import Literal

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from app.rag.store import get_vectorstore
from app.rag.rerank import rerank

Splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

def load_file_to_docs(path: str) -> list[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()
    elif ext in [".md", ".markdown"]:
        loader = UnstructuredMarkdownLoader(path)
        docs = loader.load()
    else:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()

    #into chunks
    chunks = Splitter.split_documents(docs)

    for d in chunks:
        d.metadata = d.metadata or {}
        d.metadata["chunk_id"] = d.metadata.get("chunk_id") or str(uuid.uuid4())
        d.metadata["source"] = d.metadata.get("source") or path
    return chunks

def index_docs(docs: list[Document]) -> int:
    vs = get_vectorstore()
    vs.add_documents(docs)
    return len(docs)

def retrieve(
    question: str,
    mode: Literal["vector", "hybrid", "rerank"] = "vector",
    k: int = 5
) -> list[Document]:
    vs = get_vectorstore()

    # 1) vector retriever
    vector_retriever = vs.as_retriever(search_kwargs={"k": max(k, 8)})

    if mode == "vector":
        return vector_retriever.get_relevant_documents(question)[:k]

    # 2) hybrid 
    if mode == "hybrid":
        seed_docs = vector_retriever.get_relevant_documents(question)
        bm25 = BM25Retriever.from_documents(seed_docs)
        bm25.k = max(k, 8)

        ensemble = EnsembleRetriever(
            retrievers=[bm25, vector_retriever],
            weights=[0.5, 0.5],
        )
        return ensemble.get_relevant_documents(question)[:k]

    # 3) rerank 
    if mode == "rerank":
        candidates = vector_retriever.get_relevant_documents(question)
        return rerank(question, candidates, top_k=k)

    return vector_retriever.get_relevant_documents(question)[:k]

def answer_with_citations(question: str, docs: list[Document]) -> dict:
    if not docs:
        return {
            "answer": "No supporting information found in the indexed documents.",
            "citations": []
        }

    # small + honest baseline: show evidence + short synthesis
    snippets = []
    citations = []
    for d in docs:
        txt = (d.page_content or "").strip()
        snippets.append(f"- {txt[:300]}...")
        citations.append({
            "chunk_id": d.metadata.get("chunk_id"),
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
        })

    answer = (
        f"Q: {question}\n\n"
        "Most relevant evidence:\n"
        + "\n".join(snippets)
        + "\n\nTip: In a real app, you’d plug an LLM here to synthesize an answer grounded in these citations."
    )
    return {"answer": answer, "citations": citations}
