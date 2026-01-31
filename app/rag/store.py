from pydantic_settings import BaseSettings
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    qdrant_url: str
    collection_name: str = "rag_docs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")




settings  = Settings()

def get_embeddings():
    return HuggingFaceEmbeddings(model_name = settings.embedding_model)

def get_qdrant_client():
    return QdrantClient(url=settings.qdrant_url)

from qdrant_client.http import models as qm

def ensure_collection(client: QdrantClient, collection_name: str, dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(
            size=dim,
            distance=qm.Distance.COSINE,
        ),
    )

def get_vectorstore():
    client = get_qdrant_client()
    embeddings = get_embeddings()

    # Figure out embedding dimension by embedding a tiny string once
    dim = len(embeddings.embed_query("dimension check"))
    ensure_collection(client, settings.collection_name, dim)

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )
