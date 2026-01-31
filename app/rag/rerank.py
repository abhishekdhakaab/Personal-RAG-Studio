from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import CrossEncoder

class RerankSettings(BaseSettings):
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

_s = RerankSettings()
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(_s.rerank_model)
    return _model

def rerank(query: str, docs, top_k: int = 5):
    if not docs:
        return []
    model = _get_model()
    pairs = [(query, d.page_content) for d in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]
