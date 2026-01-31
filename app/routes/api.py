import os
from fastapi import APIRouter, UploadFile, File, Form
from app.rag.pipeline import load_file_to_docs, index_docs, retrieve, answer_with_citations

router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs("app/data", exist_ok=True)
    path = os.path.join("app/data", file.filename)
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)

    docs = load_file_to_docs(path)
    n = index_docs(docs)
    return {"ok": True, "chunks_indexed": n, "file": file.filename}

@router.post("/chat")
async def chat(
    question: str = Form(...),
    mode: str = Form("vector"),
    top_k: int = Form(5),
):
    docs = retrieve(question, mode=mode, k=top_k)
    return answer_with_citations(question, docs)
