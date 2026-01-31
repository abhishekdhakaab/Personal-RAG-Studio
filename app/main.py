from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.ui import router as ui_router
from app.routes.api import router as api_router

app = FastAPI(title="RAG Studio Lite")

app.include_router(ui_router)
app.include_router(api_router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
