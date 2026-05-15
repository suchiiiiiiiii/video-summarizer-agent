"""
FastAPI backend for AI Video Summarizer (SIGNA)
Deployment: Fly.io
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile, os, shutil, sys, asyncio, uuid
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from main import run_pipeline
from core.rag_engine import ask_question

sessions: dict[str, dict] = {}

app = FastAPI(title="SIGNA — AI Video Summarizer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/api/process")
async def process(
    url: Optional[str] = Form(None),
    language: str = Form("english"),
    file: Optional[UploadFile] = File(None),
):
    if not url and not file:
        raise HTTPException(400, "Provide either a YouTube URL or an audio file.")

    source = url
    tmp_path = None

    if file:
        suffix = os.path.splitext(file.filename)[-1] or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        source = tmp_path

    try:
        result = await asyncio.to_thread(run_pipeline, source, language)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    session_id = str(uuid.uuid4())
    sessions[session_id] = {"rag_chain": result.pop("rag_chain")}

    return {
        "session_id":    session_id,
        "title":         result["title"],
        "transcript":    result["transcript"],
        "summary":       result["summary"],
        "action_items":  result["action_items"],
        "key_decisions": result["key_decisions"],
        "open_questions":result["open_questions"],
    }

class AskRequest(BaseModel):
    session_id: str
    question: str

@app.post("/api/ask")
async def ask(req: AskRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Please process a video first.")
    try:
        answer = await asyncio.to_thread(ask_question, session["rag_chain"], req.question)
    except Exception as e:
        raise HTTPException(500, f"RAG error: {e}")
    return {"answer": answer}

@app.get("/api/health")
async def health():
    return {"status": "ok", "sessions": len(sessions)}