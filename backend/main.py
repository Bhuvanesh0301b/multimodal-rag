from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import shutil
import uuid

from extractor import extract_content
from embedder import embed_and_store
from retriever import retrieve_and_answer

app = FastAPI(title="Multimodal RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("../uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed = [".pdf", ".png", ".jpg", ".jpeg", ".docx", ".pptx"]
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file: {suffix}")
    file_id = str(uuid.uuid4())[:8]
    save_name = f"{file_id}_{file.filename}"
    save_path = UPLOAD_DIR / save_name
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        chunks = extract_content(str(save_path), file.filename)
        count = embed_and_store(chunks, file.filename)
        return {"status": "success", "file": file.filename, "chunks_stored": count}
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")
    try:
        result = retrieve_and_answer(question)
        return result
    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")

@app.get("/documents")
def list_documents():
    files = list(UPLOAD_DIR.glob("*"))
    return {"documents": [f.name for f in files if f.is_file()]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)