from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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

# Simple in‑memory auth check
def check_credentials(username: str, password: str) -> bool:
    """Validate provided credentials.

    This function performs a very basic in‑memory check against a hard‑coded
    username and password (both ``"1234"``). It returns ``True`` when the
    credentials match and ``False`` otherwise.

    Args:
        username: The username supplied by the client.
        password: The password supplied by the client.

    Returns:
        bool: ``True`` if the credentials are correct, ``False`` otherwise.
    """
    return username == "1234" and password == "1234"

# Serve static files (frontend assets)
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    """Return the login page.

    The login page is a simple HTML form that posts credentials to the
    ``/login`` endpoint. After a successful login the front‑end JavaScript will
    load the main application UI.
    """
    login_path = Path("../frontend/login.html")
    if login_path.is_file():
        return FileResponse(login_path)
    return HTMLResponse("<h1>Login page not found</h1>", status_code=404)

@app.get("/health")
def health():
    """Health‑check endpoint.

    Returns a simple JSON payload indicating that the service is running.
    This can be used by monitoring tools or load balancers to verify that the
    API process is alive.
    """
    return {"status": "running"}

@app.post("/login")
def login(payload: dict = Body(...)):
    """Authenticate a user.

    Expects a JSON body containing ``username`` and ``password`` fields. The
    credentials are validated using :func:`check_credentials`. If the check
    fails, a ``401 Unauthorized`` error is raised.

    Args:
        payload: The request body parsed as a dictionary.

    Returns:
        dict: A success response with a status message when authentication
        succeeds.
    """
    user = payload.get("username", "")
    pwd = payload.get("password", "")
    if not check_credentials(user, pwd):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"status": "ok", "message": "Logged in"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file uploads and process them for RAG.

    The endpoint accepts a file, validates its extension against a whitelist,
    stores it in the ``uploads`` directory, extracts its content, and embeds the
    resulting chunks for later retrieval.

    Args:
        file: The uploaded file provided by the client.

    Returns:
        dict: Information about the processed file, including the number of
        chunks stored.
    """
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
    """Answer a question using the stored document embeddings.

    The request body must contain a ``question`` field. The function forwards the
    question to the retrieval‑and‑answer pipeline and returns the result.

    Args:
        payload: Dictionary containing the ``question`` key.

    Returns:
        dict: The answer payload produced by :func:`retrieve_and_answer`.
    """
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
    """List all uploaded documents.

    Scans the upload directory and returns a list of filenames for files that
    exist on disk.

    Returns:
        dict: A dictionary with a ``documents`` key containing the list of file
        names.
    """
    files = list(UPLOAD_DIR.glob("*"))
    return {"documents": [f.name for f in files if f.is_file()]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
