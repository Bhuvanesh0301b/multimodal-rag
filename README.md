It seems the existing README already covers the steps to run the project. However, to improve clarity, the installation command could be explicitly mentioned. Here is a proposed update for the '## 🚀 How to Run' section: 
# 🧠 Multimodal RAG — Chat with PDFs, Images & Tables

Ask questions about your documents using AI. Understands text, charts, tables, and images — all together.

## ✨ Features
- Upload PDF, PNG, JPG, DOCX, PPTX
- AI reads text, tables, and describes charts/images
- Ask natural language questions and get accurate answers
- 100% Free — no paid APIs

## 🆓 Free Stack
| Component | Tool |
|-----------|------|
| LLM (Q&A) | Groq → Llama 3.3 70B |
| Vision (images/charts) | Groq → Llama 4 Scout |
| Vector Database | ChromaDB (local) |
| Embeddings | Sentence Transformers (CPU) |
| PDF Parsing | PyMuPDF |
| Backend | FastAPI |

## 🚀 How to Run

### 1. Get free Groq API key
Sign up at https://console.groq.com (no credit card needed)

### 2. Install packages
Run 'pip install -r requirements.txt' or manually install the required packages: FastAPI, PyMuPDF, Sentence Transformers, and any other dependencies.

### 3. Run the application
Execute 'uvicorn backend.main:app --host 0.0.0.0 --port 8000' to start the FastAPI server. Then, open 'frontend/index.html' in your web browser to interact with the application.