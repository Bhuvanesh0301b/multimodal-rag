import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from typing import Dict

client = Groq()

CHROMA_DIR = "../vectorstore"
COLLECTION_NAME = "multimodal_rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def retrieve_and_answer(question: str, top_k: int = 6) -> Dict:
    collection = get_collection()
    if collection.count() == 0:
        return {"answer": "No documents uploaded yet. Please upload a file first.", "sources": [], "chunks_used": 0}
    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    if not docs:
        return {"answer": "No relevant content found.", "sources": [], "chunks_used": 0}
    context_parts = []
    sources_used = set()
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        relevance = round((1 - dist) * 100, 1)
        chunk_type = meta.get("type", "text").upper()
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        context_parts.append(f"--- Chunk {i+1} [{chunk_type}] from '{source}' page {page} (relevance: {relevance}%) ---\n{doc}")
        sources_used.add(f"{source} (page {page})")
    context = "\n\n".join(context_parts)
    system_prompt = """You are an intelligent document assistant.
You are given retrieved chunks from documents that may include TEXT, IMAGE descriptions, and TABLE data.
Answer the user's question using ONLY the provided context.
Be specific, cite numbers and details. If the answer is not in the context, say so clearly."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
        ]
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": list(sources_used),
        "chunks_used": len(docs),
        "question": question
    }