import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import hashlib

CHROMA_DIR = "../vectorstore"
COLLECTION_NAME = "multimodal_rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def embed_and_store(chunks: List[Dict], source: str) -> int:
    if not chunks:
        return 0
    collection = get_collection()
    documents = []
    metadatas = []
    ids = []
    for i, chunk in enumerate(chunks):
        content = chunk["content"]
        if not content or len(content.strip()) < 10:
            continue
        type_label = chunk["type"].upper()
        full_text = f"[{type_label}] {content}"
        chunk_id = hashlib.md5(f"{source}_{i}_{content[:50]}".encode()).hexdigest()
        documents.append(full_text)
        metadatas.append({
            "source": chunk["source"],
            "page": chunk.get("page", 1),
            "type": chunk["type"],
        })
        ids.append(chunk_id)
    if not documents:
        return 0
    BATCH_SIZE = 50
    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end]
        )
    print(f"Stored {len(documents)} chunks. Total in DB: {collection.count()}")
    return len(documents)