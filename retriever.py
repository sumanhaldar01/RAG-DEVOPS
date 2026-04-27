import ollama
import chromadb
from config import *

def get_collection():
    """Same collection getter — reused across files."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name="devops_docs",
        metadata={"hnsw:space": "cosine"}
    )

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Convert user question → vector → find nearest chunks in ChromaDB.
    Returns top_k most semantically similar chunks + their metadata.
    """
    # Embed the USER QUERY (same model as doc embeddings — critical!)
    # If embedding models differ, vectors are incompatible (garbage results)
    query_vector = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=query
    )["embedding"]

    # Query ChromaDB: find top_k vectors closest to query_vector
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector],  # our query vector
        n_results=top_k,                 # how many chunks to return
        include=["documents", "metadatas", "distances"]  # what to include
    )

    # Unpack ChromaDB response (nested lists — [0] = first query)
    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],   # chunk text
        results["metadatas"][0],   # source + page
        results["distances"][0]    # cosine distance (lower = more similar)
    ):
        chunks.append({
            "text":     text,
            "source":   meta["source"],
            "page":     meta["page"],
            "distance": round(dist, 4)  # debugging: how relevant is chunk?
        })

    return chunks