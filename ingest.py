import os
import fitz          # PyMuPDF — reads PDF bytes into text
import ollama        # Ollama Python client
import chromadb      # Local vector database
import tiktoken      # Counts tokens (same tokenizer OpenAI uses)
from config import * # Pull all settings

# ── STEP A: Load + parse PDFs ─────────────────────────────────────────────

def load_pdfs(docs_path: str) -> list[dict]:
    """
    Walk docs/ folder, open each PDF, extract raw text per page.
    Returns list of: {text: str, source: str, page: int}
    """
    documents = []

    # os.listdir → list of filenames in folder
    for filename in os.listdir(docs_path):

        # Only process .pdf files, skip .gitkeep etc
        if not filename.endswith(".pdf"):
            continue

        # Build full file path: docs/ + filename.pdf
        filepath = os.path.join(docs_path, filename)

        # fitz.open() → opens PDF, gives page iterator
        pdf = fitz.open(filepath)

        # Iterate every page (0-indexed)
        for page_num in range(len(pdf)):
            page = pdf[page_num]

            # get_text("text") → extracts plain text from page
            # strip() → remove leading/trailing whitespace
            text = page.get_text("text").strip()

            # Skip empty pages (cover images, blank pages)
            if not text:
                continue

            # Store chunk metadata alongside text
            # metadata = WHERE this text came from (for citations!)
            documents.append({
                "text":   text,
                "source": filename,
                "page":   page_num + 1  # human-readable page number
            })

        page_count = len(pdf)   # save count BEFORE close
        pdf.close()
        print(f"✓ Loaded: {filename} ({page_count} pages)")

    return documents


# ── STEP B: Chunk text into smaller pieces ────────────────────────────────

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split long page text into overlapping token-based chunks.
    WHY chunk? LLM context window is limited. Smaller = precise retrieval.
    WHY overlap? Prevents cutting sentences mid-thought.
    """
    # tiktoken encodes text → token IDs (integers)
    # cl100k_base = encoding used by most modern LLMs
    enc = tiktoken.get_encoding("cl100k_base")

    # Encode full text to token list
    tokens = enc.encode(text)

    chunks = []
    start = 0  # sliding window start position

    while start < len(tokens):
        # Window end = start + chunk_size (or end of tokens)
        end = min(start + chunk_size, len(tokens))

        # Decode token slice back to readable string
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)

        # Move window forward but KEEP overlap tokens
        # overlap = chunks share context at boundaries
        start += chunk_size - chunk_overlap

    return chunks


# ── STEP C: Embed chunks via Ollama ───────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """
    Send text to Ollama nomic-embed-text → get float vector (768 dims).
    Vector = mathematical fingerprint of meaning.
    Similar text = similar vectors (close in space).
    """
    # ollama.embeddings() calls local Ollama API
    # model = nomic-embed-text (fast, CPU-friendly)
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )
    # response["embedding"] = list of 768 floats
    return response["embedding"]


# ── STEP D: Store in ChromaDB ─────────────────────────────────────────────

def get_collection():
    """
    Connect to ChromaDB and return (or create) our collection.
    PersistentClient → saves to disk, survives restarts.
    """
    # PersistentClient(path) → reads/writes to ./chroma_db folder
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # get_or_create_collection → idempotent (safe to call many times)
    # name = logical bucket for our DevOps doc vectors
    collection = client.get_or_create_collection(
        name="devops_docs",
        # cosine similarity = best for semantic text search
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ── MAIN: Wire everything together ────────────────────────────────────────

def ingest():
    print("🚀 Starting ingestion pipeline...")

    # Load all PDFs → list of page dicts
    docs = load_pdfs(DOCS_PATH)

    if not docs:
        print("❌ No PDFs found in docs/. Add PDFs and re-run.")
        return

    # Get ChromaDB collection
    collection = get_collection()

    chunk_id = 0  # unique ID counter for ChromaDB

    for doc in docs:
        # Split page text → chunks
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk in chunks:
            # Skip whitespace-only chunks (garbage)
            if len(chunk.strip()) < 20:
                continue

            # Embed chunk → vector
            vector = embed_text(chunk)

            # Add to ChromaDB:
            # ids       = unique string ID per chunk
            # embeddings = float vector (for similarity search)
            # documents  = raw text (returned at query time)
            # metadatas  = source info (for citations)
            collection.add(
                ids=[f"chunk_{chunk_id}"],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[{
                    "source": doc["source"],
                    "page":   doc["page"]
                }]
            )

            chunk_id += 1

        print(f"  ✓ {doc['source']} page {doc['page']} → {len(chunks)} chunks")

    print(f"\n✅ Done! {chunk_id} chunks stored in ChromaDB.")


# Run if called directly: python ingest.py
if __name__ == "__main__":
    ingest()