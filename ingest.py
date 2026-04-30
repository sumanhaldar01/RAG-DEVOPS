import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"

import fitz
import ollama
import chromadb
import tiktoken
import time
from config import *

# ── Helpers ───────────────────────────────────────────────────────────────

def clear_line():
    """Overwrite current terminal line — used for progress spinner."""
    print("\r" + " " * 80 + "\r", end="", flush=True)

def progress(msg: str):
    """Print inline progress without newline."""
    print(f"\r  {msg}", end="", flush=True)

# ── Load PDFs ─────────────────────────────────────────────────────────────

def load_pdfs(docs_path: str) -> list[dict]:
    documents = []
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]

    if not pdf_files:
        return []

    print(f"\n📂 Found {len(pdf_files)} PDF(s):")
    for filename in pdf_files:
        print(f"   • {filename}")

    print()
    for filename in pdf_files:
        filepath = os.path.join(docs_path, filename)
        try:
            pdf = fitz.open(filepath)
            total_pages = len(pdf)
            loaded = 0

            for page_num in range(total_pages):
                # Show per-page progress
                progress(f"Reading {filename} — page {page_num+1}/{total_pages}")
                page = pdf[page_num]
                text = page.get_text("text").strip()
                if text and len(text) > 30:  # skip near-empty pages
                    documents.append({
                        "text":   text,
                        "source": filename,
                        "page":   page_num + 1
                    })
                    loaded += 1

            pdf.close()
            clear_line()
            print(f"  ✅ {filename} — {loaded}/{total_pages} pages with text")

        except Exception as e:
            clear_line()
            print(f"  ❌ Failed to read {filename}: {e}")

    return documents

# ── Chunk text ────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = enc.decode(tokens[start:end])
        if len(chunk.strip()) >= 20:  # skip tiny chunks
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

# ── Embed ──────────────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        raise RuntimeError(
            f"Embedding failed. Is Ollama running? "
            f"Try: ollama pull {EMBED_MODEL}\nError: {e}"
        )

# ── ChromaDB collection ───────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name="devops_docs",
        metadata={"hnsw:space": "cosine"}
    )

def get_existing_ids(collection) -> set:
    """Fetch all stored chunk IDs so we can skip already-ingested content."""
    try:
        result = collection.get(include=[])  # only fetch IDs, nothing else
        return set(result["ids"])
    except Exception:
        return set()

# ── Main ingest ────────────────────────────────────────────────────────────

def ingest():
    start_time = time.time()

    print("=" * 55)
    print("  🚀 RAG Ingest Pipeline")
    print("=" * 55)

    # Step 1: Check Ollama is alive
    print("\n[1/4] Checking Ollama connection...")
    try:
        ollama.embeddings(model=EMBED_MODEL, prompt="test")
        print(f"  ✅ Ollama OK — model: {EMBED_MODEL}")
    except Exception as e:
        print(f"  ❌ Cannot reach Ollama: {e}")
        print("     Fix: run 'ollama serve' in another terminal")
        print(f"     Then: ollama pull {EMBED_MODEL}")
        return

    # Step 2: Load PDFs
    print(f"\n[2/4] Loading PDFs from '{DOCS_PATH}'...")
    docs = load_pdfs(DOCS_PATH)

    if not docs:
        print(f"  ❌ No PDFs found in '{DOCS_PATH}'")
        print("     Drop your .pdf files there and re-run.")
        return

    print(f"\n  📄 Total pages with text: {len(docs)}")

    # Step 3: Connect to ChromaDB + check existing
    print("\n[3/4] Connecting to ChromaDB...")
    collection = get_collection()
    existing_ids = get_existing_ids(collection)
    existing_count = len(existing_ids)
    if existing_count:
        print(f"  ⚡ {existing_count} chunks already in DB — skipping duplicates")
    else:
        print("  📦 Empty DB — ingesting everything fresh")

    # Step 4: Chunk + embed + store
    print(f"\n[4/4] Chunking → Embedding → Storing...")
    print(f"      chunk_size={CHUNK_SIZE} tokens | overlap={CHUNK_OVERLAP} | top_k={TOP_K}")
    print()

    chunk_id      = existing_count   # continue ID numbering from where we left off
    skipped_pages = 0
    new_chunks    = 0
    failed_chunks = 0

    for doc_idx, doc in enumerate(docs):
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        filename_short = doc["source"][:35] + "…" if len(doc["source"]) > 35 else doc["source"]

        for c_idx, chunk in enumerate(chunks):
            chunk_key = f"chunk_{chunk_id}"

            # Skip if already ingested (idempotent re-runs)
            if chunk_key in existing_ids:
                skipped_pages += 1
                chunk_id += 1
                continue

            progress(
                f"[{doc_idx+1}/{len(docs)}] {filename_short} "
                f"p.{doc['page']} — chunk {c_idx+1}/{len(chunks)} "
                f"| new:{new_chunks} skip:{skipped_pages} fail:{failed_chunks}"
            )

            try:
                vector = embed_text(chunk)
                collection.add(
                    ids=[chunk_key],
                    embeddings=[vector],
                    documents=[chunk],
                    metadatas=[{
                        "source": doc["source"],
                        "page":   doc["page"]
                    }]
                )
                new_chunks += 1
            except Exception as e:
                failed_chunks += 1
                # Don't crash — log and continue
                clear_line()
                print(f"  ⚠️  chunk_{chunk_id} failed: {str(e)[:60]}")

            chunk_id += 1

    # Summary
    elapsed = round(time.time() - start_time, 1)
    clear_line()
    print("\n" + "=" * 55)
    print("  ✅ Ingest complete!")
    print("=" * 55)
    print(f"  New chunks stored : {new_chunks}")
    print(f"  Skipped (exist)   : {skipped_pages}")
    print(f"  Failed            : {failed_chunks}")
    print(f"  Total in DB       : {chunk_id}")
    print(f"  Time elapsed      : {elapsed}s")
    print("=" * 55)

    if failed_chunks > 0:
        print(f"\n  ⚠️  {failed_chunks} chunks failed — re-run to retry them")
    if new_chunks == 0 and skipped_pages > 0:
        print("\n  ℹ️  Nothing new to ingest. Add new PDFs to docs/ to add knowledge.")
    print()

if __name__ == "__main__":
    ingest()