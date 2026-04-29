# DevOps RAG Assistant

A simple local RAG (Retrieval-Augmented Generation) system built to learn how chunking and vector search work under the hood — powered entirely by local AI, no cloud required.

> Built with the help of [Claude](https://claude.ai) as a hands-on learning project.

## What I Learned

**Chunking** — Documents can't fit into an LLM's context window whole. So they're split into overlapping token-based chunks, preserving context at boundaries.

**Embeddings** — Each chunk is converted into a vector (a list of numbers) that represents its *meaning*. Similar chunks end up mathematically close to each other.

**Retrieval** — When you ask a question, it's embedded the same way. ChromaDB finds the closest matching chunks using cosine similarity.

**Generation** — Those chunks are handed to the LLM as context. It answers from *your documents*, not from hallucination.

## Stack

| Layer | Tool |
|---|---|
| LLM | `qwen3.5:4b` via Ollama (local) |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector Store | ChromaDB (persistent, on-disk) |
| PDF Parsing | PyMuPDF |
| API | FastAPI |

## Usage

```bash
# 1. Pull models
ollama pull qwen3.5:4b
ollama pull nomic-embed-text

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your PDFs to docs/

# 4. Ingest documents
python ingest.py

# 5. Chat
python cli.py
```
## How It Works

```
PDF → chunk → embed → ChromaDB
Question → embed → similarity search → top chunks → Qwen3:4b → Answer
```
Everything runs locally. No API keys. No internet. Your documents stay yours.

## Notes

- Re-run `ingest.py` whenever you add new documents
- Answers are grounded in your uploaded docs only
- Built on Linux with 8GB RAM — fully CPU-based

*Learning project — DevOps & MLOps journey.*
