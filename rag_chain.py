import ollama
from retriever import retrieve
from config import *

# System prompt — tells LLM its role and rules
# ⚠️ Security: explicit instruction to stay in scope
SYSTEM_PROMPT = """You are a DevOps expert assistant.
Answer ONLY using the provided context chunks below.
If the answer is not in the context, say: "I don't have that information in the loaded documents."
Never make up facts. Cite the source document and page when answering.
Be concise and technical."""

def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into readable context block.
    LLM reads this as its "open book" before answering.
    """
    context_parts = []
    for i, chunk in enumerate(chunks):
        # Label each chunk with source for citation
        context_parts.append(
            f"[Chunk {i+1} | {chunk['source']} p.{chunk['page']}]\n{chunk['text']}"
        )
    # Join all chunks with separator
    return "\n\n---\n\n".join(context_parts)


def ask(question: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks for question
    2. Build context string
    3. Send to Qwen3 4B via Ollama
    4. Return answer + sources
    """
    # Step 1: Retrieve
    chunks = retrieve(question)

    if not chunks:
        return {
            "answer":  "No documents ingested yet. Run: python ingest.py",
            "sources": []
        }

    # Step 2: Build context
    context = build_context(chunks)

    # Step 3: Build prompt (context + question together)
    user_message = f"""Context from DevOps documents:

{context}

---

Question: {question}"""

    # Step 4: Call Ollama (streaming=False → wait for full response)
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_message}
        ],
        options={
            "temperature": 0.1,  # low temp = factual, not creative
            "num_ctx":     4096, # context window size (fits in 8GB)
        }
    )

    # Extract answer text from response
    answer = response["message"]["content"]

    # Collect unique sources (deduplicate same file/page)
    seen = set()
    sources = []
    for chunk in chunks:
        key = f"{chunk['source']}:p{chunk['page']}"
        if key not in seen:
            seen.add(key)
            sources.append({"file": chunk["source"], "page": chunk["page"]})

    return {"answer": answer, "sources": sources}