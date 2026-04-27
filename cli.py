import ollama
from retriever import retrieve
from rag_chain import build_context, SYSTEM_PROMPT
from config import *

print("🤖 DevOps RAG (qwen3.5:4b) | streaming mode")
print("Type 'quit' to exit\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue

    # Retrieve relevant chunks
    chunks = retrieve(question)
    if not chunks:
        print("❌ No docs ingested. Run: python ingest.py\n")
        continue

    context = build_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    print("\n🤖 ", end="", flush=True)

    # stream=True → tokens print as they generate, no more blind waiting
    for part in ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        stream=True,  # ← KEY CHANGE
        options={
            "temperature": 0.1,
            "num_ctx": 2048,  # reduced from 4096 → fits 8GB better, faster
        }
    ):
        # Each part has delta token — print immediately
        print(part["message"]["content"], end="", flush=True)

    print("\n")

    # Show sources
    seen = set()
    print("📚 Sources:")
    for c in chunks:
        key = f"{c['source']}:p{c['page']}"
        if key not in seen:
            seen.add(key)
            print(f"   • {c['source']} — page {c['page']}")
    print()