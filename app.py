from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_chain import ask

# Create FastAPI app instance
app = FastAPI(
    title="DevOps RAG API",
    description="Local RAG over DevOps documents",
    version="1.0"
)

# Request schema — validates incoming JSON shape
class QueryRequest(BaseModel):
    question: str  # required field

    # ⚠️ Security: limit question length to prevent prompt injection
    class Config:
        max_anystr_length = 1000  # 1000 char max

# POST /ask → takes question, returns answer + sources
@app.post("/ask")
def ask_question(req: QueryRequest):
    # Basic input sanitization
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Run RAG pipeline
    result = ask(question)
    return result

# Health check — useful for monitoring
@app.get("/health")
def health():
    return {"status": "ok", "model": "qwen3:4b"}