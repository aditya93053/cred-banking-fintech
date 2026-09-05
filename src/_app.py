import json, time, uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from .guardrails import mask_pii, is_prompt_injection
from .crew import run_support
from .memory import SessionMemory
from .governance import cached_get, cached_put, risk_classification, MAX_CHARS
from .review import run_autogen_review

app = FastAPI(title="Cred Banking & FinTech Support Agent")
memory = SessionMemory()

class AskRequest(BaseModel):
    query: str
    session_id: str = "default"

class AddDocumentRequest(BaseModel):
    topic: str
    text: str

@app.get("/health")
def health(): return {"status":"ok","mock_llm":True}

@app.post("/ask")
def ask(req: AskRequest):
    if len(req.query) > MAX_CHARS: raise HTTPException(413, "Request too large")
    safe = mask_pii(req.query)
    if is_prompt_injection(safe):
        return {"answer":"I can't follow prompt-injection instructions. Please ask a normal Cred support question.","refused":True}
    hit = cached_get(safe)
    if hit:
        return {**hit, "cache_hit": True}
    memory.add(req.session_id, "user", safe)
    draft = run_support(safe, req.session_id)
    context = draft.sources
    verdict = run_autogen_review(draft, context)
    out = draft.model_copy(update={"answer": verdict.revised_answer}).model_dump()
    out["risk"] = risk_classification(safe)
    out["cache_hit"] = False
    cached_put(safe, out)
    return out

@app.post("/add-document")
def add_document(req: AddDocumentRequest):
    if not req.text.strip(): raise HTTPException(400, "text required")
    return {"accepted": True, "topic": req.topic, "note":"Document accepted for runtime extension."}

@app.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            query = await websocket.receive_text()
            if len(query) > MAX_CHARS:
                await websocket.send_json({"error":"Request too large"}); continue
            if is_prompt_injection(query):
                await websocket.send_json({"answer":"I can't follow prompt-injection instructions.","refused":True}); continue
            draft = run_support(mask_pii(query), session_id)
            verdict = run_autogen_review(draft, draft.sources)
            await websocket.send_json(draft.model_copy(update={"answer":verdict.revised_answer}).model_dump())
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
