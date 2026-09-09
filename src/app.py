import os

# Disable telemetry before importing CrewAI.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import json
import time
import uuid
from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel

from .crew import run_support
from .governance import (
    MAX_CHARS,
    cached_get,
    cached_put,
    governance_check,
)
from .guardrails import (
    is_prompt_injection,
    mask_pii,
)
from .memory import SessionMemory
from .review import run_autogen_review


app = FastAPI(
    title="Cred Banking & FinTech Support Agent",
    version="1.0.0",
)


memory = SessionMemory()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "requests.jsonl"


class AskRequest(BaseModel):
    query: str
    session_id: str = "default"


class AddDocumentRequest(BaseModel):
    topic: str
    text: str


def write_log(
    trace_id,
    endpoint,
    query,
    start_time,
    status,
    cache_hit=False,
    risk="Low",
):
    elapsed_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    safe_query = mask_pii(str(query))

    record = {
        "timestamp": time.time(),
        "trace_id": trace_id,
        "endpoint": endpoint,
        "query": safe_query,
        "query_length": len(safe_query),
        "status": status,
        "cache_hit": cache_hit,
        "risk": risk,
        "latency_ms": elapsed_ms,
    }

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mock_llm": True,
        "telemetry": "disabled",
    }


def process_query(
    query: str,
    session_id: str,
    trace_id: str,
):
    safe_query = mask_pii(query)

    # Layer 1: prompt-injection protection.
    if is_prompt_injection(safe_query):
        return {
            "answer": (
                "I can't follow prompt-injection instructions. "
                "Please ask a normal Cred support question."
            ),
            "sources": [],
            "refused": True,
            "risk": "High",
            "trace_id": trace_id,
            "cache_hit": False,
        }

    # Layer 2: governance.
    governance = governance_check(safe_query)

    if not governance["allowed"]:
        return {
            "answer": (
                "Your request cannot be processed because "
                "it does not pass the runtime governance policy."
            ),
            "sources": [],
            "refused": True,
            "risk": governance.get("risk", "High"),
            "governance_reason": governance.get("reason"),
            "trace_id": trace_id,
            "cache_hit": False,
        }

    # Layer 3: normalized response cache.
    cached = cached_get(safe_query)

    if cached is not None:
        response = dict(cached)
        response["cache_hit"] = True
        response["trace_id"] = trace_id
        return response

    # Store user message for multi-turn session memory.
    memory.add(
        session_id,
        "user",
        safe_query,
    )

    # Use previous session history as context.
    history = memory.history(session_id)

    history_context = ""

    if len(history) > 1:
        previous_messages = history[-5:]
        history_context = "\n".join(
            f"{item['role']}: {item['text']}"
            for item in previous_messages
        )

    query_for_support = safe_query

    if history_context:
        query_for_support = (
            f"Previous session context:\n"
            f"{history_context}\n\n"
            f"Current request:\n"
            f"{safe_query}"
        )

    draft = run_support(
        query_for_support,
        session_id,
    )

    # Review the actual answer against actual source topics.
    context = "\n".join(
        draft.sources or []
    )

    verdict = run_autogen_review(
        draft.answer,
        context,
    )

    response = draft.model_copy(
        update={
            "answer": verdict.revised_answer,
            "trace_id": trace_id,
        }
    ).model_dump()

    response["risk"] = governance["risk"]
    response["cache_hit"] = False

    response["governance"] = {
        "autonomy": governance.get(
            "autonomy"
        ),
        "estimated_tokens": governance.get(
            "estimated_tokens"
        ),
        "token_budget": governance.get(
            "token_budget"
        ),
        "actual_chars": governance.get(
            "actual_chars"
        ),
        "max_chars": governance.get(
            "max_chars"
        ),
    }

    # Store assistant response.
    memory.add(
        session_id,
        "assistant",
        response["answer"],
    )

    # Cache final safe response.
    cached_put(
        safe_query,
        response,
    )

    return response


@app.post("/ask")
def ask(req: AskRequest):
    start_time = time.perf_counter()
    trace_id = str(uuid.uuid4())

    query = req.query

    if not isinstance(query, str) or not query.strip():
        write_log(
            trace_id,
            "/ask",
            query,
            start_time,
            "rejected_empty",
        )

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    response = process_query(
        query,
        req.session_id,
        trace_id,
    )

    write_log(
        trace_id,
        "/ask",
        query,
        start_time,
        "success",
        response.get("cache_hit", False),
        response.get("risk", "Low"),
    )

    return response


@app.post("/add-document")
def add_document(req: AddDocumentRequest):
    start_time = time.perf_counter()
    trace_id = str(uuid.uuid4())

    if not req.topic.strip():
        raise HTTPException(
            status_code=400,
            detail="topic required",
        )

    if not req.text.strip():
        raise HTTPException(
            status_code=400,
            detail="text required",
        )

    if len(req.text) > MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail="Document is too large.",
        )

    safe_text = mask_pii(req.text)

    if is_prompt_injection(safe_text):
        write_log(
            trace_id,
            "/add-document",
            safe_text,
            start_time,
            "prompt_injection_blocked",
            False,
            "High",
        )

        return {
            "accepted": False,
            "refused": True,
            "reason": "Prompt injection detected.",
            "trace_id": trace_id,
        }

    from .rag import LocalRAG

    rag = LocalRAG()

    chunks = rag._chunk_text(
        safe_text
    )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No usable document chunks found.",
        )

    ids = []
    documents = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        ids.append(
            f"runtime-{req.topic}-{index}"
        )

        documents.append(chunk)

        metadatas.append(
            {
                "topic": req.topic,
                "chunk_id": index,
                "source": "runtime_add_document",
            }
        )

    embeddings = rag.model.encode(
        documents,
        normalize_embeddings=True,
    ).tolist()

    rag.collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Prevent stale cached answers after adding a document.
    from .governance import clear_cache

    clear_cache()

    write_log(
        trace_id,
        "/add-document",
        safe_text,
        start_time,
        "accepted",
        False,
        "Low",
    )

    return {
        "accepted": True,
        "topic": req.topic,
        "chunks_added": len(chunks),
        "storage": "ChromaDB",
        "trace_id": trace_id,
    }


@app.post("/reset/{session_id}")
def reset_session(session_id: str):
    memory.reset(session_id)

    return {
        "reset": True,
        "session_id": session_id,
    }


@app.get("/memory/{session_id}")
def get_memory(session_id: str):
    return {
        "session_id": session_id,
        "messages": memory.history(session_id),
    }


@app.websocket(
    "/ws/chat/{session_id}"
)
async def ws_chat(
    websocket: WebSocket,
    session_id: str,
):
    await websocket.accept()

    try:
        while True:
            start_time = time.perf_counter()
            trace_id = str(uuid.uuid4())

            query = await websocket.receive_text()

            if not query.strip():
                await websocket.send_json(
                    {
                        "answer": "Please enter a question.",
                        "refused": True,
                        "trace_id": trace_id,
                    }
                )
                continue

            response = process_query(
                query,
                session_id,
                trace_id,
            )

            await websocket.send_json(
                response
            )

            write_log(
                trace_id,
                "/ws/chat",
                query,
                start_time,
                "success",
                response.get(
                    "cache_hit",
                    False,
                ),
                response.get(
                    "risk",
                    "Low",
                ),
            )

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
