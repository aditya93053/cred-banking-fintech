import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from .guardrails import mask_pii, is_prompt_injection
from .crew import run_support
from .memory import SessionMemory
from .governance import (
    cached_get,
    cached_put,
    governance_check,
    MAX_CHARS,
)
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

    safe_query = mask_pii(query)

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

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mock_llm": True,
        "telemetry": "disabled",
    }


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

    safe_query = mask_pii(query)

    if is_prompt_injection(safe_query):
        response = {
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

        write_log(
            trace_id,
            "/ask",
            safe_query,
            start_time,
            "prompt_injection_blocked",
            False,
            "High",
        )

        return response

    governance = governance_check(safe_query)

    if not governance["allowed"]:
        response = {
            "answer": (
                "Your request cannot be processed because it "
                "does not pass the runtime governance policy."
            ),
            "sources": [],
            "refused": True,
            "risk": governance.get("risk", "High"),
            "governance_reason": governance.get("reason"),
            "trace_id": trace_id,
            "cache_hit": False,
        }

        write_log(
            trace_id,
            "/ask",
            safe_query,
            start_time,
            "governance_rejected",
            False,
            governance.get("risk", "High"),
        )

        return response

    cached = cached_get(safe_query)

    if cached:
        response = dict(cached)
        response["cache_hit"] = True
        response["trace_id"] = trace_id

        write_log(
            trace_id,
            "/ask",
            safe_query,
            start_time,
            "cache_hit",
            True,
            governance["risk"],
        )

        return response

    memory.add(
        req.session_id,
        "user",
        safe_query,
    )

    draft = run_support(
        safe_query,
        req.session_id,
    )

    context = draft.sources or []

    verdict = run_autogen_review(
        draft.answer,
        "\n".join(context),
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
        "autonomy": governance.get("autonomy"),
        "estimated_tokens": governance.get("estimated_tokens"),
        "token_budget": governance.get("token_budget"),
        "max_chars": governance.get("max_chars"),
    }

    cached_put(
        safe_query,
        response,
    )

    write_log(
        trace_id,
        "/ask",
        safe_query,
        start_time,
        "success",
        False,
        governance["risk"],
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
        "note": "Document accepted for runtime extension.",
        "trace_id": trace_id,
    }


@app.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
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

            safe_query = mask_pii(query)

            if is_prompt_injection(safe_query):
                await websocket.send_json(
                    {
                        "answer": (
                            "I can't follow prompt-injection "
                            "instructions."
                        ),
                        "refused": True,
                        "risk": "High",
                        "trace_id": trace_id,
                    }
                )

                write_log(
                    trace_id,
                    "/ws/chat",
                    safe_query,
                    start_time,
                    "prompt_injection_blocked",
                    False,
                    "High",
                )

                continue

            governance = governance_check(safe_query)

            if not governance["allowed"]:
                await websocket.send_json(
                    {
                        "answer": (
                            "Your request cannot be processed "
                            "because it exceeds the runtime "
                            "governance policy."
                        ),
                        "refused": True,
                        "risk": governance.get("risk", "High"),
                        "governance_reason": governance.get(
                            "reason"
                        ),
                        "trace_id": trace_id,
                    }
                )
                continue

            memory.add(
                session_id,
                "user",
                safe_query,
            )

            cached = cached_get(safe_query)

            if cached:
                response = dict(cached)
                response["cache_hit"] = True
                response["trace_id"] = trace_id

                await websocket.send_json(response)

                write_log(
                    trace_id,
                    "/ws/chat",
                    safe_query,
                    start_time,
                    "cache_hit",
                    True,
                    governance["risk"],
                )

                continue

            draft = run_support(
                safe_query,
                session_id,
            )

            context = draft.sources or []

            verdict = run_autogen_review(
                draft.answer,
                "\n".join(context),
            )

            response = draft.model_copy(
                update={
                    "answer": verdict.revised_answer,
                    "trace_id": trace_id,
                }
            ).model_dump()

            response["risk"] = governance["risk"]
            response["cache_hit"] = False

            cached_put(
                safe_query,
                response,
            )

            await websocket.send_json(response)

            write_log(
                trace_id,
                "/ws/chat",
                safe_query,
                start_time,
                "success",
                False,
                governance["risk"],
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
