# Cred Banking & FinTech — Domain Support Agent

> **Track completed: Banking & FinTech (Cred).**

This repository implements the final capstone as a reproducible, deterministic reference implementation. It contains the dataset, local RAG core, CrewAI crew, session memory, guardrails, FastAPI API/WebSocket, structured evaluation, AutoGen review stage, governance controls, and caching.

## Dataset design choices
- Random seed: `42`
- Records: `50`
- Categories: Personal Loan, Home Loan, Auto Loan, Education Loan, Business Loan
- Statuses: Submitted, Under Review, Approved, Rejected, Disbursed
- Category/status sampling is deterministic and repaired so every category has >=3 records and every status has >=1 record.
- Loan amount range: ₹50,000–₹50,00,000
- Days since created: 0–30
- Fraud-review rate is deterministically kept within 10–30%.

## Required KB topics
The local knowledge base contains 12 documents covering:
1. Loan eligibility by loan type
2. EMI rules
3. Credit-card fee structure
4. KYC documents
5. Fraud-dispute process
6. Account closure
7. Interest-rate slabs
8. Prepayment penalty
9. Minimum balance
10. Credit-score factors
11. Joint-account rules
12. NRI eligibility

## Architecture
`User -> FastAPI -> guardrails/cache -> CrewAI agents -> local RAG + deterministic lookup -> Pydantic response -> AutoGen governance review`

No external LLM API key is required. `MOCK_LLM=1` is the default.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m src.app
```

API:
- `GET /health`
- `POST /ask`
- `POST /add-document`
- `WS /ws/chat/{session_id}`

Run tests:
```bash
pytest -q
```

Run evaluation:
```bash
python -m src.evaluation
```

Run calibration:
```bash
python -m src.calibration
```

## Determinism / governance
The application disables telemetry, masks PAN/Aadhaar/account-number patterns, rejects prompt-injection attempts, refuses unsupported questions, rejects oversized requests, applies a per-request token/cost budget, classifies risk, and records JSONL audit events.

## Submission checklist
- [x] Deterministic >=40-record dataset
- [x] 12 KB topics
- [x] Two chunking strategies + separate collections
- [x] Similarity-threshold calibration artifact
- [x] Retrieval precision/recall
- [x] Loan status lookup + escalation score
- [x] >=3 CrewAI roles
- [x] Session memory + reset
- [x] Pydantic response validation
- [x] PII / injection / groundedness guardrails
- [x] FastAPI + WebSocket
- [x] JSONL logging + trace IDs
- [x] 15-query evaluation
- [x] AutoGen 2-agent review stage
- [x] Four-layer governance
- [x] Normalized-query cache
