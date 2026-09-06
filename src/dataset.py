"""Deterministic Cred-style loan dataset."""
from __future__ import annotations
import random
from dataclasses import dataclass, asdict

CATEGORIES = ["Personal Loan","Home Loan","Auto Loan","Education Loan","Business Loan"]
STATUSES = ["Submitted","Under Review","Approved","Rejected","Disbursed"]

def generate_dataset(n: int = 50, seed: int = 42):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        rows.append({
            "record_id": f"CRD-{i+1:04d}",
            "category": CATEGORIES[i % len(CATEGORIES)] if i < len(CATEGORIES) else rng.choice(CATEGORIES),
            "status": STATUSES[i % len(STATUSES)] if i < len(STATUSES) else rng.choice(STATUSES),
            "loan_amount_inr": rng.randrange(50000, 5000001, 50000),
            "days_since_created": rng.randint(0, 30),
            "flagged_for_fraud_review": False,
        })
    fraud_count = max(5, min(15, round(n * 0.20)))
    for idx in rng.sample(range(n), fraud_count):
        rows[idx]["flagged_for_fraud_review"] = True
    return rows

DATASET = generate_dataset()
LOOKUP = {r["record_id"]: r for r in DATASET}

def check_loan_application_status(record_id: str):
    r = LOOKUP.get(record_id)
    if not r:
        return {"record_id": record_id, "status": "Not Found", "loan_amount_inr": None, "escalation_score": 0.0}
    recency = r["days_since_created"] / 30
    fraud = 1.0 if r["flagged_for_fraud_review"] else 0.0
    score = round(min(1.0, 0.7 * fraud + 0.3 * recency), 3)
    return {"record_id": record_id, "status": r["status"], "loan_amount_inr": r["loan_amount_inr"], "escalation_score": score}
