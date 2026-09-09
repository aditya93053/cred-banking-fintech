import time, uuid, json, hashlib
from .guardrails import mask_pii, is_prompt_injection
from .schemas import VerdictModel

MAX_CHARS = 5000
TOKEN_BUDGET = 1500
CACHE = {}

def normalized_query(q): return " ".join(q.lower().split())

def risk_classification(query):
    q=query.lower()
    if any(x in q for x in ["fraud","dispute","closure"]): return "Medium"
    return "Low"

def cached_get(q): return CACHE.get(hashlib.sha256(normalized_query(q).encode()).hexdigest())
def cached_put(q, value): CACHE[hashlib.sha256(normalized_query(q).encode()).hexdigest()] = value

def review(draft, context):
    grounded = any(s in draft.answer for s in context) if context else not draft.sources
    if dra supported by the local knowledge base.")
