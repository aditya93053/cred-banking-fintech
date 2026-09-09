import hashlib

from .schemas import VerdictModel


MAX_CHARS = 5000
TOKEN_BUDGET = 1500

CACHE = {}

AUTONOMY_LEVEL = "least"


def least_autonomy_check():
    return {
        "allowed": True,
        "autonomy": AUTONOMY_LEVEL,
        "reason": (
            "Only the minimum support actions required "
            "for the request are permitted."
        ),
    }


def risk_classification(query: str) -> str:
    query_lower = query.lower()

    high_risk_terms = [
        "hack",
        "steal",
        "password",
        "otp",
        "credential",
    ]

    medium_risk_terms = [
        "fraud",
        "dispute",
        "chargeback",
        "unauthorized",
        "scam",
        "closure",
    ]

    if any(
        term in query_lower
        for term in high_risk_terms
    ):
        return "High"

    if any(
        term in query_lower
        for term in medium_risk_terms
    ):
        return "Medium"

    return "Low"


def estimate_tokens(query: str) -> int:
    """
    Deterministic approximate token estimator.

    A conservative character-based approximation is used
    because MOCK_LLM runs without an external tokenizer/API.
    """
    return max(
        1,
        (len(query) + 3) // 4,
    )


def runtime_budget_check(query: str) -> dict:
    estimated_tokens = estimate_tokens(query)

    if estimated_tokens > TOKEN_BUDGET:
        return {
            "allowed": False,
            "estimated_tokens": estimated_tokens,
            "token_budget": TOKEN_BUDGET,
            "reason": (
                "Request exceeds the runtime token budget."
            ),
        }

    return {
        "allowed": True,
        "estimated_tokens": estimated_tokens,
        "token_budget": TOKEN_BUDGET,
        "reason": "Runtime token budget passed.",
    }


def oversized_request_check(query: str) -> dict:
    actual_chars = len(query)

    if actual_chars > MAX_CHARS:
        return {
            "allowed": False,
            "actual_chars": actual_chars,
            "max_chars": MAX_CHARS,
            "reason": (
                "Request exceeds the maximum "
                "character limit."
            ),
        }

    return {
        "allowed": True,
        "actual_chars": actual_chars,
        "max_chars": MAX_CHARS,
        "reason": "Request size is within the allowed limit.",
    }


def governance_check(query: str) -> dict:
    if not isinstance(query, str):
        return {
            "allowed": False,
            "risk": "High",
            "reason": "Query must be a string.",
        }

    if not query.strip():
        return {
            "allowed": False,
            "risk": "Low",
            "reason": "Query cannot be empty.",
        }

    autonomy = least_autonomy_check()

    risk = risk_classification(query)

    budget = runtime_budget_check(query)

    if not budget["allowed"]:
        return {
            "allowed": False,
            "risk": "High",
            "reason": budget["reason"],
            "estimated_tokens": budget["estimated_tokens"],
            "token_budget": budget["token_budget"],
            "autonomy": autonomy["autonomy"],
        }

    size = oversized_request_check(query)

    if not size["allowed"]:
        return {
            "allowed": False,
            "risk": "High",
            "reason": size["reason"],
            "actual_chars": size["actual_chars"],
            "max_chars": size["max_chars"],
            "autonomy": autonomy["autonomy"],
        }

    return {
        "allowed": True,
        "risk": risk,
        "reason": "All governance layers passed.",
        "estimated_tokens": budget["estimated_tokens"],
        "token_budget": TOKEN_BUDGET,
        "actual_chars": size["actual_chars"],
        "max_chars": MAX_CHARS,
        "autonomy": autonomy["autonomy"],
    }


def normalized_query(query: str) -> str:
    return " ".join(
        query.lower().split()
    )


def _cache_key(query: str) -> str:
    normalized = normalized_query(query)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def cached_get(query: str):
    return CACHE.get(
        _cache_key(query)
    )


def cached_put(query: str, value):
    CACHE[
        _cache_key(query)
    ] = value


def cache_size() -> int:
    return len(CACHE)


def clear_cache():
    CACHE.clear()


def review(draft, context):
    """
    Deterministic policy review.

    A safely refused answer is approved.
    A response with grounded sources is approved.
    Otherwise it is revised to a safe fallback.
    """

    context = context or []

    if draft.refused:
        return VerdictModel(
            decision="APPROVE",
            reason="Response is safely refused.",
            revised_answer=draft.answer,
        )

    if draft.sources and context:
        return VerdictModel(
            decision="APPROVE",
            reason=(
                "Response contains grounded source support."
            ),
            revised_answer=draft.answer,
        )

    return VerdictModel(
        decision="REVISE",
        reason="Draft lacks grounded support.",
        revised_answer=(
            "I cannot provide that answer because it "
            "is not supported by the local knowledge base."
        ),
    )


def governance_evidence(query: str) -> dict:
    result = governance_check(query)

    return {
        "allowed": result.get("allowed"),
        "risk": result.get("risk"),
        "reason": result.get("reason"),
        "autonomy": result.get(
            "autonomy",
            AUTONOMY_LEVEL,
        ),
        "estimated_tokens": result.get(
            "estimated_tokens",
            estimate_tokens(query),
        ),
        "token_budget": TOKEN_BUDGET,
        "actual_chars": len(query),
        "max_chars": MAX_CHARS,
        "cache_key": _cache_key(query),
    }


def cache_evidence(query: str, value):
    """
    Demonstrates cache miss followed by cache hit.
    """

    first = cached_get(query)

    if first is None:
        cached_put(query, value)

    second = cached_get(query)

    return {
        "first_lookup_hit": first is not None,
        "second_lookup_hit": second is not None,
        "cache_size": cache_size(),
        "normalized_query": normalized_query(query),
    }


if __name__ == "__main__":
    print("Governance demo")
    print(
        governance_evidence(
            "What documents are required for KYC?"
        )
    )

    print("\nCache demo")
    print(
        cache_evidence(
            "What documents are required for KYC?",
            {
                "answer": "KYC requires identity and address verification."
            },
        )
    )
