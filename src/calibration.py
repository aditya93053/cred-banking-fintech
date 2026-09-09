from .rag import LocalRAG


IN_SCOPE_QUERIES = [
    "What documents are required for KYC?",
    "How is EMI calculated for a personal loan?",
    "What factors affect credit score?",
    "What is the prepayment penalty?",
    "What is the minimum balance requirement?",
]

OUT_OF_SCOPE_QUERIES = [
    "What is the weather today?",
    "Who won the football match yesterday?",
    "How do I cook pasta?",
]


def calibrate_threshold():
    rag = LocalRAG()

    in_scores = []
    out_scores = []

    for query in IN_SCOPE_QUERIES:
        hits = rag.retrieve(query, k=1)

        if hits:
            in_scores.append(float(hits[0]["score"]))

    for query in OUT_OF_SCOPE_QUERIES:
        hits = rag.retrieve(query, k=1)

        if hits:
            out_scores.append(float(hits[0]["score"]))

    if not in_scores or not out_scores:
        raise RuntimeError(
            "Calibration requires both in-scope and out-of-scope scores."
        )

    max_out = max(out_scores)
    min_in = min(in_scores)

    if max_out < min_in:
        threshold = (max_out + min_in) / 2
    else:
        # Conservative fallback when the score distributions overlap.
        threshold = min_in * 0.90

    return {
        "threshold": round(threshold, 4),
        "in_scope_scores": [round(x, 4) for x in in_scores],
        "out_of_scope_scores": [round(x, 4) for x in out_scores],
        "min_in_scope": round(min_in, 4),
        "max_out_of_scope": round(max_out, 4),
    }


if __name__ == "__main__":
    result = calibrate_threshold()

    print("RAG Threshold Calibration")
    print("-------------------------")
    print(f"Calibrated threshold: {result['threshold']}")
    print(f"In-scope scores: {result['in_scope_scores']}")
    print(f"Out-of-scope scores: {result['out_of_scope_scores']}")
    print(f"Minimum in-scope score: {result['min_in_scope']}")
    print(f"Maximum out-of-scope score: {result['max_out_of_scope']}")
