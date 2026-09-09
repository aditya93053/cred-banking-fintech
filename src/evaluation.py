from typing import Dict, List

from .rag import LocalRAG


EVALUATION_QUERIES = [
    "What documents are required for KYC?",
    "How is EMI calculated for a personal loan?",
    "What is the credit card fee structure?",
    "How can I dispute a fraudulent transaction?",
    "What factors affect my credit score?",
]


RELEVANT_TOPICS = [
    ["kyc_documents"],
    ["emi_rules"],
    ["credit_card_fee_structure"],
    ["fraud_dispute_process"],
    ["credit_score_factors"],
]


def document_level_precision_recall(
    queries: List[str],
    relevant_topics: List[List[str]],
    k: int = 3,
) -> Dict:

    rag = LocalRAG()

    total_retrieved = 0
    total_relevant_retrieved = 0
    total_relevant = 0

    rows = []

    for query, relevant in zip(
        queries,
        relevant_topics,
    ):
        hits = rag.retrieve(
            query,
            k=k,
        )

        retrieved_topics = [
            hit["topic"]
            for hit in hits
        ]

        relevant_set = set(relevant)
        retrieved_set = set(
            retrieved_topics
        )

        true_positive = len(
            relevant_set & retrieved_set
        )

        retrieved_count = len(
            retrieved_set
        )

        relevant_count = len(
            relevant_set
        )

        total_retrieved += retrieved_count
        total_relevant_retrieved += true_positive
        total_relevant += relevant_count

        rows.append(
            {
                "query": query,
                "relevant_documents": list(
                    relevant_set
                ),
                "retrieved_documents": retrieved_topics,
                "true_positive": true_positive,
                "retrieved_count": retrieved_count,
                "relevant_count": relevant_count,
            }
        )

    precision = (
        total_relevant_retrieved
        / total_retrieved
        if total_retrieved
        else 0.0
    )

    recall = (
        total_relevant_retrieved
        / total_relevant
        if total_relevant
        else 0.0
    )

    return {
        "precision": round(
            precision,
            4,
        ),
        "recall": round(
            recall,
            4,
        ),
        "total_retrieved": total_retrieved,
        "total_relevant_retrieved": (
            total_relevant_retrieved
        ),
        "total_relevant": total_relevant,
        "rows": rows,
    }


def print_evaluation(result: Dict):
    print("\nRAG DOCUMENT-LEVEL EVALUATION")
    print("=" * 60)

    for index, row in enumerate(
        result["rows"],
        start=1,
    ):
        print(f"\nQuery {index}:")
        print(f"  {row['query']}")

        print(
            "  Relevant documents: "
            f"{row['relevant_documents']}"
        )

        print(
            "  Retrieved documents: "
            f"{row['retrieved_documents']}"
        )

        print(
            "  True Positive: "
            f"{row['true_positive']}"
        )

    print("\n" + "-" * 60)

    print(
        "Precision arithmetic:"
    )

    print(
        f"TP / Retrieved = "
        f"{result['total_relevant_retrieved']} / "
        f"{result['total_retrieved']} = "
        f"{result['precision']}"
    )

    print(
        "\nRecall arithmetic:"
    )

    print(
        f"TP / Relevant = "
        f"{result['total_relevant_retrieved']} / "
        f"{result['total_relevant']} = "
        f"{result['recall']}"
    )

    print(
        "\nRecommendation:"
    )

    if (
        result["precision"] >= 0.80
        and result["recall"] >= 0.80
    ):
        print(
            "Retrieval quality is acceptable "
            "for the evaluated support queries."
        )
    else:
        print(
            "Retrieval quality should be improved "
            "before production use."
        )


def run_rag_evaluation():
    result = document_level_precision_recall(
        queries=EVALUATION_QUERIES,
        relevant_topics=RELEVANT_TOPICS,
        k=3,
    )

    print_evaluation(result)

    return result


def evaluation_summary():
    result = run_rag_evaluation()

    return {
        "query_count": len(
            result["rows"]
        ),
        "precision": result["precision"],
        "recall": result["recall"],
        "recommendation": (
            "PASS"
            if (
                result["precision"] >= 0.80
                and result["recall"] >= 0.80
            )
            else "IMPROVE"
        ),
    }


if __name__ == "__main__":
    evaluation_summary()
