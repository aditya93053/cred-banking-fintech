from typing import List, Dict

from .rag import LocalRAG


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

    for query, relevant in zip(queries, relevant_topics):
        hits = rag.retrieve(query, k=k)

        retrieved_topics = [
            hit["topic"]
            for hit in hits
        ]

        relevant_set = set(relevant)
        retrieved_set = set(retrieved_topics)

        true_positive = len(
            relevant_set & retrieved_set
        )

        total_retrieved += len(retrieved_set)
        total_relevant_retrieved += true_positive
        total_relevant += len(relevant_set)

        rows.append(
            {
                "query": query,
                "relevant": list(relevant_set),
                "retrieved": retrieved_topics,
                "true_positive": true_positive,
            }
        )

    precision = (
        total_relevant_retrieved / total_retrieved
        if total_retrieved
        else 0.0
    )

    recall = (
        total_relevant_retrieved / total_relevant
        if total_relevant
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "total_retrieved": total_retrieved,
        "total_relevant_retrieved": total_relevant_retrieved,
        "total_relevant": total_relevant,
        "rows": rows,
    }


def run_rag_evaluation():
    queries = [
        "What documents are required for KYC?",
        "How is EMI calculated?",
        "What is the credit card fee structure?",
        "How can I dispute a fraudulent transaction?",
        "What factors affect my credit score?",
    ]

    relevant_topics = [
        ["kyc_documents"],
        ["emi_rules"],
        ["credit_card_fee_structure"],
        ["fraud_dispute_process"],
        ["credit_score_factors"],
    ]

    result = document_level_precision_recall(
        queries,
        relevant_topics,
        k=3,
    )

    print("RAG Evaluation")
    print("-------------")
    print(
        f"Precision = {result['total_relevant_retrieved']} / "
        f"{result['total_retrieved']} = "
        f"{result['precision']}"
    )
    print(
        f"Recall = {result['total_relevant_retrieved']} / "
        f"{result['total_relevant']} = "
        f"{result['recall']}"
    )

    return result


if __name__ == "__main__":
    run_rag_evaluation()
