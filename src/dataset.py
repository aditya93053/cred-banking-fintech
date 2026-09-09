import random
from datetime import datetime, timedelta


LOAN_TYPES = [
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Education Loan",
    "Business Loan",
]

STATUSES = [
    "Submitted",
    "Under Review",
    "Approved",
    "Rejected",
    "Disbursed",
]


def generate_dataset(seed=42, n=50):
    random.seed(seed)

    records = []

    fraud_count = 0

    for i in range(n):
        record_id = f"CRD-{1000 + i}"

        loan_type = random.choice(LOAN_TYPES)
        status = random.choice(STATUSES)

        loan_amount = random.randint(
            50000,
            5000000,
        )

        created_at = (
            datetime(2026, 1, 1)
            + timedelta(
                days=random.randint(0, 240)
            )
        ).strftime("%Y-%m-%d")

        fraud_flag = random.random() < 0.20

        if fraud_flag:
            fraud_count += 1

        records.append(
            {
                "record_id": record_id,
                "loan_type": loan_type,
                "loan_amount_inr": loan_amount,
                "status": status,
                "created_at": created_at,
                "fraud_flag": fraud_flag,
            }
        )

    # Guarantee every loan category appears at least 3 times.
    for index, loan_type in enumerate(LOAN_TYPES):
        for offset in range(3):
            records[
                index * 3 + offset
            ]["loan_type"] = loan_type

    # Guarantee every status appears at least once.
    for index, status in enumerate(STATUSES):
        records[index]["status"] = status

    return records


DATASET = generate_dataset()


def _escalation_score(record):
    fraud_component = 1.0 if record["fraud_flag"] else 0.0

    created = datetime.strptime(
        record["created_at"],
        "%Y-%m-%d",
    )

    today = datetime(2026, 9, 9)

    age_days = max(
        0,
        (today - created).days,
    )

    normalized_recency = min(
        age_days / 240.0,
        1.0,
    )

    score = (
        0.70 * fraud_component
        + 0.30 * normalized_recency
    )

    return round(
        min(max(score, 0.0), 1.0),
        4,
    )


def check_loan_application_status(record_id):
    for record in DATASET:
        if record["record_id"] == record_id:
            return {
                "record_id": record["record_id"],
                "status": record["status"],
                "loan_amount_inr": record[
                    "loan_amount_inr"
                ],
                "escalation_score": _escalation_score(
                    record
                ),
            }

    return {
        "record_id": record_id,
        "status": "Not Found",
        "loan_amount_inr": None,
        "escalation_score": 0.0,
    }


def dataset_statistics():
    category_counts = {
        loan_type: sum(
            1
            for record in DATASET
            if record["loan_type"] == loan_type
        )
        for loan_type in LOAN_TYPES
    }

    status_counts = {
        status: sum(
            1
            for record in DATASET
            if record["status"] == status
        )
        for status in STATUSES
    }

    fraud_count = sum(
        1
        for record in DATASET
        if record["fraud_flag"]
    )

    fraud_percentage = (
        fraud_count / len(DATASET)
    ) * 100

    return {
        "records": len(DATASET),
        "category_counts": category_counts,
        "status_counts": status_counts,
        "fraud_count": fraud_count,
        "fraud_percentage": round(
            fraud_percentage,
            2,
        ),
        "seed": 42,
        "amount_range_inr": [
            50000,
            5000000,
        ],
    }


if __name__ == "__main__":
    print(dataset_statistics())

    print(
        check_loan_application_status(
            "CRD-1000"
        )
    )
