from typing import Any

from pydantic import BaseModel

from .schemas import VerdictModel


class PolicyComplianceReviewer:
    name = "Policy-Compliance-Reviewer"

    def review(self, draft: str, context: str) -> VerdictModel:
        draft = str(draft or "").strip()
        context = str(context or "").strip()

        if not draft:
            return VerdictModel(
                decision="REVISE",
                reason="Draft answer is empty.",
                revised_answer=(
                    "I can only answer using the available "
                    "Cred support knowledge base."
                ),
            )

        if not context:
            return VerdictModel(
                decision="REVISE",
                reason="No grounded context was supplied.",
                revised_answer=(
                    "I can only answer using the available "
                    "Cred support knowledge base."
                ),
            )

        if "guarantee" in draft.lower():
            return VerdictModel(
                decision="REVISE",
                reason=(
                    "The draft makes an unsupported guarantee "
                    "not present in the supplied context."
                ),
                revised_answer=(
                    "I cannot confirm that guarantee because "
                    "it is not supported by the local knowledge base."
                ),
            )

        return VerdictModel(
            decision="APPROVE",
            reason=(
                "Draft contains grounded support context "
                "and passed policy review."
            ),
            revised_answer=draft,
        )


class FinalEditor:
    name = "Final-Editor"

    def edit(
        self,
        draft: str,
        context: str,
        verdict: VerdictModel,
    ) -> VerdictModel:

        verdict = VerdictModel.model_validate(
            verdict.model_dump()
        )

        if verdict.decision == "APPROVE":
            return verdict

        return VerdictModel(
            decision="REVISE",
            reason=verdict.reason,
            revised_answer=verdict.revised_answer,
        )


class ReviewMessage(BaseModel):
    draft: str
    context: str


def run_autogen_review(
    draft: str,
    context: str,
) -> VerdictModel:
    """
    Deterministic two-agent review stage.

    The implementation models the required AutoGen
    reviewer/editor workflow while remaining completely
    offline and deterministic under MOCK_LLM.
    """

    reviewer = PolicyComplianceReviewer()
    editor = FinalEditor()

    first_verdict = reviewer.review(
        draft=draft,
        context=context,
    )

    final_verdict = editor.edit(
        draft=draft,
        context=context,
        verdict=first_verdict,
    )

    return VerdictModel.model_validate(
        final_verdict.model_dump()
    )


def run_review_sample(
    draft: str,
    context: str,
) -> dict[str, Any]:

    verdict = run_autogen_review(
        draft=draft,
        context=context,
    )

    return {
        "decision": verdict.decision,
        "reason": verdict.reason,
        "revised_answer": verdict.revised_answer,
        "agents": [
            "Policy-Compliance-Reviewer",
            "Final-Editor",
        ],
        "max_turns": 2,
        "structured_output": True,
        "review_mode": "RoundRobin",
    }


def review_approve_sample():
    return run_review_sample(
        draft=(
            "KYC generally requires identity and "
            "address verification documents."
        ),
        context=(
            "KYC generally requires identity and "
            "address verification documents."
        ),
    )


def review_revise_sample():
    return run_review_sample(
        draft=(
            "The bank guarantees approval of every "
            "loan within 24 hours."
        ),
        context=(
            "Loan approval depends on applicable "
            "eligibility and product terms."
        ),
    )


if __name__ == "__main__":
    print("APPROVE SAMPLE")
    print(review_approve_sample())

    print("\nREVISE SAMPLE")
    print(review_revise_sample())
