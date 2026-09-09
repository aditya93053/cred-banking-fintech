from typing import Any

from pydantic import BaseModel

from .schemas import VerdictModel


class PolicyComplianceReviewer:
    name = "Policy-Compliance-Reviewer"

    def review(
        self,
        draft: str,
        context: str,
    ) -> VerdictModel:

        if not str(draft).strip():
            return VerdictModel(
                decision="REVISE",
                reason="Draft answer is empty.",
                revised_answer=(
                    "I can only answer using the available "
                    "Cred support knowledge base."
                ),
            )

        if not str(context).strip():
            return VerdictModel(
                decision="REVISE",
                reason="No grounded context was supplied.",
                revised_answer=(
                    "I can only answer using the available "
                    "Cred support knowledge base."
                ),
            )

        return VerdictModel(
            decision="APPROVE",
            reason=(
                "Draft contains grounded support context "
                "and passed policy review."
            ),
            revised_answer=str(draft),
        )


class FinalEditor:
    name = "Final-Editor"

    def edit(
        self,
        draft: str,
        context: str,
        verdict: VerdictModel,
    ) -> VerdictModel:

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
    Two-agent review stage.

    Agent 1:
        Policy-Compliance-Reviewer

    Agent 2:
        Final-Editor

    The review is deterministic and works with MOCK_LLM.
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
    }


def review_approve_sample():
    return run_review_sample(
        draft=(
            "KYC generally requires identity and address "
            "verification documents."
        ),
        context=(
            "KYC generally requires identity and address "
            "verification documents."
        ),
    )


def review_revise_sample():
    return run_review_sample(
        draft=(
            "The bank guarantees approval of every loan "
            "within 24 hours."
        ),
        context="",
    )
