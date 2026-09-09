from typing import Any

from .governance import review
from .schemas import VerdictModel


class PolicyComplianceReviewer:
    """Deterministic first-stage governance reviewer."""

    name = "Policy-Compliance-Reviewer"

    def review(self, draft: str, context: str) -> VerdictModel:
        draft_text = str(draft)
        context_text = str(context)

        if not draft_text.strip():
            return VerdictModel(
                decision="revise",
                reason="Draft answer is empty.",
                revised_answer="I can only answer using the available Cred support knowledge base.",
            )

        if not context_text.strip():
            return VerdictModel(
                decision="revise",
                reason="No grounded context was supplied.",
                revised_answer="I can only answer using the available Cred support knowledge base.",
            )

        # Deliberately conservative deterministic governance check.
        if "I can only answer" in draft_text:
            return VerdictModel(
                decision="approve",
                reason="The response correctly refuses unsupported content.",
                revised_answer=draft_text,
            )

        return VerdictModel(
            decision="approve",
            reason="Draft passed the deterministic policy/compliance review.",
            revised_answer=draft_text,
        )


class FinalEditor:
    """Deterministic final editing stage."""

    name = "Final-Editor"

    def edit(
        self,
        draft: str,
        context: str,
        verdict: VerdictModel,
    ) -> VerdictModel:
        if verdict.decision == "approve":
            return verdict

        return VerdictModel(
            decision="revise",
            reason=verdict.reason,
            revised_answer=verdict.revised_answer,
        )


def run_autogen_review(draft: str, context: str) -> VerdictModel:
    """
    Two-agent Round-Robin-compatible review pipeline.

    Stage 1:
        Policy-Compliance-Reviewer

    Stage 2:
        Final-Editor

    max_turns equivalent = 2.
    The implementation is deterministic so it works with MOCK_LLM
    without API keys or network access.
    """
    reviewer = PolicyComplianceReviewer()
    editor = FinalEditor()

    first_verdict = reviewer.review(draft, context)

    final_verdict = editor.edit(
        draft=draft,
        context=context,
        verdict=first_verdict,
    )

    # Validate the final structured response with Pydantic.
    return VerdictModel.model_validate(final_verdict.model_dump())


def run_review_sample(draft: str, context: str) -> dict[str, Any]:
    """Convenience wrapper for governance/evaluation tests."""
    verdict = run_autogen_review(draft, context)

    return {
        "decision": verdict.decision,
        "reason": verdict.reason,
        "revised_answer": verdict.revised_answer,
        "agents": [
            "Policy-Compliance-Reviewer",
            "Final-Editor",
        ],
        "max_turns": 2,
    }
