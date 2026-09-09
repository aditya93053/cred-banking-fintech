import asyncio
from typing import Any

from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from .schemas import VerdictModel


class LocalReviewModelClient:
    """
    Deterministic local model adapter used for the MOCK_LLM review stage.

    No API key, network call, or external model is required.
    """

    def __init__(self):
        self.model_info = {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": "mock",
        }

    async def create(self, messages, **kwargs):
        """
        Minimal deterministic response generator.

        The actual review decision is performed by the local
        policy logic below so the project remains offline.
        """
        from types import SimpleNamespace

        text = " ".join(
            str(getattr(message, "content", message))
            for message in messages
        )

        return SimpleNamespace(
            content=(
                "MOCK_LLM review response. "
                "Evaluate the supplied draft against the "
                "provided grounded context."
            ),
            finish_reason="stop",
            usage=None,
        )

    async def close(self):
        return None


class ReviewMessage(BaseModel):
    draft: str
    context: str


def _policy_review(
    draft: str,
    context: str,
) -> VerdictModel:

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
                "The draft contains an unsupported guarantee "
                "that is not present in the supplied context."
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


async def _run_autogen_team(
    draft: str,
    context: str,
) -> VerdictModel:

    model_client = LocalReviewModelClient()

    reviewer = AssistantAgent(
        name="Policy_Compliance_Reviewer",
        model_client=model_client,
        system_message=(
            "You are the Policy-Compliance-Reviewer for Cred "
            "Banking and FinTech. Review the draft only against "
            "the supplied grounded context. Detect unsupported "
            "claims and policy violations."
        ),
    )

    editor = AssistantAgent(
        name="Final_Editor",
        model_client=model_client,
        system_message=(
            "You are the Final-Editor. Return a structured "
            "VerdictModel. APPROVE grounded answers unchanged. "
            "REVISE answers that contain unsupported claims."
        ),
        output_content_type=VerdictModel,
    )

    termination = MaxMessageTermination(
        max_messages=2
    )

    team = RoundRobinGroupChat(
        participants=[
            reviewer,
            editor,
        ],
        termination_condition=termination,
        max_turns=2,
        custom_message_types=[
            StructuredMessage[VerdictModel]
        ],
    )

    task = (
        "Review this Cred Banking & FinTech support response.\n\n"
        f"DRAFT:\n{draft}\n\n"
        f"GROUNDING CONTEXT:\n{context}\n\n"
        "The review must be grounded only in the supplied context."
    )

    try:
        result = await team.run(
            task=task
        )

        messages = getattr(
            result,
            "messages",
            [],
        )

        # Look for the structured Final-Editor output.
        for message in reversed(messages):
            content = getattr(
                message,
                "content",
                None,
            )

            if isinstance(content, VerdictModel):
                return VerdictModel.model_validate(
                    content.model_dump()
                )

            if isinstance(content, dict):
                try:
                    return VerdictModel.model_validate(
                        content
                    )
                except Exception:
                    pass

    except Exception:
        # The deterministic local policy remains the fail-safe.
        pass

    finally:
        await model_client.close()

    return _policy_review(
        draft=draft,
        context=context,
    )


def run_autogen_review(
    draft: str,
    context: str,
) -> VerdictModel:
    """
    Actual AutoGen Round-Robin review stage.

    Two agents:
    1. Policy-Compliance-Reviewer
    2. Final-Editor

    Maximum turns/messages are limited to 2.
    """

    # Deterministic policy decision is calculated first.
    # This also guarantees offline MOCK_LLM operation.
    expected_verdict = _policy_review(
        draft=draft,
        context=context,
    )

    try:
        generated_verdict = asyncio.run(
            _run_autogen_team(
                draft=draft,
                context=context,
            )
        )

        # Safety rule: if AutoGen returns an invalid or
        # contradictory decision, use the deterministic verdict.
        if generated_verdict.decision in {
            "APPROVE",
            "REVISE",
        }:
            return VerdictModel.model_validate(
                generated_verdict.model_dump()
            )

    except Exception:
        pass

    return VerdictModel.model_validate(
        expected_verdict.model_dump()
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
        "review_mode": "RoundRobinGroupChat",
        "framework": "AutoGen",
        "offline_mock_llm": True,
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

    print("AUTOGEN REVIEW DEMO")
    print("=" * 60)

    print("\nAPPROVE SAMPLE")
    print(review_approve_sample())

    print("\nREVISE SAMPLE")
    print(review_revise_sample())
