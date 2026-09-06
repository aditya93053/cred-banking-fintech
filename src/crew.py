from crewai import Agent, Task, Crew
from crewai.llms.base_llm import BaseLLM

from .rag import LocalRAG
from .dataset import check_loan_application_status
from .schemas import SupportResponse

import re


class MockLLM(BaseLLM):
    """Deterministic local LLM used for zero-key/offline execution."""

    def __init__(self, model="mock-llm"):
        super().__init__(model=model)

    def call(self, messages, **kwargs):
        if isinstance(messages, list):
            text = " ".join(
                str(m.get("content", "")) if isinstance(m, dict) else str(m)
                for m in messages
            )
        else:
            text = str(messages)

        return (
            "Observation: Local deterministic MOCK_LLM response. "
            "Use only the retrieved banking context and available tool results. "
            f"Request: {text[:500]}"
        )


class RetrievalTool:
    name = "retrieval_tool"
    description = "Retrieve grounded support context from the local knowledge base."

    def run(self, query):
        return LocalRAG().retrieve(query)


class LookupTool:
    name = "lookup_tool"
    description = "Look up a deterministic loan application record."

    def run(self, record_id):
        return check_loan_application_status(record_id)


def extract_record_id(text):
    match = re.search(r"CRD-\d{4}", text.upper())
    return match.group(0) if match else None


def run_support(query, session_id="default"):
    rag = LocalRAG()
    record_id = extract_record_id(query)

    lookup = (
        check_loan_application_status(record_id)
        if record_id
        else None
    )

    answer, sources = rag.answer(query)

    if answer is None and lookup and lookup["status"] != "Not Found":
        answer = (
            f"Application {record_id} is {lookup['status']}. "
            f"Loan amount: ₹{lookup['loan_amount_inr']:,}. "
            f"Escalation score: {lookup['escalation_score']}."
        )
        sources = ["dataset.lookup"]

    if answer is None:
        return SupportResponse(
            answer=(
                "I can only answer questions covered by the "
                "Cred Banking & FinTech support knowledge base."
            ),
            sources=[],
            record_lookup=lookup,
            refused=True,
            risk="Low",
        )

    return SupportResponse(
        answer=answer,
        sources=sources,
        record_lookup=lookup,
        risk=(
            "Medium"
            if lookup and lookup["escalation_score"] >= 0.5
            else "Low"
        ),
    )


def build_crewai_crew():
    llm = MockLLM()

    retrieval_tool = RetrievalTool()
    lookup_tool = LookupTool()

    retrieval = Agent(
        role="Retrieval Agent",
        goal="Retrieve grounded banking context.",
        backstory="Uses the local RAG retrieval tool.",
        tools=[retrieval_tool],
        llm=llm,
    )

    lookup = Agent(
        role="Lookup Agent",
        goal="Look up application status.",
        backstory="Uses deterministic application records.",
        tools=[lookup_tool],
        llm=llm,
    )

    composer = Agent(
        role="Composer",
        goal="Compose concise grounded responses.",
        backstory="Never invents unsupported policy.",
        tools=[retrieval_tool, lookup_tool],
        llm=llm,
    )

    task = Task(
        description=(
            "Compose a grounded support answer using the available "
            "retrieval and lookup tools. Include only supported information."
        ),
        expected_output="A validated support response.",
        agent=composer,
    )

    return Crew(
        agents=[retrieval, lookup, composer],
        tasks=[task],
    )
