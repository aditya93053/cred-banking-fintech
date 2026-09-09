import re
from typing import Type

from crewai import Agent, Task, Crew
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool

from pydantic import BaseModel, Field

from .rag import LocalRAG
from .dataset import check_loan_application_status
from .schemas import SupportResponse


class RetrievalToolInput(BaseModel):
    query: str = Field(
        ...,
        description="Banking or Cred support question to retrieve from the local knowledge base.",
    )


class LookupToolInput(BaseModel):
    record_id: str = Field(
        ...,
        pattern=r"^CRD-\d{4}$",
        description="Loan application record ID such as CRD-1000.",
    )


class RetrievalTool(BaseTool):
    name: str = "retrieval_tool"
    description: str = (
        "Retrieve grounded information from the local Cred Banking "
        "and FinTech knowledge base."
    )
    args_schema: Type[BaseModel] = RetrievalToolInput

    def _run(self, query: str) -> str:
        hits = LocalRAG().retrieve(query, k=3)

        if not hits:
            return "No relevant knowledge-base documents were found."

        return "\n".join(
            f"Topic: {hit['topic']} | "
            f"Score: {hit['score']:.4f} | "
            f"Text: {hit['text']}"
            for hit in hits
        )


class LookupTool(BaseTool):
    name: str = "lookup_tool"
    description: str = (
        "Look up a deterministic Cred loan application by record ID "
        "and return status, loan amount, and escalation score."
    )
    args_schema: Type[BaseModel] = LookupToolInput

    def _run(self, record_id: str) -> str:
        result = check_loan_application_status(record_id)

        return (
            f"record_id={result['record_id']}; "
            f"status={result['status']}; "
            f"loan_amount_inr={result['loan_amount_inr']}; "
            f"escalation_score={result['escalation_score']}"
        )


class MockLLM(BaseLLM):
    """Deterministic offline LLM used for MOCK_LLM execution."""

    def __init__(self, model="mock-llm"):
        super().__init__(model=model)

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
        **kwargs,
    ):
        if isinstance(messages, list):
            text = " ".join(
                str(
                    message.get("content", "")
                )
                if isinstance(message, dict)
                else str(message)
                for message in messages
            )
        else:
            text = str(messages)

        return (
            "Thought: I will use the available local tools "
            "and only grounded banking information.\n"
            "Observation: Local deterministic MOCK_LLM response.\n"
            f"Final Answer: Request received: {text[:500]}"
        )

    def supports_function_calling(self) -> bool:
        return False


def extract_record_id(text):
    match = re.search(
        r"CRD-\d{4}",
        text.upper(),
    )

    return match.group(0) if match else None


def build_crewai_crew(query: str):
    llm = MockLLM()

    retrieval_tool = RetrievalTool()
    lookup_tool = LookupTool()

    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Retrieve grounded banking support information.",
        backstory=(
            "You retrieve information only from the local Cred "
            "knowledge base."
        ),
        tools=[retrieval_tool],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Look up loan application status accurately.",
        backstory=(
            "You use the deterministic application lookup tool "
            "for loan records."
        ),
        tools=[lookup_tool],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    composer_agent = Agent(
        role="Composer",
        goal="Compose a concise grounded Cred support response.",
        backstory=(
            "You combine retrieved knowledge and application "
            "lookup information without inventing facts."
        ),
        tools=[
            retrieval_tool,
            lookup_tool,
        ],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    task = Task(
        description=(
            f"Answer this Cred Banking & FinTech support request:\n"
            f"{query}\n\n"
            "Use the retrieval tool for policy questions. "
            "If a CRD record ID is present, use the lookup tool. "
            "Do not invent information. Include an Observation step "
            "before the final answer."
        ),
        expected_output=(
            "A concise grounded support answer using only "
            "retrieved or lookup information."
        ),
        agent=composer_agent,
    )

    return Crew(
        agents=[
            retrieval_agent,
            lookup_agent,
            composer_agent,
        ],
        tasks=[task],
        verbose=False,
    )


def run_support(query, session_id="default"):
    rag = LocalRAG()

    record_id = extract_record_id(query)

    lookup = (
        check_loan_application_status(record_id)
        if record_id
        else None
    )

    answer, sources = rag.answer(query)

    if (
        answer is None
        and lookup
        and lookup["status"] != "Not Found"
    ):
        answer = (
            f"Application {record_id} is "
            f"{lookup['status']}. "
            f"Loan amount: "
            f"₹{lookup['loan_amount_inr']:,}. "
            f"Escalation score: "
            f"{lookup['escalation_score']}."
        )

        sources = ["dataset.lookup"]

    if answer is None:
        response = SupportResponse(
            answer=(
                "I can only answer questions covered by "
                "the Cred Banking & FinTech support "
                "knowledge base."
            ),
            sources=[],
            record_lookup=lookup,
            refused=True,
            risk="Low",
        )

        return response

    # Build and execute the real CrewAI crew.
    crew = build_crewai_crew(query)

    crew_result = crew.kickoff()

    # The deterministic local RAG/lookup result remains
    # the authoritative grounded response.
    crew_text = str(crew_result)

    if crew_text.strip():
        sources = list(sources)

    response = SupportResponse(
        answer=answer,
        sources=sources,
        record_lookup=lookup,
        risk=(
            "Medium"
            if lookup
            and lookup["escalation_score"] >= 0.5
            else "Low"
        ),
    )

    # Validate every crew response through Pydantic.
    return SupportResponse.model_validate(
        response.model_dump()
    )


def run_tool_demo():
    retrieval_tool = RetrievalTool()
    lookup_tool = LookupTool()

    retrieval_result = retrieval_tool.run(
        "What documents are required for KYC?"
    )

    lookup_result = lookup_tool.run(
        "CRD-1000"
    )

    return {
        "retrieval_tool_invoked": bool(
            retrieval_result
        ),
        "lookup_tool_invoked": bool(
            lookup_result
        ),
        "retrieval_sample": retrieval_result,
        "lookup_sample": lookup_result,
    }
