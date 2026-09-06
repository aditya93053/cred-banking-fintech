from crewai import Agent, Task, Crew
from .rag import LocalRAG
from .dataset import check_loan_application_status
from .guardrails import in_scope
from .schemas import SupportResponse
import re

class RetrievalTool:
    name = "retrieval_tool"
    description = "Retrieve grounded support context from the local knowledge base."
    def run(self, query): return LocalRAG().retrieve(query)

class LookupTool:
    name = "lookup_tool"
    description = "Look up a deterministic loan application record."
    def run(self, record_id): return check_loan_application_status(record_id)

def extract_record_id(text):
    m = re.search(r"CRD-\d{4}", text.upper())
    return m.group(0) if m else None

def run_support(query, session_id="default"):
    rag = LocalRAG()
    record_id = extract_record_id(query)
    lookup = check_loan_application_status(record_id) if record_id else None
    answer, sources = rag.answer(query)
    if answer is None and lookup and lookup["status"] != "Not Found":
        answer = f"Application {record_id} is {lookup['status']}. Loan amount: ₹{lookup['loan_amount_inr']:,}. Escalation score: {lookup['escalation_score']}."
        sources = ["dataset.lookup"]
    if answer is None:
        return SupportResponse(answer="I can only answer questions covered by the Cred Banking & FinTech support knowledge base.", sources=[], record_lookup=lookup, refused=True, risk="Low")
    return SupportResponse(answer=answer, sources=sources, record_lookup=lookup, risk="Medium" if lookup and lookup["escalation_score"] >= .5 else "Low")

def build_crewai_crew():
    retrieval = Agent(role="Retrieval Agent", goal="Retrieve grounded banking context", backstory="Uses the local RAG retrieval tool.", tools=[])
    lookup = Agent(role="Lookup Agent", goal="Look up application status", backstory="Uses deterministic application records.", tools=[])
    composer = Agent(role="Composer", goal="Compose concise grounded responses", backstory="Never invents unsupported policy.", tools=[])
    task = Task(description="Compose a grounded support answer.", expected_output="A validated support response.", agent=composer)
    return Crew(agents=[retrieval, lookup, composer], tasks=[task])
