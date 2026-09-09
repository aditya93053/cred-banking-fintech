from typing import List, Optional

from pydantic import BaseModel, Field


class LoanLookup(BaseModel):
    record_id: str
    status: str
    loan_amount_inr: Optional[int] = None
    escalation_score: float = Field(
        ge=0.0,
        le=1.0,
    )


class SupportResponse(BaseModel):
    answer: str
    sources: List[str] = Field(
        default_factory=list
    )
    record_lookup: Optional[LoanLookup] = None
    refused: bool = False
    risk: str = "Low"
    trace_id: str = ""

    @classmethod
    def validate_response(cls, data):
        return cls.model_validate(data)


class VerdictModel(BaseModel):
    decision: str
    reason: str
    revised_answer: str

    @classmethod
    def validate_verdict(cls, data):
        return cls.model_validate(data)
