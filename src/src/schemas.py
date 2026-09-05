from pydantic import BaseModel, Field
from typing import List, Optional

class SupportResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    record_lookup: Optional[dict] = None
    refused: bool = False
    risk: str = "Low"
    trace_id: str = ""

class VerdictModel(BaseModel):
    decision: str
    reason: str
    revised_answer: str
