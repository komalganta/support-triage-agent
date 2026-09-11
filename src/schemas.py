from typing import Literal
from pydantic import BaseModel, Field
from typing import Optional, TypedDict

Category = Literal["billing", "bug_report", "feature_request", "account_access"]
Urgency = Literal["low", "medium", "high"]

class TicketClassification(BaseModel):
    category: Category = Field(description="Best-fit category for the ticket")
    urgency: Urgency = Field(description="How time-sensitive this ticket is")
    reasoning: str = Field(description="One-sentence justification for the above")

class RetrievedArticle(BaseModel):
    title: str
    content: str
    similarity_score: float

class DraftReply(BaseModel):
    reply_text: str = Field(description="The proposed reply to the customer")
    confidence: float = Field(
        description="0-1 self-assessed confidence that this reply is accurate "
        "and fully grounded in the retrieved articles"
    )
    grounded_in: list[str] = Field(
        description="Titles of the retrieved articles actually used"
    )

class TicketState(TypedDict, total=False):
    ticket_text: str
    classification: TicketClassification
    retrieved_articles: list[RetrievedArticle]
    draft: DraftReply
    routing_decision: Optional[Literal["auto_send", "escalate"]]
    routing_reason: Optional[str]