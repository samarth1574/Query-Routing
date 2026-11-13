"""Request and response models (they also drive the OpenAPI docs at /docs)."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["payments", "cards", "loans", "account", "fraud"]
Channel = Literal["web", "app", "email", "chat", "phone", "branch", "api"]


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      examples=["Someone used my card 4111 1111 1111 1111 without my permission"])
    channel: Channel = "web"
    customer_ref: str | None = Field(default=None, max_length=64)
    create_ticket: bool = True

    @field_validator("text")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("Query text is empty.")
        return v


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    channel: Channel = "api"
    create_tickets: bool = False


class Intent(BaseModel):
    intent: str
    label: str
    confidence: float


class Rule(BaseModel):
    rule: str
    detail: str


class Keyword(BaseModel):
    word: str
    start: int
    end: int
    impact: float


class Decision(BaseModel):
    text: str
    pii_redacted: list[str]
    category: Category
    label: str
    intent: Intent | None
    intent_alternatives: list[Intent]
    confidence: float
    model_category: Category
    secondary_category: Category | None
    probabilities: dict[str, float]
    needs_review: bool
    department: str
    priority: Literal["urgent", "high", "normal"]
    sla_hours: float
    due_at: str
    rules_applied: list[Rule]
    keywords: list[Keyword]
    model_version: str
    latency_ms: float
    ticket_ref: str | None = None


class TicketUpdate(BaseModel):
    status: Literal["open", "in_progress", "resolved"] | None = None
    category: Category | None = Field(default=None, description="Re-route to this category (recorded as feedback).")
    assignee: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=2000)
    actor: str = Field(default="agent", max_length=64)


class ResolveRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      examples=["I don't recognise a payment of £45 at Amazon, please help"])
    channel: Channel = "chat"
    customer_ref: str | None = Field(default=None, max_length=64, examples=["CUST-001"])
    create_ticket: bool = True
    parallel: bool = Field(default=True, description="Run independent tool calls and retrieval concurrently.")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("Query text is empty.")
        return v


class ResolveBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    customer_refs: list[str | None] | None = None
    workers: int | None = Field(default=None, ge=1, le=32, description="Queries resolved concurrently.")
    parallel_tools: bool = True
