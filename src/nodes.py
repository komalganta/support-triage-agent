import os

from dotenv import load_dotenv
from groq import RateLimitError
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from src.kb import load_kb_articles, get_vectorstore, retrieve as kb_retrieve
from src.schemas import DraftReply, TicketClassification, TicketState

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY"),
)

_articles = load_kb_articles()
vectorstore = get_vectorstore(_articles)

CONFIDENCE_THRESHOLD = 0.7


@retry(
    retry=lambda retry_state: isinstance(retry_state.outcome.exception(), RateLimitError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def invoke_with_retry(structured_llm, prompt):
    """Calls the LLM, automatically retrying with backoff if we hit a rate limit.

    Groq's free tier caps tokens-per-minute. Running a batch of tickets back
    to back can hit that ceiling mid-run. Instead of crashing the whole batch,
    this waits (2s, then roughly doubling up to 30s) and retries, up to 5
    attempts, before giving up.
    """
    return structured_llm.invoke(prompt)


def classify_node(state: TicketState) -> dict:
    structured_llm = llm.with_structured_output(TicketClassification)

    prompt = f"""Classify this customer support ticket.

Ticket: {state["ticket_text"]}

Assign a category, an urgency level, and a one-sentence reasoning for your choices.

Urgency guidelines:
- high: customer is fully blocked (can't log in, can't pay, data loss), explicit
  urgency language ("urgent", "ASAP"), or a security/financial risk
- medium: a real problem, but the customer has a workaround or isn't fully blocked
  (e.g. a specific feature misbehaving while the rest of the product works)
- low: cosmetic issues, feature requests, or general questions with no real
  time pressure

A broken feature is not automatically "high" — only mark high if the customer
is genuinely blocked or there's explicit time pressure."""

    classification = invoke_with_retry(structured_llm, prompt)
    return {"classification": classification}


def retrieve_node(state: TicketState) -> dict:
    articles = kb_retrieve(vectorstore, state["ticket_text"], k=3)
    return {"retrieved_articles": articles}


def draft_node(state: TicketState) -> dict:
    structured_llm = llm.with_structured_output(DraftReply)

    articles_text = "\n\n".join(
        f"Article: {a.title}\n{a.content}" for a in state["retrieved_articles"]
    )

    prompt = f"""You are drafting a reply to a customer support ticket.

Ticket: {state["ticket_text"]}

Here are the most relevant help-center articles we found (they may or may not
actually be relevant to this specific ticket):

{articles_text}

Write a reply to the customer based ONLY on the information in these articles.
If none of the articles actually address the customer's issue, say so honestly
in your reply instead of making something up, and reflect that with a low
confidence score.

Also report:
- confidence (0-1): how confident you are that this reply is accurate and
  fully grounded in the articles above
- grounded_in: titles of the articles you actually used"""

    draft = invoke_with_retry(structured_llm, prompt)
    return {"draft": draft}


def route_node(state: TicketState) -> dict:
    urgency = state["classification"].urgency
    confidence = state["draft"].confidence

    if urgency == "high":
        return {
            "routing_decision": "escalate",
            "routing_reason": "High urgency ticket routed to human review.",
        }

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "routing_decision": "escalate",
            "routing_reason": f"Low draft confidence ({confidence:.2f}) below threshold ({CONFIDENCE_THRESHOLD}).",
        }

    return {
        "routing_decision": "auto_send",
        "routing_reason": f"High confidence ({confidence:.2f}) and non-urgent; safe to auto-send.",
    }


def send_node(state: TicketState) -> dict:
    print(f"[AUTO-SEND] Reply sent to customer:\n{state['draft'].reply_text}")
    return {}


def escalate_node(state: TicketState) -> dict:
    print(f"[ESCALATED] {state['routing_reason']}\nDraft for human reviewer:\n{state['draft'].reply_text}")
    return {}