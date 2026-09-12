import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
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

def classify_node(state: TicketState) -> dict:
    structured_llm = llm.with_structured_output(TicketClassification)

    prompt = f"""Classify this customer support ticket.

Ticket: {state["ticket_text"]}

Assign a category, an urgency level, and a one-sentence reasoning for your choices."""

    classification = structured_llm.invoke(prompt)

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

    draft = structured_llm.invoke(prompt)
    return {"draft": draft}

CONFIDENCE_THRESHOLD = 0.7

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