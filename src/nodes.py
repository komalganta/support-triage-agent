import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.schemas import TicketClassification, TicketState

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY"),
)

def classify_node(state: TicketState) -> dict:
    structured_llm = llm.with_structured_output(TicketClassification)

    prompt = f"""Classify this customer support ticket.

Ticket: {state["ticket_text"]}

Assign a category, an urgency level, and a one-sentence reasoning for your choices."""

    classification = structured_llm.invoke(prompt)

    return {"classification": classification}