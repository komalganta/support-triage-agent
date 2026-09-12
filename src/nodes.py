import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.kb import load_kb_articles, get_vectorstore, retrieve as kb_retrieve

from src.schemas import TicketClassification, TicketState

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