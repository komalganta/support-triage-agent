from langgraph.graph import StateGraph, END

from src.schemas import TicketState
from src.nodes import classify_node, retrieve_node, draft_node, route_node, send_node, escalate_node


def route_condition(state: TicketState) -> str:
    return state["routing_decision"]


def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("draft", draft_node)
    graph.add_node("route", route_node)
    graph.add_node("send", send_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "route")

    graph.add_conditional_edges(
        "route",
        route_condition,
        {"auto_send": "send", "escalate": "escalate"},
    )

    graph.add_edge("send", END)
    graph.add_edge("escalate", END)

    return graph.compile()