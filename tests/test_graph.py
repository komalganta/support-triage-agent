import json

import pytest

from src.graph import build_graph


@pytest.fixture
def graph():
    return build_graph()


@pytest.fixture
def tickets():
    with open("data/tickets.json") as f:
        return json.load(f)


def test_every_ticket_gets_a_routing_decision(graph, tickets):
    for ticket in tickets:
        result = graph.invoke({"ticket_text": ticket["text"]})
        assert result.get("routing_decision") in {"auto_send", "escalate"}

def test_data_loss_ticket_escalates(graph):
    result = graph.invoke({
        "ticket_text": "URGENT: all our team's data disappeared this morning, we're panicking"
    })
    assert result["routing_decision"] == "escalate"


def test_payment_method_question_auto_sends(graph):
    result = graph.invoke({
        "ticket_text": "How do I update my payment method on file?"
    })
    assert result["routing_decision"] == "auto_send"