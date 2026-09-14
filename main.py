import argparse
import json

from dotenv import load_dotenv

from src.graph import build_graph

load_dotenv()


def run_single(ticket_text: str) -> None:
    graph = build_graph()
    result = graph.invoke({"ticket_text": ticket_text})

    print(f"\nCategory: {result['classification'].category}")
    print(f"Urgency: {result['classification'].urgency}")
    print(f"Routing: {result['routing_decision']} — {result['routing_reason']}")


def run_batch(path: str) -> None:
    graph = build_graph()
    with open(path) as f:
        tickets = json.load(f)

    for ticket in tickets:
        result = graph.invoke({"ticket_text": ticket["text"]})
        print(f"\n{'=' * 60}")
        print(f"TICKET: {ticket['text']}")
        print(f"Category: {result['classification'].category} | Urgency: {result['classification'].urgency}")
        print(f"Routing: {result['routing_decision']} — {result['routing_reason']}")
        print(f"Reasoning: {result['classification'].reasoning}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticket", help="A single ticket to run through the agent")
    group.add_argument("--batch", help="Path to a JSON file of tickets")
    args = parser.parse_args()

    if args.ticket:
        run_single(args.ticket)
    else:
        run_batch(args.batch)