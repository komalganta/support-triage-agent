# Support Ticket Triage & Response Agent

An AI agent that reads a customer support ticket and figures out what to do with it, built using **LangGraph** (part of the LangChain ecosystem).

Given a ticket, the agent:
1. **Classifies** it: category (billing, bug report, feature request, or account access) and urgency (low, medium, high)
2. **Retrieves** the most relevant help center articles using hosted embeddings and a vector search
3. **Drafts** a reply based only on what it actually found. If nothing relevant exists, it says so instead of making something up
4. **Routes** the ticket, either auto sending the reply or escalating it to a human, based on urgency and how confident the draft actually is

## Why LangGraph (not just a LangChain chain)

A basic chain can only go step 1 to step 2 to step 3 to step 4, in a straight line, every time. It can't make a real decision partway through.

This project needed a real decision: should this ticket be handled automatically, or does it need a human? LangGraph lets the graph actually branch based on what the earlier steps found. High urgency or a low confidence draft sends the ticket down a different path than a routine, well matched one. That branching logic is the whole point of using LangGraph here.

## How it works

```
classify -> retrieve -> draft -> route --+-- auto_send
                                          +-- escalate
```

- **classify**: an LLM call (Groq) that returns a structured result, not just text. It's an actual typed object with `category`, `urgency`, and a `reasoning` field explaining the choice
- **retrieve**: turns the ticket into a vector using Hugging Face's hosted embedding API, then searches a local Chroma vector store for the closest matching help articles
- **draft**: another LLM call, grounded only in the retrieved articles. It also self reports a confidence score and which articles it actually used
- **route**: plain Python logic (no LLM here on purpose). High urgency always escalates. Otherwise, low draft confidence escalates. Everything else auto sends

## Project layout

```
support-triage-agent/
├── src/
│   ├── schemas.py     # Data models (Pydantic + TypedDict) used everywhere
│   ├── kb.py           # Hosted embeddings, Chroma vector store, retrieval
│   ├── nodes.py         # The four node functions: classify, retrieve, draft, route
│   └── graph.py           # Wires the nodes into an actual LangGraph graph
├── data/
│   ├── tickets.json      # Sample tickets to test with
│   └── kb_articles.json  # Mock help center articles
├── tests/
│   └── test_graph.py      # Automated tests, including a couple of known outcome checks
├── main.py                 # Run a single ticket or a whole batch from the terminal
├── streamlit_app.py          # Simple web UI for demos
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

You'll need two free API keys:
- **Groq** (console.groq.com/keys): powers the classify and draft LLM calls
- **Hugging Face** (huggingface.co/settings/tokens): powers the hosted embeddings, so nothing heavy runs on your own machine

Drop both into your `.env` file.

## Run it

Single ticket:
```bash
python main.py --ticket "My payment failed twice and I was charged both times, please help urgently"
```

The whole sample set at once:
```bash
python main.py --batch data/tickets.json
```

Or the web UI, for a nicer demo:
```bash
streamlit run streamlit_app.py
```

## A bug I actually ran into (and fixed)

Early on, almost every ticket was getting classified as `urgency: high`, even a stuck CSV export button, which is annoying but not actually urgent. Turns out the classify prompt never explained what "urgency" was supposed to mean, so the model was just treating "something is broken" as the same thing as "this is time sensitive."

Fixed it by giving the prompt real criteria for each urgency level, plus an explicit line telling it that a broken feature isn't automatically high urgency. After that, urgency levels matched what a person would actually expect. Blocked or urgent things stayed high, everything else spread out across medium and low.

## Status

- [x] Pydantic schemas + TicketState
- [x] Hosted embeddings (Hugging Face) + Chroma vector store
- [x] Classify + draft nodes with structured LLM output
- [x] Conditional routing (real branch, not just a linear chain)
- [x] CLI (single ticket + batch mode)
- [x] Automated tests (batch coverage + known outcome regression tests)
- [x] Streamlit demo UI

## Future Plans

- Add a groundedness check loop. If the draft comes back low confidence, route back to retrieval with a refined query instead of just escalating
- Expand the KB and ticket set for a bigger, more realistic demo
- Swap Chroma for pgvector to reuse the infra from my compliance RAG project