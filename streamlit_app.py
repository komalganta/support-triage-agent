import streamlit as st

from src.graph import build_graph

st.set_page_config(page_title="Support Ticket Triage Agent", page_icon="🎫")

st.title("🎫 Support Ticket Triage Agent")
st.caption("LangGraph agent: classifies, retrieves relevant docs, drafts a reply, and routes to auto-send or human escalation.")


@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

ticket_text = st.text_area(
    "Paste a customer support ticket:",
    placeholder="e.g. My payment failed twice and I was charged both times, please help urgently",
    height=100,
)

if st.button("Run Triage Agent", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a ticket first.")
    else:
        with st.spinner("Classifying, retrieving context, and drafting a reply..."):
            result = graph.invoke({"ticket_text": ticket_text})

        st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Category", result["classification"].category)
    col2.metric("Urgency", result["classification"].urgency)
    col3.metric(
        "Routing",
        "✅ Auto-send" if result["routing_decision"] == "auto_send" else "🚨 Escalate",
    )

    st.caption(f"Reasoning: {result['classification'].reasoning}")
    st.caption(f"Routing reason: {result['routing_reason']}")

    st.subheader("Retrieved KB Articles")
    for article in result["retrieved_articles"]:
        st.write(f"**{article.title}** — similarity: {article.similarity_score:.2f}")

    st.subheader("Draft Reply")
    st.info(result["draft"].reply_text)
    st.caption(f"Draft confidence: {result['draft'].confidence:.2f}")