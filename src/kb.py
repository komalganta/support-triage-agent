import json
import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from src.schemas import RetrievedArticle

load_dotenv()

def load_kb_articles(path: str = "data/kb_articles.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
)

def build_vectorstore(articles: list[dict]) -> Chroma:
    documents = [
        Document(
            page_content=article["content"],
            metadata={"title": article["title"], "category": article["category"]},
        )
        for article in articles
    ]

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vectorstore() -> Chroma:
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )


def get_vectorstore(articles: list[dict]) -> Chroma:
    """Load the existing store if one exists on disk; otherwise build it fresh."""
    if os.path.exists("chroma_db"):
        return load_vectorstore()
    return build_vectorstore(articles)

def retrieve(vectorstore: Chroma, query: str, k: int = 3) -> list[RetrievedArticle]:
    results = vectorstore.similarity_search_with_score(query, k=k)

    return [
        RetrievedArticle(
            title=doc.metadata["title"],
            content=doc.page_content,
            similarity_score=1 - distance,
        )
        for doc, distance in results
    ]