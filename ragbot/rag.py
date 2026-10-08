"""Retrieve relevant policy chunks (hybrid search) and answer with citations."""
from __future__ import annotations

from dataclasses import dataclass, field

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

from .clients import embed
from .config import Settings
from .index import VECTOR_FIELD

SYSTEM_PROMPT = """You are a helpful assistant for restaurant crew and managers.
Answer ONLY using the numbered policy excerpts provided in the user message.
Cite every fact with its excerpt number in square brackets, e.g. [1] or [2][3].
If the excerpts do not contain the answer, say you could not find it in the
policies and suggest the person asks their shift manager. Keep answers short
and practical."""


@dataclass
class Source:
    number: int
    source: str
    title: str
    content: str
    score: float | None = None


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)


def retrieve(question: str, s: Settings, oai: AzureOpenAI, search: SearchClient) -> list[Source]:
    """Hybrid retrieval: keyword (BM25) + vector similarity, fused by Azure AI Search."""
    vector = embed(oai, s.embedding_deployment, [question])[0]
    vector_query = VectorizedQuery(vector=vector, k_nearest_neighbors=s.top_k, fields=VECTOR_FIELD)
    results = search.search(
        search_text=question,
        vector_queries=[vector_query],
        select=["source", "title", "content"],
        top=s.top_k,
    )
    sources: list[Source] = []
    for i, doc in enumerate(results, start=1):
        sources.append(
            Source(
                number=i,
                source=doc["source"],
                title=doc["title"],
                content=doc["content"],
                score=doc.get("@search.score"),
            )
        )
    return sources


def build_context(sources: list[Source]) -> str:
    return "\n\n".join(f"[{src.number}] ({src.source} - {src.title})\n{src.content}" for src in sources)


def answer_question(question: str, s: Settings, oai: AzureOpenAI, search: SearchClient) -> Answer:
    sources = retrieve(question, s, oai, search)
    if not sources:
        return Answer(question, "I couldn't find anything about that in the policies.", [])
    response = oai.chat.completions.create(
        model=s.chat_deployment,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Policy excerpts:\n{build_context(sources)}\n\nQuestion: {question}",
            },
        ],
    )
    text = response.choices[0].message.content or ""
    return Answer(question=question, answer=text.strip(), sources=sources)
