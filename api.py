"""Minimal FastAPI endpoint for the policy chatbot.

Run:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs to try POST /ask.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragbot.clients import openai_client, search_client
from ragbot.config import get_settings
from ragbot.rag import answer_question

app = FastAPI(title="Restaurant Policy RAG Chatbot", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What temperature must the cool room be?"])


class SourceOut(BaseModel):
    number: int
    source: str
    title: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@lru_cache
def _clients():
    s = get_settings()
    return s, openai_client(s), search_client(s)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        s, oai, search = _clients()
        result = answer_question(req.question, s, oai, search)
    except RuntimeError as exc:  # missing configuration
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(
        answer=result.answer,
        sources=[SourceOut(number=x.number, source=x.source, title=x.title) for x in result.sources],
    )
