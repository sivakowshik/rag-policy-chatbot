"""Test the retrieve -> prompt -> answer flow with fake Azure clients (offline)."""
from types import SimpleNamespace

from ragbot.config import Settings
from ragbot.rag import answer_question

SETTINGS = Settings(
    openai_endpoint="https://example.openai.azure.com/",
    openai_api_key="fake",
    openai_api_version="2024-10-21",
    chat_deployment="chat",
    embedding_deployment="embed",
    embedding_dimensions=3,
    search_endpoint="https://example.search.windows.net",
    search_admin_key="fake",
    search_index="test",
    top_k=2,
    chunk_size=800,
    chunk_overlap=120,
)


class FakeOpenAI:
    def __init__(self):
        self.prompts = []
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    def _embed(self, model, input):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in input])

    def _chat(self, model, messages, temperature):
        self.prompts.append(messages)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Only close-trained staff can close [1]."))]
        )


class FakeSearch:
    def __init__(self, docs):
        self.docs = docs
        self.kwargs = None

    def search(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.docs)


def test_answer_includes_sources_and_numbered_context():
    oai = FakeOpenAI()
    search = FakeSearch(
        [{"source": "closing-procedure.md", "title": "Who can close", "content": "Only close-trained...", "@search.score": 0.9}]
    )
    result = answer_question("Who can close?", SETTINGS, oai, search)
    assert "[1]" in result.answer
    assert result.sources[0].source == "closing-procedure.md"
    assert search.kwargs["search_text"] == "Who can close?"  # hybrid: keyword + vector
    assert len(search.kwargs["vector_queries"]) == 1
    assert "[1] (closing-procedure.md - Who can close)" in oai.prompts[0][1]["content"]


def test_no_results_returns_fallback_without_calling_chat():
    oai = FakeOpenAI()
    result = answer_question("Anything?", SETTINGS, oai, FakeSearch([]))
    assert result.sources == []
    assert oai.prompts == []
