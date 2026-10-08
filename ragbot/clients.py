"""Factory functions for the Azure OpenAI and Azure AI Search clients."""
from __future__ import annotations

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import AzureOpenAI

from .config import Settings


def openai_client(s: Settings) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=s.openai_endpoint,
        api_key=s.openai_api_key,
        api_version=s.openai_api_version,
    )


def search_client(s: Settings) -> SearchClient:
    return SearchClient(s.search_endpoint, s.search_index, AzureKeyCredential(s.search_admin_key))


def index_client(s: Settings) -> SearchIndexClient:
    return SearchIndexClient(s.search_endpoint, AzureKeyCredential(s.search_admin_key))


def embed(client: AzureOpenAI, deployment: str, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the Azure OpenAI embedding deployment."""
    response = client.embeddings.create(model=deployment, input=texts)
    return [item.embedding for item in response.data]
