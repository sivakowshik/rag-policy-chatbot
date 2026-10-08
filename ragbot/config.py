"""Load settings from environment variables (or a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value or value.startswith("<"):
        raise RuntimeError(
            f"Missing setting {name}. Copy .env.example to .env and fill it in."
        )
    return value


@dataclass(frozen=True)
class Settings:
    openai_endpoint: str
    openai_api_key: str
    openai_api_version: str
    chat_deployment: str
    embedding_deployment: str
    embedding_dimensions: int
    search_endpoint: str
    search_admin_key: str
    search_index: str
    top_k: int
    chunk_size: int
    chunk_overlap: int


def get_settings() -> Settings:
    return Settings(
        openai_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=_require("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        chat_deployment=_require("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        embedding_deployment=_require("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
        search_endpoint=_require("AZURE_SEARCH_ENDPOINT"),
        search_admin_key=_require("AZURE_SEARCH_ADMIN_KEY"),
        search_index=os.getenv("AZURE_SEARCH_INDEX", "restaurant-policies"),
        top_k=int(os.getenv("TOP_K", "5")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
    )
