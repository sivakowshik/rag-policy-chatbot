"""Ingest the policy documents: chunk -> embed -> upload to Azure AI Search.

Usage:
    python ingest.py                      # uses data/sample_sops
    python ingest.py --docs path/to/docs  # any folder of .md files
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ragbot.chunking import load_and_chunk
from ragbot.clients import embed, index_client, openai_client, search_client
from ragbot.config import get_settings
from ragbot.index import VECTOR_FIELD, ensure_index

BATCH_SIZE = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="data/sample_sops", help="Folder of markdown files")
    args = parser.parse_args()

    s = get_settings()
    chunks = load_and_chunk(Path(args.docs), s.chunk_size, s.chunk_overlap)
    if not chunks:
        raise SystemExit(f"No .md files found in {args.docs}")
    print(f"Chunked {len(chunks)} pieces from {args.docs}")

    ensure_index(index_client(s), s.search_index, s.embedding_dimensions)
    print(f"Index '{s.search_index}' is ready")

    oai = openai_client(s)
    search = search_client(s)
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        vectors = embed(oai, s.embedding_deployment, [c.content for c in batch])
        docs = [
            {
                "id": c.id,
                "source": c.source,
                "title": c.title,
                "content": c.content,
                "chunk_index": c.chunk_index,
                VECTOR_FIELD: v,
            }
            for c, v in zip(batch, vectors)
        ]
        results = search.upload_documents(documents=docs)
        ok = sum(1 for r in results if r.succeeded)
        print(f"Uploaded {ok}/{len(docs)} chunks")
    print("Done.")


if __name__ == "__main__":
    main()
