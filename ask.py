"""Ask the policy chatbot a question from the command line.

Usage:
    python ask.py "Who is allowed to close the store?"
    python ask.py            # interactive mode
"""
from __future__ import annotations

import sys

from ragbot.clients import openai_client, search_client
from ragbot.config import get_settings
from ragbot.rag import answer_question


def print_answer(question: str) -> None:
    s = get_settings()
    result = answer_question(question, s, openai_client(s), search_client(s))
    print(f"\n{result.answer}\n")
    if result.sources:
        print("Sources:")
        for src in result.sources:
            print(f"  [{src.number}] {src.source} - {src.title}")
    print()


def main() -> None:
    if len(sys.argv) > 1:
        print_answer(" ".join(sys.argv[1:]))
        return
    print("Restaurant policy chatbot. Type a question, or 'quit' to exit.")
    while True:
        question = input("> ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            print_answer(question)


if __name__ == "__main__":
    main()
