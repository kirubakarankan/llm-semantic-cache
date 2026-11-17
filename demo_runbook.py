"""Answers come from one runbook file. Edit the file while this runs and ask again.

Needs Ollama running. Ctrl-C to quit.
"""
import time
from pathlib import Path

from langchain_ollama import ChatOllama

from cache import SemanticCache, ask

RUNBOOK = Path("runbooks/etcdNoLeader.md")
llm = ChatOllama(model="llama3.2", temperature=0, num_predict=60)
cache = SemanticCache()


def answer_from_runbook(question):
    prompt = f"Answer in one sentence using only this runbook.\n\n{RUNBOOK.read_text()}\nQuestion: {question}"
    return llm.invoke(prompt).content, [RUNBOOK]


while True:
    question = input("\nquestion> ")
    start = time.perf_counter()
    answer, hit = ask(question, cache, answer_from_runbook)
    seconds = time.perf_counter() - start
    print(f"{'HIT ' if hit else 'MISS'} {seconds:6.3f}s  {answer[:150]}")
