"""Type questions and see which ones the cache answers. Needs Ollama running. Ctrl-C to quit.

Usage: python demo.py            exact cache
       python demo.py semantic   semantic cache
       python demo.py semantic 0.8      ... with a threshold of 0.8
       python demo.py semantic 0.9 10   ... and answers that expire after 10 seconds
"""
import sys
import time

from langchain_ollama import ChatOllama

from cache import ExactCache, SemanticCache, ask

llm = ChatOllama(model="llama3.2", temperature=0, num_predict=80)
semantic = len(sys.argv) > 1 and sys.argv[1] == "semantic"
threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
max_age = float(sys.argv[3]) if len(sys.argv) > 3 else None
cache = SemanticCache(threshold, max_age=max_age) if semantic else ExactCache()

while True:
    question = input("\nquestion> ")
    if semantic:
        score, saved, _ = cache.closest(question)
        print(f"     closest saved question: {saved!r}  score {score:.3f}  (hit needs {cache.threshold})")
    start = time.perf_counter()
    answer, hit = ask(question, cache, llm)
    seconds = time.perf_counter() - start
    print(f"{'HIT ' if hit else 'MISS'} {seconds:6.3f}s  {answer[:150]}")
