"""A notebook of questions already answered, checked before asking the model."""
import re
import time
from functools import cache

from sentence_transformers import SentenceTransformer


@cache  # loading the model takes seconds, so do it once and only when it is needed
def model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5", revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")


def fingerprint(text):
    # Scaled to length 1, so multiplying two fingerprints gives a score: 1 = same meaning, lower = less alike.
    return model().encode(text, normalize_embeddings=True)


def names(text):
    """Alert names in the text: words with a capital in the middle, like etcdNoLeader."""
    return {name.lower() for name in re.findall(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b", text)}


def about_different_alerts(question, saved_question):
    # A name is exact: etcdNoLeader and etcdMembersDown look alike but need different answers.
    asked, saved = names(question), names(saved_question)
    return bool(asked and saved and asked != saved)


class ExactCache:
    """Remembers answers by the exact text of the question."""

    def __init__(self):
        self.answers = {}

    def get(self, question):
        return self.answers.get(question)

    def put(self, question, answer):
        self.answers[question] = answer


class SemanticCache:
    """Remembers answers by the meaning of the question."""

    # shortcut: kept in memory and compared one by one, so it is lost on restart and slows
    # down past a few thousand entries. Move to a vector store when either one matters.
    def __init__(self, threshold=0.9, check_names=True, max_age=None, clock=time.time):
        self.threshold = threshold
        self.check_names = check_names  # switched off only to measure what the check is worth
        self.max_age = max_age  # seconds an answer stays usable; None means it never expires
        self.clock = clock  # tests pass a fake clock so they do not have to wait
        self.entries = []  # one (fingerprint, question, answer, saved_at) per saved question

    def closest(self, question):
        """Return (score, saved_question, saved_answer) for the most similar saved question.

        Saved questions that name a different alert are left out.
        """
        if self.max_age is not None:
            oldest_allowed = self.clock() - self.max_age
            self.entries = [entry for entry in self.entries if entry[3] >= oldest_allowed]
        asked = fingerprint(question)
        scored = [
            (float(saved @ asked), q, a)
            for saved, q, a, _ in self.entries
            if not (self.check_names and about_different_alerts(question, q))
        ]
        return max(scored, default=(0.0, None, None))

    def get(self, question):
        score, _, answer = self.closest(question)
        return answer if score >= self.threshold else None

    def put(self, question, answer):
        self.entries.append((fingerprint(question), question, answer, self.clock()))


def ask(question, cache, llm):
    """Return (answer, came_from_cache)."""
    answer = cache.get(question)
    if answer is not None:
        return answer, True
    answer = llm.invoke(question).content
    cache.put(question, answer)
    return answer, False
