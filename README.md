# llm-semantic-cache

[![tests](https://github.com/kirubakarankan/llm-semantic-cache/actions/workflows/tests.yml/badge.svg)](https://github.com/kirubakarankan/llm-semantic-cache/actions/workflows/tests.yml)

A cache for an SRE assistant that answers reworded questions without calling the model. On 60 labelled question pairs it saves **50% of model calls** while serving a wrong answer for **3%** of different questions. A cached answer comes back in about **14 ms** instead of **2.3 s**.

## Demo

```text
$ python demo.py semantic
question> Why is etcdNoLeader firing?
     closest saved question: None  score 0.000  (hit needs 0.9)
MISS  2.270s  The etcdNoLeader alert fires when ...

question> What causes the etcdNoLeader alert?
     closest saved question: 'Why is etcdNoLeader firing?'  score 0.911  (hit needs 0.9)
HIT   0.014s  The etcdNoLeader alert fires when ...
```

## Why it matters

During an incident, ten engineers get paged for the same alert and ask the assistant the same thing in ten different ways. Without a cache, each question pays for a model call and waits seconds for an answer. With it, the first question pays and the other nine are answered in milliseconds.

A wrong answer is worse than a slow one. An engineer who is told how to fix the wrong alert loses time during an outage. So the cache is tuned to miss when it is unsure, and it never shares an answer between two different alerts.

## How it works

```text
question ──► exact match? ──yes──► cached answer
                │ no
                ▼
        meaning match ≥ 0.9?          ── also checks:
        same alert name?                 not expired,
                │                        runbook unchanged
        yes ◄───┴───► no
         │             │
   cached answer    run the pipeline ──► save answer + checksums of the runbooks it used
```

- `ExactCache` matches identical text.
- `SemanticCache` turns each question into an embedding (`bge-small`, local) and compares it with the saved ones. A score of 1.0 means the same meaning.
- **Name check:** questions that name different alerts (for example `etcdNoLeader` and `etcdMembersDown`) never share an answer.
- **Expiry:** answers older than `max_age` seconds are dropped.
- **Source check:** each answer stores a checksum of the runbook files it used. If a runbook changes or disappears, its answers are dropped.
- `ask(question, cache, answer_fn)` wraps any pipeline. On a hit it returns early, so the model is never called.

## Results

Report card on 60 labelled pairs (`make eval`). *Savings* is the share of same-meaning pairs that hit; *wrong answers* is the share of different-meaning pairs that hit.

| Threshold | Savings | Wrong answers | Wrong answers (name check) |
|---|---|---|---|
| 0.80 | 67% | 63% | 30% |
| 0.85 | 60% | 43% | 20% |
| **0.90** | **50%** | 13% | **3%** |
| 0.95 | 43% | 7% | 3% |

Replay of 20 questions traced in Langfuse (`replay.py`, local `llama3.2`):

| | Hit (7) | Miss (13) |
|---|---|---|
| Median time | 14 ms | 2.27 s |
| Tokens | 0 | about 117 |
| Steps in the trace | 1 (cache only) | 2 (cache + model call) |

Cost per query is 0 USD here because the model runs locally; with a hosted model every hit saves the full price of a call.

## Quickstart

```bash
make setup
make test
make eval
```

`demo.py` and `replay.py` also need [Ollama](https://ollama.com) running with `llama3.2`. `replay.py` needs Langfuse keys: copy `.env.example` to `.env` and fill it in.

## Design decisions

- **The cache sits in front of the whole pipeline, on the question only.** Measured on real prompts from my RAG project: the shared instructions made unrelated prompts score about 0.73 alike, and long prompts were cut off at the embedding model's 512-token limit. Caching the question also skips the search step, not only the model.
- **Threshold 0.90.** Lower thresholds save more but serve far more wrong answers (see the table). For ops advice, a miss is cheaper than a wrong answer.
- **An in-memory list, not a vector database.** Every comparison is visible and easy to test. It is lost on restart and slows down past a few thousand entries; that is when to move to a vector store.

## Tests and evals

- 21 unit tests (`make test`): exact and semantic hits, the name check, expiry with a fake clock, and the source check.
- `make eval`: the report card above.
- CI runs `ruff` and the tests on every push.

## Limitations and next steps

- One wrong answer remains at 0.90: "how do I diagnose X?" matches "how do I fix X?" for the same alert.
- Heavily reworded questions miss (all 3 hard pairs in the replay scored 0.73–0.83).
- The 60 pairs and 20 replay questions are small; a larger set from real incident chat would give firmer numbers.
- `replay.py` runs locally only: CI has no model and no Langfuse keys.
- Next: connect the cache to the real RAG pipeline from `sre-hybrid-rag`.

## Where it fits in SRE Copilot

This is project 2 of six. The cache sits at the front door of the copilot: every question passes through it before triage, search or the agent run.

```text
question ─► [cache] ─► [triage] ─► [agent] ─► [approval] ─► answer
```

## Credits

Designs I read and learned from:

- [zilliztech/GPTCache](https://github.com/zilliztech/GPTCache): the standard embed, search, score, return-or-call design
- [redis/redis-vl-python](https://github.com/redis/redis-vl-python): a readable semantic cache with a threshold and expiry
- [BerriAI/litellm](https://github.com/BerriAI/litellm): how a production gateway caches
