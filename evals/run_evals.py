"""Score the cache on labelled question pairs at several thresholds. Run: make eval"""
import json
from pathlib import Path

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def rates(scored, threshold):
    """scored is a list of (truth, score). Return (savings, wrong_answers) as shares from 0 to 1."""
    same = [score >= threshold for truth, score in scored if truth == "same"]
    different = [score >= threshold for truth, score in scored if truth == "different"]
    return sum(same) / len(same), sum(different) / len(different)


def score_pairs(pairs, check_names):
    from cache import SemanticCache

    scored = []
    for pair in pairs:
        cache = SemanticCache(check_names=check_names)
        cache.put(pair["saved"], "saved answer")
        score, _, _ = cache.closest(pair["asked"])  # 0.0 when the name check blocks the pair
        scored.append((pair["truth"], score))
    return scored


if __name__ == "__main__":
    with open(Path(__file__).parent / "pairs.jsonl") as f:
        pairs = [json.loads(line) for line in f]
    without, with_check = score_pairs(pairs, check_names=False), score_pairs(pairs, check_names=True)

    print("| Threshold | Savings | Wrong answers | Savings (name check) | Wrong answers (name check) |")
    print("|---|---|---|---|---|")
    for threshold in THRESHOLDS:
        cells = [*rates(without, threshold), *rates(with_check, threshold)]
        print(f"| {threshold:.2f} | " + " | ".join(f"{value:.0%}" for value in cells) + " |")

    print("\nWrong answers still served at 0.90 with the name check on:")
    for pair, (truth, score) in zip(pairs, with_check, strict=True):
        if truth == "different" and score >= 0.90:
            print(f"  {score:.3f}  [{pair['kind']}]  {pair['saved']!r} -> {pair['asked']!r}")
