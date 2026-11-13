from evals.run_evals import rates

# (truth, score) for four pairs
SCORED = [("same", 0.95), ("same", 0.80), ("different", 0.92), ("different", 0.60)]


def test_savings_is_the_share_of_same_pairs_that_hit():
    savings, _ = rates(SCORED, threshold=0.9)
    assert savings == 0.5


def test_wrong_answers_is_the_share_of_different_pairs_that_hit():
    _, wrong = rates(SCORED, threshold=0.9)
    assert wrong == 0.5


def test_a_higher_threshold_lowers_both():
    assert rates(SCORED, threshold=0.99) == (0.0, 0.0)
