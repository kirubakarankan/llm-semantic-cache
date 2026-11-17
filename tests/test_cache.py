from cache import ExactCache, SemanticCache, ask, names


def answers_in_turn(*texts, sources=()):
    """A stand-in for the real pipeline: gives the next text each time it is asked."""
    remaining = iter(texts)
    return lambda question: (next(remaining), list(sources))


def test_first_time_a_question_is_asked_the_pipeline_answers_it():
    answer_fn = answers_in_turn("answer one")
    assert ask("what is etcdNoLeader?", ExactCache(), answer_fn) == ("answer one", False)


def test_same_question_again_comes_from_the_cache_without_running_the_pipeline():
    answer_fn = answers_in_turn("answer one", "answer two")
    cache = ExactCache()
    ask("what is etcdNoLeader?", cache, answer_fn)
    assert ask("what is etcdNoLeader?", cache, answer_fn) == ("answer one", True)


def test_reworded_question_is_a_miss_for_an_exact_cache():
    answer_fn = answers_in_turn("answer one", "answer two")
    cache = ExactCache()
    ask("what is etcdNoLeader?", cache, answer_fn)
    assert ask("explain etcdNoLeader", cache, answer_fn) == ("answer two", False)


SAVED = "what does the etcdNoLeader alert mean?"


def test_semantic_cache_hits_on_a_reworded_question():
    cache = SemanticCache()
    cache.put(SAVED, "no leader for over a minute")
    assert cache.get("explain the etcdNoLeader alert") == "no leader for over a minute"


def test_semantic_cache_misses_on_an_unrelated_question():
    cache = SemanticCache()
    cache.put(SAVED, "no leader for over a minute")
    assert cache.get("how do I rotate my AWS access keys?") is None


def test_semantic_cache_misses_when_it_is_empty():
    assert SemanticCache().get(SAVED) is None


def test_closest_reports_the_score_and_which_saved_question_matched():
    cache = SemanticCache()
    cache.put(SAVED, "no leader for over a minute")
    cache.put("how do I rotate my AWS access keys?", "use the IAM console")
    score, question, answer = cache.closest("explain the etcdNoLeader alert")
    assert (question, answer) == (SAVED, "no leader for over a minute")
    assert 0.9 < score <= 1.0


def test_names_finds_alert_names_and_ignores_ordinary_words():
    assert names("How do I fix KubePodCrashLooping?") == {"kubepodcrashlooping"}
    assert names("my pod keeps restarting") == set()


def test_different_alert_names_never_share_an_answer_even_with_a_low_threshold():
    cache = SemanticCache(threshold=0.75)
    cache.put(SAVED, "no leader for over a minute")
    assert cache.get("what does the etcdMembersDown alert mean?") is None


def test_question_without_an_alert_name_can_still_hit():
    cache = SemanticCache(threshold=0.75)
    cache.put(SAVED, "no leader for over a minute")
    assert cache.get("etcd has no leader, what is that?") == "no leader for over a minute"


def test_lookup_skips_a_closer_question_about_another_alert():
    cache = SemanticCache(threshold=0.75)
    cache.put(SAVED, "no leader for over a minute")
    cache.put("explain etcdMembersDown", "a member is down")
    assert cache.get("what does the etcdMembersDown alert mean?") == "a member is down"


def test_name_check_can_be_switched_off_to_measure_what_it_is_worth():
    cache = SemanticCache(threshold=0.75, check_names=False)
    cache.put(SAVED, "no leader for over a minute")
    assert cache.get("what does the etcdMembersDown alert mean?") == "no leader for over a minute"


def test_answer_is_reused_before_it_expires_and_not_after():
    now = [1000.0]
    cache = SemanticCache(max_age=10, clock=lambda: now[0])
    cache.put(SAVED, "no leader for over a minute")
    now[0] += 3
    assert cache.get(SAVED) == "no leader for over a minute"
    now[0] += 12
    assert cache.get(SAVED) is None


def test_expired_answers_are_removed_from_the_notebook():
    now = [1000.0]
    cache = SemanticCache(max_age=10, clock=lambda: now[0])
    cache.put(SAVED, "no leader for over a minute")
    now[0] += 11
    cache.get(SAVED)
    assert cache.entries == []


def test_answer_is_reused_while_its_runbook_is_unchanged(tmp_path):
    runbook = tmp_path / "etcdNoLeader.md"
    runbook.write_text("Restart the etcd pods.")
    cache = SemanticCache()
    cache.put(SAVED, "restart the pods", sources=[runbook])
    assert cache.get(SAVED) == "restart the pods"


def test_answer_is_dropped_when_its_runbook_is_edited(tmp_path):
    runbook = tmp_path / "etcdNoLeader.md"
    runbook.write_text("Restart the etcd pods.")
    cache = SemanticCache()
    cache.put(SAVED, "restart the pods", sources=[runbook])
    runbook.write_text("Restore from the latest snapshot.")
    assert cache.get(SAVED) is None


def test_answer_is_dropped_when_its_runbook_is_deleted(tmp_path):
    runbook = tmp_path / "etcdNoLeader.md"
    runbook.write_text("Restart the etcd pods.")
    cache = SemanticCache()
    cache.put(SAVED, "restart the pods", sources=[runbook])
    runbook.unlink()
    assert cache.get(SAVED) is None


def test_ask_gives_a_fresh_answer_after_the_runbook_changes(tmp_path):
    runbook = tmp_path / "etcdNoLeader.md"
    runbook.write_text("Restart the etcd pods.")
    answer_fn = answers_in_turn("restart the pods", "restore the snapshot", sources=[runbook])
    cache = SemanticCache()
    ask(SAVED, cache, answer_fn)
    runbook.write_text("Restore from the latest snapshot.")
    assert ask(SAVED, cache, answer_fn) == ("restore the snapshot", False)
