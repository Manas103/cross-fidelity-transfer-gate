import numpy as np
from src.sequential import sequential_readout


def test_identical_arms_never_reach_significance():
    rng = np.random.default_rng(0)
    a = rng.random(300) > 0.5
    b = a.copy()  # identical outcomes on both arms
    result = sequential_readout(a, b, checkpoint_every=50)
    assert result["first_significant_n"] is None


def test_a_strong_gap_is_detected_and_stays_significant():
    a = np.array([True] * 300)
    b = np.array([False] * 300)
    result = sequential_readout(a, b, checkpoint_every=50)
    assert result["first_significant_n"] == 50
    assert result["stayed_significant_after_first"]


def test_checkpoint_count_matches_schedule():
    a = np.array([True, False] * 150)
    b = np.array([False, True] * 150)
    result = sequential_readout(a, b, checkpoint_every=100)
    assert [c["n"] for c in result["checkpoints"]] == [100, 200, 300]
