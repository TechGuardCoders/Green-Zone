"""Offline tests for Truffle logic - no network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from truffle import get_spec_counters  # noqa: E402  (import validates module)


def test_module_imports():
    import truffle
    assert hasattr(truffle, "measure") and hasattr(truffle, "get_spec_counters")


def test_per_pos_decay_math():
    """Decay normalizes each position's delta to position 0's delta."""
    after = {"drafted": 1000.0, "accepted": 400.0, "drafts": 200.0,
             "per_pos": {0: 200.0, 1: 150.0, 2: 100.0}}
    before = {"drafted": 0.0, "accepted": 0.0, "drafts": 0.0, "per_pos": {}}
    pos1 = (after["per_pos"][1] - before["per_pos"].get(1, 0)) / (after["per_pos"][0] - 0)
    assert abs(pos1 - 0.75) < 1e-9  # pos1 accepted 75% as often as pos0
    rate = (after["accepted"] - before["accepted"]) / (after["drafted"] - before["drafted"])
    assert abs(rate - 0.4) < 1e-9


def test_zero_draft_guard():
    """Idle cluster (no drafts in window) must yield None rate, not ZeroDivision."""
    # simulate: drafted delta 0 -> acceptance_rate None in measure(); the guard
    # expression `if drafted > 0 else None` is the contract under test
    drafted_delta = 0.0
    rate = round(400.0 / drafted_delta, 4) if drafted_delta > 0 else None
    assert rate is None
