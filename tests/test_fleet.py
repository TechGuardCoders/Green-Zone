"""Fleet tests: node resolution + metrics scraping against a mocked HTTP endpoint."""
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import green_zone  # noqa: E402
from green_zone import FLEET_NODES, get_spec_counters, resolve_base, snapshot_node  # noqa: E402


METRICS = """# HELP spec counters
vllm:spec_decode_num_draft_tokens_total 1000.0
vllm:spec_decode_num_accepted_tokens_total 400.0
vllm:spec_decode_num_drafts_total 200.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 200.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 150.0
"""


class FakeResponse:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_resolve_base_default_unchanged():
    """No node -> the original single-node BASE (backward compat)."""
    assert resolve_base(None) == green_zone.BASE


def test_resolve_base_known_nodes():
    for name, url in FLEET_NODES.items():
        assert resolve_base(name) == url


def test_resolve_base_unknown_node_exits():
    with pytest.raises(SystemExit):
        resolve_base("nope")


def test_get_spec_counters_mocked_http(monkeypatch):
    seen = {}

    def fake_urlopen(url, timeout=10):
        seen["url"] = url
        return FakeResponse(METRICS)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    c = get_spec_counters("http://node-a:8421")
    assert seen["url"] == "http://node-a:8421/metrics", "only /metrics is ever hit"
    assert c["drafted"] == 1000.0 and c["accepted"] == 400.0
    assert c["per_pos"] == {0: 200.0, 1: 150.0}


def test_snapshot_node_read_only(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=10: FakeResponse(METRICS))
    snap = snapshot_node("tensorfold")
    assert snap["node"] == "tensorfold"
    assert snap["base"] == FLEET_NODES["tensorfold"]
    assert snap["cumulative_acceptance_rate"] == 0.4
    assert snap["per_pos_positions"] == [0, 1]


def test_snapshot_node_idle_cluster_rate_none(monkeypatch):
    idle = "vllm:spec_decode_num_draft_tokens_total 0.0\n"
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=10: FakeResponse(idle))
    snap = snapshot_node("sushi")
    assert snap["cumulative_acceptance_rate"] is None
