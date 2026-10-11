"""
Green-Zone - speculative decoding teardown.

The draft model is the green_zone pig: cheap, fast, sniffs out candidate tokens.
The target model is the chef: expensive, verifies, accepts only the good ones.
Accepted-token rate is the ratio that decides whether the whole scheme pays.

Measures against the LIVE cluster (read-only; /metrics needs no auth):
- acceptance rate (drafted vs accepted tokens) via cumulative counter deltas
- per-position acceptance decay (pos 0..K) - why long draft chains stall out
- concurrency response: does acceptance degrade under batch pressure?
- cost per ACCEPTED token: accepted tokens are the only ones that count -
  rejected draft work is compute you paid for and threw away
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
KEY = os.environ.get("VLLM_API_KEY")
MODEL = os.environ.get("VLLM_MODEL", "glm-5.3-flash")

PROMPTS = [
    "Summarize the trade-offs of speculative decoding in two sentences.",
    "Write one paragraph explaining KV cache pressure to an SRE.",
    "In one sentence: why does draft length have a sweet spot?",
    "Explain prefix caching to a developer who has never run an inference server.",
    "Two sentences on why MoE throughput is bandwidth-bound.",
    "Write a JSON object with keys draft, accepted, rate describing speculation.",
    "What does per-position acceptance decay tell you about draft quality?",
    "One sentence: when is speculative decoding NOT worth it?",
]


def get_spec_counters() -> dict:
    """Scrape spec-decode counters. Returns drafted/accepted/per-position."""
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        text = r.read().decode()
    counters = {"drafted": 0.0, "accepted": 0.0, "drafts": 0.0, "per_pos": {}}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            counters["drafted"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            counters["accepted"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            counters["drafts"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            pos = int(line.split('position="')[1].split('"')[0])
            counters["per_pos"][pos] = float(line.split()[-1])
    return counters


def chat(prompt: str, max_tokens: int, timeout: float = 180) -> dict:
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.3}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {KEY}"} if KEY else {})})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
        usage = body.get("usage", {})
        return {"ok": True, "wall_s": time.perf_counter() - t0,
                "completion_tokens": usage.get("completion_tokens", 0)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "wall_s": time.perf_counter() - t0}


def measure(concurrency: int, requests: int, max_tokens: int) -> dict:
    """One measurement window: counter delta across N requests at concurrency c."""
    before = get_spec_counters()
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda i: chat(PROMPTS[i % len(PROMPTS)], max_tokens),
                              range(requests)))
    wall = time.perf_counter() - t0
    after = get_spec_counters()

    ok = [r for r in results if r["ok"]]
    drafted = after["drafted"] - before["drafted"]
    accepted = after["accepted"] - before["accepted"]
    ctok = sum(r["completion_tokens"] for r in ok)

    return {
        "concurrency": concurrency,
        "requests": requests, "ok": len(ok), "failed": len(results) - len(ok),
        "wall_s": round(wall, 2),
        "completion_tokens": ctok,
        "agg_tok_s": round(ctok / wall, 1) if wall > 0 else 0,
        "drafted_delta": drafted,
        "accepted_delta": accepted,
        "acceptance_rate": round(accepted / drafted, 4) if drafted > 0 else None,
        "accept_ratio_per_draft": round(accepted / (after["drafts"] - before["drafts"]), 3)
                                  if after["drafts"] > before["drafts"] else None,
        "per_pos_decay": {p: round((after["per_pos"].get(p, 0) - before["per_pos"].get(p, 0)) /
                                   (after["per_pos"][0] - before["per_pos"].get(0, 1e-9) or 1), 3)
                          for p in sorted(after["per_pos"]) if p > 0}
                  if drafted > 0 else {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--out", default="benchmarks")
    args = ap.parse_args()

    print(f"Green-Zone: measuring speculative decoding acceptance on {MODEL}")
    print("(draft=pig, target=chef - read-only against the live cluster)\n")

    run = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "windows": []}
    for c in (1, 2, 4):
        w = measure(concurrency=c, requests=args.requests, max_tokens=args.max_tokens)
        run["windows"].append(w)
        print(f"c={c}: {w['ok']}/{w['requests']} ok, {w['agg_tok_s']} tok/s, "
              f"acceptance {w['acceptance_rate']}, "
              f"accepted/draft-cycle {w['accept_ratio_per_draft']}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    path = out / f"green_zone_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(run, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
