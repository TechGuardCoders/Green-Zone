# Green-Zone

**Speculative decoding teardown — the draft model is the green_zone pig (cheap, fast, sniffs candidates); the target model is the chef (expensive, verifies, accepts only the good ones).**

Measured against the live [spark-glm-cluster](https://github.com/TechGuardCoders/spark-glm-cluster) (GLM-5.3-Flash EXL3, DFlash2 speculation), entirely read-only: acceptance counters from `/metrics` deltas, no cluster mutation.

Part of the [TechGuardCoders portfolio](https://github.com/orgs/TechGuardCoders/repositories): Cost Peep → Batcher → Spinal → Ball Knowledge → Bastion → Furnace → **Green-Zone** → …

## Measured (live, 48 requests, 0 failures)

| Concurrency | Aggregate tok/s | Acceptance rate | Accepted/draft-cycle |
|---|---|---|---|
| c=1 | 20.6 | **22.5%** | 1.58 |
| c=2 | 30.2 | **21.0%** | 1.47 |
| c=4 | 42.7 | **22.3%** | 1.56 |

**Per-position acceptance decay** (each position's acceptance, normalized to position 0):

```
pos:    0      1      2      3      4      5      6
c=1:  1.00   .597   .333   .204   .133   .082   .035
c=2:  1.00   .552   .309   .183   .090   .057   .041
c=4:  1.00   .608   .324   .185   .102   .057   .028
```

Textbook geometric decay: each draft position is accepted ~60% as often as the one before. By position 4, drafts are mostly noise.

## What the numbers mean

1. **~1.56 accepted tokens per draft cycle** — each draft round costs one target-model verification pass but yields ~1.5 tokens of output. Without speculation the ratio is exactly 1.0; this is the spec-decode dividend.
2. **Acceptance is concurrency-INSENSITIVE (~21-22%)** — acceptance is a property of draft/target agreement, not load. This is why Batcher's knee at c=2 exists: spec overhead is fixed, so its *benefit* shrinks in relative terms as bandwidth-bound MoE batching takes over. Green-Zone explains Batcher's knee from the inside.
3. **Draft length has a hard sweet spot.** With geometric decay ~0.6, a draft chain of K has expected accepted = 1/(1−0.6·(K−1)/K)-ish — drafting beyond position 3-4 is wasted pig-work that still costs draft-model compute and verification latency.

## Cost per accepted token

Rejected draft tokens are compute you paid for and threw away. With 78% rejection, draft-model efficiency matters more than raw speed — a faster pig that guesses worse can LOSE money. The cost-per-accepted-token KPI this produces feeds Cost Peep directly.

## Run it

```bash
VLLM_API_KEY=<key> python green_zone.py --requests 16 --max-tokens 100
# read-only: chat completions + /metrics scrapes, nothing else
```

## Tests

3 offline tests (decay math, zero-draft guard, module contract). CI-safe.

## Fleet: benchmark any node

`--node NAME` targets a named fleet node; with `--metrics-only` it scrapes only that node's `/metrics` (read-only, no `/v1` traffic) and saves a snapshot. Nodes: `glm-head`, `tensorfold`, `bonsai-1`, `bonsai-2`, `sushi`. No `--node` keeps the original single-node behavior.

```bash
python green_zone.py --node tensorfold --metrics-only
```
