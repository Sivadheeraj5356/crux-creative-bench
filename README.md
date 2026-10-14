# creative-bench

A small, honest eval suite for ad-creative analysis agents - built as a work
sample for teams shipping AI creative strategists (like GetCrux).

The bet behind it: an agent that tags creatives, calls creative fatigue and
picks A/B winners should be measured by hidden graders on pinned tasks, not
by how its chat traces read. Scores are written by the runner only - an agent
cannot award itself points.

## What it measures

Three task families, each graded 0-1 on every trial:

| Task | What the agent sees | What the grader checks |
| --- | --- | --- |
| `tag_creative` | primary text, headline, CTA button, format | messaging angle, CTA presence, hook type (exact match, per-field partial credit) |
| `detect_fatigue` | 14 days of impressions / clicks / frequency | fatigue only when frequency >= 3 **and** CTR is down >= 30% from the first-3-day baseline, sustained 2 days. Healthy creatives with low-frequency CTR dips are traps. |
| `pick_winner` | two variants with impressions and clicks | two-proportion z-test at alpha = 0.05 (two-sided). `inconclusive` is a real answer, and raw-CTR guessing fails on it. |

The dataset is generated deterministically (`--seed`), so the benchmark is
reproducible byte-for-byte. Ground truth is never handed to the agent; the
fatigue and A/B graders recompute truth from raw inputs.

## Run it

No dependencies, Python 3.10+:

```bash
python creative_bench.py run --agent baseline   # heuristic reference bar
python creative_bench.py run --agent oracle     # grader consistency: must be 1.000
python creative_bench.py test                   # unittest suite
```

Reference numbers on the shipped seed (7):

- oracle: **1.000** (grader consistency)
- heuristic baseline: **~0.81** overall - tags easy creatives fine, but drops
  points exactly where naive agents should: it calls fatigue on CTR dips that
  come with low frequency, and it picks A/B winners without significance
  testing.

To plug in a real agent, pass any callable `(task_type, task_input) -> dict`
to `run()` - answers only, scores come back from the hidden graders.

## Why this shape

Creative tooling already has plenty of generation. The harder problem is
trusting the loop: is this creative actually fatigued, did that variant
actually win, did the agent tag the angle or just pattern-match the copy.
Small verifiable benches like this are how you catch regressions in the
analysis agents before customers do.
