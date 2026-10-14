#!/usr/bin/env python3
"""creative_bench.py - a small, honest eval suite for ad-creative analysis agents.

Built as a work sample for GetCrux (AI creative strategist for paid media).
The bet: an agent that tags creatives, calls creative fatigue and picks A/B
winners should be measured by hidden graders on pinned tasks, not by chat.

Three task families, 0-1 reward per trial, written scores only:

  tag_creative    classify messaging angle / CTA presence / hook type
  detect_fatigue  call whether a creative is fatigued and the day it started,
                  under a fixed rule: frequency >= 3 and CTR down >= 30% from
                  the first-3-day baseline, sustained for 2 consecutive days
  pick_winner     pick the A/B winner by two-proportion z-test (alpha = 0.05,
                  two-sided); "inconclusive" is a real answer

No dependencies. Python 3.10+.

    python creative_bench.py run --agent baseline   # heuristic reference bar
    python creative_bench.py run --agent oracle     # grader consistency == 1.0
    python creative_bench.py test                   # unittest suite
"""
from __future__ import annotations

import argparse
import json
import math
import random
import unittest
from dataclasses import dataclass, field

Z_95_TWO_SIDED = 1.9599639851205416
FATIGUE_FREQ_THRESHOLD = 3.0
FATIGUE_CTR_DROP = 0.30
ANGLES = ["urgency", "social_proof", "feature", "offer", "fear", "curiosity"]


# ---------------------------------------------------------------- stats

def two_proportion_z(clicks_a: int, n_a: int, clicks_b: int, n_b: int) -> float:
    if min(n_a, n_b) <= 0:
        raise ValueError("impressions must be positive")
    if not (0 <= clicks_a <= n_a and 0 <= clicks_b <= n_b):
        raise ValueError("clicks must be within [0, impressions]")
    p_pool = (clicks_a + clicks_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0.0:
        return 0.0
    return (clicks_a / n_a - clicks_b / n_b) / se


def ab_verdict(clicks_a: int, n_a: int, clicks_b: int, n_b: int) -> str:
    z = two_proportion_z(clicks_a, n_a, clicks_b, n_b)
    if abs(z) < Z_95_TWO_SIDED:
        return "inconclusive"
    return "A" if z > 0 else "B"


# ---------------------------------------------------------------- dataset

def _ctr(day: dict) -> float:
    return day["clicks"] / day["impressions"] if day["impressions"] else 0.0


def fatigue_ground_truth(daily: list[dict]) -> tuple[bool, int | None]:
    baseline = sum(_ctr(d) for d in daily[:3]) / 3
    for i in range(3, len(daily) - 1):
        if (
            daily[i]["frequency"] >= FATIGUE_FREQ_THRESHOLD
            and daily[i + 1]["frequency"] >= FATIGUE_FREQ_THRESHOLD
            and _ctr(daily[i]) <= baseline * (1 - FATIGUE_CTR_DROP)
            and _ctr(daily[i + 1]) <= baseline * (1 - FATIGUE_CTR_DROP)
        ):
            return True, daily[i]["day"]
    return False, None


_COPY_BANK = {
    "urgency": [
        ("Only 48 hours left to lock in launch pricing", "Sale ends soon", "Shop now"),
        ("Last chance: prices go up Monday", "Don't miss out", "Get the deal"),
    ],
    "social_proof": [
        ("Join 40,000 marketers who ship winning ads", "Loved by growth teams", "See why"),
        ('"Cut our CPA by half" - Priya, Head of Growth', "Customers talk", "Read reviews"),
    ],
    "feature": [
        ("Auto-tag every creative by hook, angle and format", "New: AI tagging", "Learn more"),
        ("One dashboard for Meta, TikTok and Google spend", "All channels, one view", "Explore"),
    ],
    "offer": [
        ("Start free for 14 days, no card required", "Free trial", "Try free"),
        ("Get 2 months free on annual plans", "Limited offer", "Claim offer"),
    ],
    "fear": [
        ("Your competitors test 10x more creatives than you", "Falling behind?", "Catch up"),
        ("Ad fatigue is eating your budget while you sleep", "Stop the bleed", "Fix it"),
    ],
    "curiosity": [
        ("The creative format nobody tests but everyone should", "You won't guess it", "See it"),
        ("What do the biggest ad accounts do differently?", "The answer inside", "Find out"),
    ],
}


def _classify_hook(text: str) -> str:
    if "?" in text:
        return "question"
    if '"' in text:
        return "quote"
    if any(ch.isdigit() for ch in text):
        return "stat"
    if text.split(" ", 1)[0] in ("Only", "Last", "Stop", "Don't", "Join", "Get", "Start"):
        return "bold_claim"
    return "none"


def make_tag_tasks(rng: random.Random, n: int) -> list[dict]:
    tasks = []
    for i in range(n):
        angle = rng.choice(ANGLES)
        primary, headline, cta = rng.choice(_COPY_BANK[angle])
        has_cta = rng.random() > 0.25
        creative = {
            "id": f"cr-{i:03d}",
            "primary_text": primary,
            "headline": headline,
            "cta_button": cta if has_cta else None,
            "format": rng.choice(["video", "static", "carousel"]),
        }
        tasks.append({
            "type": "tag_creative",
            "input": creative,
            "expected": {
                "messaging_angle": angle,
                "has_cta": has_cta,
                "hook_type": _classify_hook(primary),
            },
        })
    return tasks


def make_fatigue_tasks(rng: random.Random, n: int, days: int = 14) -> list[dict]:
    tasks = []
    for i in range(n):
        fatigued = rng.random() > 0.4
        base_ctr = rng.uniform(0.015, 0.035)
        start_day = rng.randint(6, days - 3) if fatigued else None
        # Frequency growth: for fatigued creatives, pick a rate so frequency
        # crosses the threshold right at the intended start day; otherwise a
        # generic ramp. The rule below is the single source of truth.
        if start_day:
            rate = rng.uniform(3.0, 3.6) / start_day
        else:
            rate = rng.uniform(0.15, 0.35)
        # Half of the healthy creatives get a CTR dip with LOW frequency
        # (e.g. audience expansion): the rule says not fatigued, naive
        # drop-only detectors fire anyway.
        dip_day = None
        if start_day is None and rng.random() > 0.5:
            dip_day = rng.randint(4, days - 4)
        series = []
        for d in range(days):
            ctr = base_ctr
            if start_day is not None and d >= start_day:
                drop = min(0.55, 0.35 + (d - start_day) * 0.08)
                ctr = base_ctr * (1 - drop)
            elif dip_day is not None and d >= dip_day:
                ctr = base_ctr * 0.62
            ctr *= rng.uniform(0.97, 1.03)  # light noise, never trips the 30% rule
            impressions = rng.randint(8000, 20000)
            series.append({
                "day": d,
                "impressions": impressions,
                "clicks": int(impressions * ctr),
                "frequency": round(1.0 + d * rate, 2),
            })
        truth_flag, truth_day = fatigue_ground_truth(series)
        tasks.append({
            "type": "detect_fatigue",
            "input": {"creative_id": f"ad-{i:03d}", "daily": series},
            "expected": {"fatigued": truth_flag, "start_day": truth_day},
        })
    return tasks


def make_winner_tasks(rng: random.Random, n: int) -> list[dict]:
    tasks = []
    for i in range(n):
        scenario = "inconclusive" if i % 2 == 0 else "clear"  # forced mix
        if scenario == "inconclusive":
            n_a = rng.randint(1500, 4000)
            ctr_a, ctr_b = 0.020, rng.uniform(0.0205, 0.023)
        else:
            n_a = rng.randint(20000, 60000)
            ctr_a = 0.020
            ctr_b = 0.020 * rng.uniform(1.35, 1.8)
            if rng.random() > 0.5:
                ctr_a, ctr_b = ctr_b, ctr_a
        n_b = int(n_a * rng.uniform(0.8, 1.2))
        variants = [
            {"name": "A", "impressions": n_a, "clicks": int(n_a * ctr_a)},
            {"name": "B", "impressions": n_b, "clicks": int(n_b * ctr_b)},
        ]
        tasks.append({
            "type": "pick_winner",
            "input": {"hypothesis": "variant B lifts CTR over control A", "variants": variants},
            "expected": {},  # verdict is recomputed by the grader, never stored
        })
    return tasks


def build_dataset(seed: int = 7, per_type: int = 6) -> list[dict]:
    rng = random.Random(seed)
    tasks = (
        make_tag_tasks(rng, per_type)
        + make_fatigue_tasks(rng, per_type)
        + make_winner_tasks(rng, per_type)
    )
    rng.shuffle(tasks)
    return tasks


# ---------------------------------------------------------------- graders

def grade_tag_creative(expected: dict, answer: dict) -> tuple[float, str]:
    fields = ["messaging_angle", "has_cta", "hook_type"]
    misses = [f for f in fields if answer.get(f) != expected[f]]
    return (len(fields) - len(misses)) / len(fields), (
        "all tags correct" if not misses else "wrong: " + ",".join(misses)
    )


def grade_detect_fatigue(task_input: dict, answer: dict) -> tuple[float, str]:
    fatigued, start_day = fatigue_ground_truth(task_input["daily"])
    score, reasons = 0.0, []
    if bool(answer.get("fatigued")) == fatigued:
        score += 0.5
    else:
        reasons.append(f"fatigued flag wrong (truth={fatigued})")
    if fatigued:
        if answer.get("start_day") == start_day:
            score += 0.5
        else:
            reasons.append(f"start_day wrong (truth={start_day})")
    elif answer.get("start_day") in (None, -1):
        score += 0.5
    else:
        reasons.append("invented a start day on a healthy creative")
    return score, "; ".join(reasons) or "correct"


def grade_pick_winner(task_input: dict, answer: dict) -> tuple[float, str]:
    a, b = task_input["variants"]
    truth = ab_verdict(a["clicks"], a["impressions"], b["clicks"], b["impressions"])
    given = str(answer.get("winner", "")).strip().upper()
    if given == truth.upper():
        return 1.0, "correct"
    return 0.0, f"answered {given or '?'}, correct was {truth}"


GRADERS = {
    "tag_creative": lambda task, ans: grade_tag_creative(task["expected"], ans),
    "detect_fatigue": lambda task, ans: grade_detect_fatigue(task["input"], ans),
    "pick_winner": lambda task, ans: grade_pick_winner(task["input"], ans),
}


# ---------------------------------------------------------------- agents

_BASELINE_KEYWORDS = {
    "urgency": ["last chance", "only", "ends soon", "hours left", "prices go up"],
    "social_proof": ["join", "loved", "customers", "reviews", '"', "marketers who"],
    "feature": ["new:", "auto-", "dashboard", "one view", "learn more"],
    "offer": ["free", "offer", "trial", "claim"],
    "fear": ["competitor", "behind", "eating", "stop the", "falling"],
    "curiosity": ["guess", "nobody", "differently", "inside"],
}


def baseline_agent(task_type: str, task_input: dict) -> dict:
    """Heuristic reference bar: no LLM, no ground truth. Must land < 1.0."""
    if task_type == "tag_creative":
        text = "{} {} {}".format(
            task_input["primary_text"], task_input["headline"], task_input.get("cta_button") or ""
        ).lower()
        angle = "feature"
        for candidate, keys in _BASELINE_KEYWORDS.items():
            if any(k in text for k in keys):
                angle = candidate
                break
        hook = "none"
        if "?" in task_input["primary_text"]:
            hook = "question"
        elif '"' in task_input["primary_text"]:
            hook = "quote"
        elif any(ch.isdigit() for ch in task_input["primary_text"]):
            hook = "stat"
        return {
            "messaging_angle": angle,
            "has_cta": bool(task_input.get("cta_button")),
            "hook_type": hook,
        }
    if task_type == "detect_fatigue":
        daily = task_input["daily"]
        baseline = sum(_ctr(d) for d in daily[:3]) / 3
        for d in daily[3:]:
            if _ctr(d) <= baseline * 0.7:  # naive: ignores frequency + sustain
                return {"fatigued": True, "start_day": d["day"]}
        return {"fatigued": False, "start_day": None}
    if task_type == "pick_winner":
        a, b = task_input["variants"]
        ctr_a, ctr_b = a["clicks"] / a["impressions"], b["clicks"] / b["impressions"]
        if abs(ctr_a - ctr_b) < 1e-9:
            return {"winner": "inconclusive"}
        return {"winner": "A" if ctr_a > ctr_b else "B"}  # naive: no significance
    return {}


def oracle_agent(task_type: str, task_input: dict, expected: dict | None = None) -> dict:
    """Sanity harness with access to hidden truth. Must score exactly 1.0."""
    if task_type == "tag_creative":
        return dict(expected)
    if task_type == "detect_fatigue":
        fatigued, start_day = fatigue_ground_truth(task_input["daily"])
        return {"fatigued": fatigued, "start_day": start_day}
    if task_type == "pick_winner":
        a, b = task_input["variants"]
        return {"winner": ab_verdict(a["clicks"], a["impressions"], b["clicks"], b["impressions"])}
    return {}


# ---------------------------------------------------------------- runner

@dataclass
class TrialResult:
    task_type: str
    score: float
    reason: str


@dataclass
class BenchReport:
    agent: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def reward(self) -> float:
        return sum(t.score for t in self.trials) / len(self.trials) if self.trials else 0.0

    def by_type(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for t in self.trials:
            buckets.setdefault(t.task_type, []).append(t.score)
        return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}


def run(agent_fn, seed: int = 7, per_type: int = 6, agent_name: str = "agent") -> BenchReport:
    """One full trial. `agent_fn(task_type, task_input) -> dict`; scores are
    recorded only by this runner."""
    report = BenchReport(agent=agent_name)
    for task in build_dataset(seed=seed, per_type=per_type):
        if agent_fn is oracle_agent:
            answer = agent_fn(task["type"], task["input"], task.get("expected"))
        else:
            answer = agent_fn(task["type"], task["input"]) or {}
        score, reason = GRADERS[task["type"]](task, answer)
        report.trials.append(TrialResult(task["type"], round(score, 4), reason))
    return report


def render(report: BenchReport) -> str:
    lines = [f"agent: {report.agent}", f"overall reward: {report.reward:.3f}", ""]
    for task_type, score in report.by_type().items():
        lines.append(f"  {task_type:<16} {score:.3f}")
    lines.append("")
    for i, t in enumerate(report.trials):
        flag = "ok  " if t.score == 1.0 else ("part" if t.score > 0 else "miss")
        lines.append(f"[{flag}] trial {i:02d} {t.task_type:<16} {t.score:.2f}  {t.reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------- tests

class TestStats(unittest.TestCase):
    def test_clear_winner(self):
        self.assertEqual(ab_verdict(600, 20000, 400, 20000), "A")

    def test_inconclusive_small_sample(self):
        self.assertEqual(ab_verdict(42, 2000, 40, 2000), "inconclusive")

    def test_zero_impressions_rejected(self):
        with self.assertRaises(ValueError):
            ab_verdict(0, 0, 5, 100)

    def test_identical_zero_ctr_is_inconclusive(self):
        self.assertEqual(ab_verdict(0, 1000, 0, 1000), "inconclusive")


class TestGraders(unittest.TestCase):
    def test_fatigue_rule_detects_generated_fatigue(self):
        rng = random.Random(3)
        for task in make_fatigue_tasks(rng, 10):
            truth_flag, truth_day = fatigue_ground_truth(task["input"]["daily"])
            self.assertEqual(truth_flag, task["expected"]["fatigued"])
            self.assertEqual(truth_day, task["expected"]["start_day"])

    def test_fatigue_needs_frequency_and_sustain(self):
        # CTR collapses but frequency stays low -> NOT fatigue by the rule.
        daily = [
            {"day": d, "impressions": 10000, "clicks": int(10000 * (0.03 if d < 5 else 0.01)),
             "frequency": 1.2}
            for d in range(10)
        ]
        self.assertEqual(fatigue_ground_truth(daily), (False, None))

    def test_pick_winner_grader_marks_raw_ctr_guess_wrong(self):
        task = {"input": {"variants": [
            {"name": "A", "impressions": 2000, "clicks": 42},
            {"name": "B", "impressions": 2000, "clicks": 40},
        ]}}
        score, _ = grade_pick_winner(task["input"], {"winner": "A"})
        self.assertEqual(score, 0.0)  # correct answer is inconclusive


class TestBenchmark(unittest.TestCase):
    def test_oracle_scores_perfect(self):
        report = run(oracle_agent, agent_name="oracle")
        self.assertEqual(report.reward, 1.0)

    def test_baseline_below_oracle_but_nonzero(self):
        report = run(baseline_agent, agent_name="baseline")
        self.assertLess(report.reward, 1.0)
        self.assertGreater(report.reward, 0.3)

    def test_dataset_is_deterministic(self):
        self.assertEqual(build_dataset(seed=7), build_dataset(seed=7))

    def test_empty_agent_scores_low(self):
        report = run(lambda t, i: {}, agent_name="empty")
        self.assertLessEqual(report.reward, 0.4)


# ---------------------------------------------------------------- cli

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--agent", default="baseline", choices=["baseline", "oracle"])
    run_p.add_argument("--seed", type=int, default=7)
    run_p.add_argument("--per-type", type=int, default=6)
    run_p.add_argument("--json", action="store_true")
    sub.add_parser("test")
    args = parser.parse_args()

    if args.cmd == "test":
        unittest.main(argv=[parser.prog], exit=True, verbosity=2)
        return

    agent_fn = baseline_agent if args.agent == "baseline" else oracle_agent
    report = run(agent_fn, seed=args.seed, per_type=args.per_type, agent_name=args.agent)
    if args.json:
        print(json.dumps({
            "agent": report.agent,
            "reward": report.reward,
            "by_type": report.by_type(),
            "trials": [t.__dict__ for t in report.trials],
        }, indent=2))
    else:
        print(render(report))


if __name__ == "__main__":
    main()
