"""
analyze_seed_results.py
==========================
Computes the full statistical comparison (mean/std, win rate, paired
t-test, effect size, 95% CI) from seed_results/results.jsonl —
the accumulated output of run_baseline_comparison.py (seeds 42, 123, 777,
2024, 31415, 8, 99) and run_seed_batch.py (additional seeds).

This is the exact analysis behind ARCHITECTURE.md §17.3's numbers,
re-runnable standalone rather than requiring re-running the (expensive)
model training that produced results.jsonl in the first place.

Usage:
    python -m unlearning_meta.analyze_seed_results
"""

from __future__ import annotations
import json
import math
import os
import statistics


RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "seed_results", "results.jsonl")


def load_results(path: str = RESULTS_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(rows: list[dict]) -> dict:
    a = [r["a_score"] for r in rows]
    b = [r["b_score"] for r in rows]
    d = [r["delta"] for r in rows]
    n = len(rows)

    a_mean, a_std = statistics.mean(a), statistics.stdev(a)
    b_mean, b_std = statistics.mean(b), statistics.stdev(b)
    d_mean, d_std = statistics.mean(d), statistics.stdev(d)
    b_win_rate = sum(1 for x in d if x > 0) / n
    cohens_d = d_mean / d_std if d_std > 0 else float("nan")

    se = d_std / math.sqrt(n)
    ci_lo, ci_hi = d_mean - 1.96 * se, d_mean + 1.96 * se

    result = {
        "n": n, "a_mean": a_mean, "a_std": a_std,
        "b_mean": b_mean, "b_std": b_std,
        "delta_mean": d_mean, "delta_std": d_std,
        "b_win_rate": b_win_rate, "cohens_d": cohens_d,
        "ci_95_low": ci_lo, "ci_95_high": ci_hi,
        "ci_includes_zero": ci_lo <= 0 <= ci_hi,
    }

    try:
        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_rel(b, a)
        result["t_stat"] = t_stat
        result["p_value"] = p_value
        result["significant_at_0.05"] = p_value < 0.05
    except ImportError:
        result["p_value"] = None

    return result


def print_report(result: dict) -> None:
    n = result["n"]
    print(f"n = {n} seeds\n")
    print(f"  A (fixed hyperparameters): {result['a_mean']:.4f} ± {result['a_std']:.4f}")
    print(f"  B (full Experience Engine): {result['b_mean']:.4f} ± {result['b_std']:.4f}")
    print(f"  Delta (B - A):             {result['delta_mean']:+.4f} ± {result['delta_std']:.4f}")
    print(f"  B win rate:                {result['b_win_rate']:.0%} "
          f"({round(result['b_win_rate']*n)}/{n})")
    print(f"  Cohen's d (paired):        {result['cohens_d']:.3f}")
    print(f"  95% CI on delta:           [{result['ci_95_low']:+.4f}, "
          f"{result['ci_95_high']:+.4f}]"
          f"{'  <- includes zero' if result['ci_includes_zero'] else ''}")

    if result.get("p_value") is not None:
        sig = "SIGNIFICANT" if result["significant_at_0.05"] else "NOT significant"
        print(f"  Paired t-test:             t={result['t_stat']:.3f}  "
              f"p={result['p_value']:.4f}  ({sig} at p<0.05)")
    else:
        print("  (scipy unavailable — no formal significance test computed)")

    # Sample size needed for 80% power at this effect size (rough estimate,
    # standard paired-t-test formula: n ≈ ((z_a/2 + z_b) / d)^2)
    d = abs(result["cohens_d"])
    if d > 0:
        n_needed = ((1.96 + 0.84) / d) ** 2
        print(f"\n  Estimated seeds needed for 80% power at this effect size: "
              f"~{n_needed:.0f}")
        if n_needed > n:
            print("  (i.e. more seeds than currently collected — "
                  "see run_seed_batch.py to extend)")


def main() -> None:
    rows = load_results()
    result = analyze(rows)
    print_report(result)


if __name__ == "__main__":
    main()
