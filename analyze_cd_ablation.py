"""
analyze_cd_ablation.py
=========================
Executes the pre-registered analysis plan from
precommit/commitment_cd.txt: joins B (from results.jsonl) with C and D
(from results_cd.jsonl) on seed, runs the Friedman omnibus test first,
and only proceeds to Bonferroni-corrected pairwise Wilcoxon tests if the
omnibus test is significant — avoiding the multiple-comparisons problem
that three independent uncorrected pairwise t-tests would create.

Usage:
    python -m unlearning_meta.analyze_cd_ablation
"""

from __future__ import annotations
import json
import math
import os
import statistics

from scipy import stats as scipy_stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AB_PATH = os.path.join(BASE_DIR, "seed_results", "results.jsonl")
CD_PATH = os.path.join(BASE_DIR, "seed_results", "results_cd.jsonl")


def load_joined() -> list[dict]:
    b_by_seed = {}
    with open(AB_PATH) as f:
        for line in f:
            row = json.loads(line)
            b_by_seed[row["seed"]] = row["b_score"]

    joined = []
    with open(CD_PATH) as f:
        for line in f:
            row = json.loads(line)
            seed = row["seed"]
            if seed in b_by_seed:
                joined.append({
                    "seed": seed, "b": b_by_seed[seed],
                    "c": row["c_score"], "d": row["d_score"],
                })
    return joined


def confidence_interval(values: list[float], confidence: float = 0.95) -> tuple:
    n = len(values)
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    z = 1.96 if confidence == 0.95 else scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    return mean - z * se, mean + z * se


def plot_boxplot(
    rows: list[dict], chi2: float, p_omnibus: float, kendalls_w: float,
    save_path: str,
) -> None:
    """
    Boxplot of B/C/D score distributions across the matched seeds — makes
    the overlap behind the null Friedman result visually immediate, per
    external review round 6's suggestion. Stats are passed in (computed
    once in main()) rather than recomputed here, so the title can never
    drift out of sync with the printed numbers above it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = [r["b"] for r in rows]
    c = [r["c"] for r in rows]
    d = [r["d"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([b, c, d], tick_labels=["B\n(full system)",
                                            "C\n(no rule extraction)",
                                            "D\n(no strategy evolution)"],
                    patch_artist=True, widths=0.5)
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual seed points (paired, so a faint connecting sense
    # of the matched design is visible, not just independent clouds)
    for i, vals in enumerate([b, c, d], start=1):
        x = [i + (0.08 * (-1) ** j) for j, _ in enumerate(vals)]
        ax.scatter(x, vals, alpha=0.35, s=14, color="black", zorder=3)

    n = len(rows)
    ax.set_ylabel("Composite score (target-aware, §16)")
    ax.set_title(
        f"B vs C vs D — {n} matched seeds\n"
        f"Friedman χ²={chi2:.3f}, p={p_omnibus:.3f} "
        f"({'not ' if p_omnibus >= 0.05 else ''}significant), "
        f"Kendall's W={kendalls_w:.3f}",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nBoxplot saved to {save_path}")


def main() -> None:
    rows = load_joined()
    n = len(rows)
    b = [r["b"] for r in rows]
    c = [r["c"] for r in rows]
    d = [r["d"] for r in rows]

    print(f"n = {n} matched seeds\n")
    for name, vals in [("B (full system)", b), ("C (no rule extraction)", c),
                       ("D (no strategy evol.)", d)]:
        lo, hi = confidence_interval(vals)
        print(f"  {name:<24}: {statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]")
    print()

    # ── Step 1: omnibus test (pre-registered to run FIRST, alone) ──────────
    stat, p_omnibus = scipy_stats.friedmanchisquare(b, c, d)
    k = 3   # conditions
    kendalls_w = stat / (n * (k - 1))
    print(f"Friedman omnibus test: chi2={stat:.3f}  p={p_omnibus:.4f}")
    print(f"Kendall's W (effect size): {kendalls_w:.4f}  "
          f"({'negligible' if kendalls_w < 0.1 else 'small' if kendalls_w < 0.3 else 'moderate+'})")

    plot_path = os.path.join(BASE_DIR, "seed_results", "bcd_boxplot.png")
    plot_boxplot(rows, stat, p_omnibus, kendalls_w, plot_path)

    if p_omnibus >= 0.05:
        print("\nOmnibus test NOT significant (p >= 0.05).")
        print("Per the pre-registered plan (commitment_cd.txt), pairwise")
        print("tests are NOT run — doing so now would be exactly the")
        print("multiple-comparisons fishing this design was set up to avoid.")
        print("\nConclusion: no confirmed difference among B, C, and D.")
        return

    print("\nOmnibus test significant (p < 0.05) — proceeding to pre-registered")
    print("Bonferroni-corrected pairwise Wilcoxon tests (alpha = 0.05/3 = 0.0167):\n")

    pairs = [("B", b, "C", c), ("B", b, "D", d), ("C", c, "D", d)]
    alpha_corrected = 0.05 / 3
    for name1, v1, name2, v2 in pairs:
        w_stat, p = scipy_stats.wilcoxon(v1, v2)
        mean_diff = statistics.mean(v1) - statistics.mean(v2)
        sig = "SIGNIFICANT" if p < alpha_corrected else "not significant"
        print(f"  {name1} vs {name2}: mean_diff={mean_diff:+.4f}  "
              f"W={w_stat:.1f}  p={p:.4f}  ({sig} at Bonferroni-corrected 0.0167)")


if __name__ == "__main__":
    main()
