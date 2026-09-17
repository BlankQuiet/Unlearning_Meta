"""
run_multiseed_comparison.py
==============================
Multi-seed statistical comparison of the fixed-hyperparameter baseline vs.
the full Experience Engine, on the instance-level benchmark.

Why this exists
------------------
A single-seed run (seed=42) of run_baseline_comparison.py found the fixed
baseline OUTSCORING the full system. Two more seeds (123, 777), run to
sanity-check that result before writing it up, found the OPPOSITE —  the
full system winning by a wider margin than its one loss. This is a live
demonstration of exactly why "multi-seed statistical evaluation" has been
flagged as a priority by external review across multiple rounds: a
single-seed comparison here would have supported *either* conclusion
depending on which seed happened to be run, which is not a comparison at
all. This script runs enough seeds to report a mean ± std and a paired
per-seed win rate instead of a single, potentially-misleading number.

Run:
    python -m unlearning_meta.run_multiseed_comparison
"""

from __future__ import annotations
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unlearning_meta.run_baseline_comparison import build_base_config, run_condition
from unlearning_meta.models.base_model import MLP
from unlearning_meta.data.dataset_utils import (
    load_digits_instance_forgetting,
    compute_retrain_baseline_forget_acc,
)
from unlearning_meta.utils.win_compat import enable_console_compat
from unlearning_meta.main import set_seed, pretrain

SEEDS = [42, 123, 777, 2024, 31415, 8, 99]


def main() -> None:
    enable_console_compat()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  MULTI-SEED COMPARISON — fixed baseline vs full system      ║")
    print(f"║  {len(SEEDS)} seeds: {SEEDS}")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    a_scores, b_scores, deltas = [], [], []

    for seed in SEEDS:
        cfg = build_base_config()
        cfg.training.seed = seed
        cfg.training.device = device
        set_seed(seed)

        fl, rl, tl, full = load_digits_instance_forgetting(
            forget_fraction=0.10, batch_size=32, seed=seed,
        )
        base_model = MLP(64, [128, 64], 10).to(device)
        pretrain(base_model, full, epochs=15, lr=0.002, device=device, verbose=False)
        baseline_forget_acc = compute_retrain_baseline_forget_acc(
            fl, rl, model_factory=lambda: MLP(64, [128, 64], 10),
            epochs=15, lr=0.002, device=device, seed=seed,
        )

        a = run_condition("A_fixed", base_model, cfg, baseline_forget_acc,
                          fl, rl, tl, use_experience_engine=False)
        b = run_condition("B_full", base_model, cfg, baseline_forget_acc,
                          fl, rl, tl, use_experience_engine=True)

        a_scores.append(a["mean_score"])
        b_scores.append(b["mean_score"])
        delta = b["mean_score"] - a["mean_score"]
        deltas.append(delta)

        print(f"seed={seed:>6}: A(fixed)={a['mean_score']:.4f}  "
              f"B(full)={b['mean_score']:.4f}  Δ(B-A)={delta:+.4f}  "
              f"{'B wins' if delta > 0 else 'A wins'}")

    print("\n" + "═" * 70)
    print("SUMMARY")
    print("═" * 70)
    n = len(SEEDS)
    a_mean, a_std = statistics.mean(a_scores), statistics.stdev(a_scores)
    b_mean, b_std = statistics.mean(b_scores), statistics.stdev(b_scores)
    delta_mean = statistics.mean(deltas)
    delta_std  = statistics.stdev(deltas)
    b_win_rate = sum(1 for d in deltas if d > 0) / n

    print(f"  A (fixed hyperparameters): {a_mean:.4f} ± {a_std:.4f}  (n={n})")
    print(f"  B (full Experience Engine): {b_mean:.4f} ± {b_std:.4f}  (n={n})")
    print(f"  Paired delta (B - A):      {delta_mean:+.4f} ± {delta_std:.4f}")
    print(f"  B wins on {sum(1 for d in deltas if d > 0)}/{n} seeds "
          f"({b_win_rate:.0%})")

    # Paired t-test (two-sided) if scipy is available, else report raw stats only
    try:
        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_rel(b_scores, a_scores)
        print(f"  Paired t-test: t={t_stat:.3f}  p={p_value:.4f}  "
              f"({'significant at p<0.05' if p_value < 0.05 else 'NOT significant at p<0.05'})")
    except ImportError:
        print("  (scipy not available — skipping formal significance test; "
              "report mean/std/win-rate above as the evidence instead)")

    return a_scores, b_scores, deltas


if __name__ == "__main__":
    main()
