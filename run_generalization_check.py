"""
run_generalization_check.py
==============================
Exploratory check of whether §17.4's confirmed A-vs-B effect generalizes
beyond the one setup (digits, 64->128->64->10 MLP, forget_fraction=0.10)
used for every seed run in §10-17 (140+ model-training runs, all on that
one configuration).

This is explicitly NOT a full-powered replication of §17.4's rigor —
it uses n=15 seeds, pre-registered as an exploratory sample size (not
derived from a power calculation), specifically to get an initial signal
on whether the effect holds at all under a changed condition before
committing the much larger n≈55+ investment a properly-powered check of
each dimension (forget_fraction, model size, dataset) would need. See
ARCHITECTURE.md §18 for the full reasoning behind checking this
dimension first and treating this as preliminary.

Usage:
    python -m unlearning_meta.run_generalization_check --forget-fraction 0.20
"""

from __future__ import annotations
import argparse
import json
import os
import statistics
import time

from scipy import stats as scipy_stats

from unlearning_meta.run_baseline_comparison import build_base_config, run_condition
from unlearning_meta.models.base_model import MLP
from unlearning_meta.data.dataset_utils import (
    load_digits_instance_forgetting,
    compute_retrain_baseline_forget_acc,
)
from unlearning_meta.main import set_seed, pretrain
from unlearning_meta.utils.win_compat import enable_console_compat

SEEDS = [42, 123, 777, 2024, 31415, 8, 99, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
# Deliberately the SAME first 15 seeds already used in §17's A/B seed set —
# not to reuse their §17 scores (a different forget_fraction changes the
# task entirely, so those scores don't transfer) but so a reader can see
# this isn't quietly cherry-picked seeds; they're the same ones already
# committed to in this document for the original forget_fraction.


def main() -> None:
    enable_console_compat()
    p = argparse.ArgumentParser()
    p.add_argument("--forget-fraction", type=float, default=0.20)
    args = p.parse_args()

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  GENERALIZATION CHECK (exploratory, n=15, not full-powered) ║")
    print(f"║  forget_fraction={args.forget_fraction} (§10-17 all used 0.10)          ")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    a_scores, b_scores, deltas = [], [], []
    t0 = time.time()

    for seed in SEEDS:
        cfg = build_base_config()
        cfg.training.seed = seed
        set_seed(seed)

        fl, rl, tl, full = load_digits_instance_forgetting(
            forget_fraction=args.forget_fraction, batch_size=32, seed=seed,
        )
        base_model = MLP(64, [128, 64], 10)
        pretrain(base_model, full, epochs=15, lr=0.002, device="cpu", verbose=False)
        baseline = compute_retrain_baseline_forget_acc(
            fl, rl, model_factory=lambda: MLP(64, [128, 64], 10),
            epochs=15, lr=0.002, device="cpu", seed=seed,
        )

        a = run_condition("A", base_model, cfg, baseline, fl, rl, tl,
                          use_experience_engine=False)
        b = run_condition("B", base_model, cfg, baseline, fl, rl, tl,
                          use_experience_engine=True)
        delta = b["mean_score"] - a["mean_score"]
        a_scores.append(a["mean_score"])
        b_scores.append(b["mean_score"])
        deltas.append(delta)

        elapsed = time.time() - t0
        print(f"seed={seed:>6}: A={a['mean_score']:.4f}  B={b['mean_score']:.4f}  "
              f"delta={delta:+.4f}  {'B wins' if delta > 0 else 'A wins'}  "
              f"[{elapsed:.0f}s]")

    n = len(SEEDS)
    a_mean, a_std = statistics.mean(a_scores), statistics.stdev(a_scores)
    b_mean, b_std = statistics.mean(b_scores), statistics.stdev(b_scores)
    d_mean, d_std = statistics.mean(deltas), statistics.stdev(deltas)
    b_win_rate = sum(1 for x in deltas if x > 0) / n

    print("\n" + "═" * 70)
    print(f"EXPLORATORY SUMMARY (n={n}, forget_fraction={args.forget_fraction})")
    print("═" * 70)
    print(f"  A (fixed):  {a_mean:.4f} ± {a_std:.4f}")
    print(f"  B (full):   {b_mean:.4f} ± {b_std:.4f}")
    print(f"  Delta(B-A): {d_mean:+.4f} ± {d_std:.4f}")
    print(f"  B win rate: {sum(1 for x in deltas if x>0)}/{n} = {b_win_rate:.0%}")

    t_stat, p_value = scipy_stats.ttest_rel(b_scores, a_scores)
    cohens_d = d_mean / d_std if d_std > 0 else float("nan")
    print(f"  Paired t-test: t={t_stat:.3f}  p={p_value:.4f}  "
          f"({'significant' if p_value < 0.05 else 'NOT significant (expected at n=15 — exploratory only)'})")
    print(f"  Cohen's d: {cohens_d:.3f}")

    print("\n  Note: n=15 is not powered to confirm significance on its own")
    print("  (§17.3-17.4 needed 55 for that at a similar effect size). This")
    print("  is a directional check: does B still trend positive here?")
    print(f"  {'YES' if d_mean > 0 else 'NO'} — {'consistent with' if d_mean > 0 else 'inconsistent with'} "
          f"generalization beyond forget_fraction=0.10.")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "seed_results",
                            f"generalization_ff{args.forget_fraction}.json")
    with open(out_path, "w") as f:
        json.dump({
            "forget_fraction": args.forget_fraction, "n": n,
            "a_mean": a_mean, "b_mean": b_mean, "delta_mean": d_mean,
            "b_win_rate": b_win_rate, "p_value": p_value, "cohens_d": cohens_d,
        }, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
