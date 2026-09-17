"""
run_wine_exploratory_comparison.py
======================================
IDEAS.md Idea 11's first cross-dataset generalization check. Explicitly
exploratory (n=5, NOT pre-registered), mirroring §18's own low-key style
for a first generalization look -- not a confirmatory claim.

Same Condition A (fixed hyperparameters, use_experience_engine=False)
vs Condition B (full dual-engine system, use_experience_engine=True)
comparison as §17.4's original digits confirmation, on
load_wine_instance_forgetting instead -- same hyperparameters, UNCHANGED
from build_base_config()'s digits-calibrated defaults, except model
input/output dims (13/3, not 64/10) and batch_size (8, not 32), which
must change simply to match the data's actual shape. Every other
hyperparameter (forget_lr, noise_scale, ewc_lambda, fisher_samples,
etc.) is left exactly as calibrated for digits, deliberately -- this
checks whether the SAME fixed system transfers unchanged, not whether a
retuned version could be made to work.

Usage:
    python -m unlearning_meta.run_wine_exploratory_comparison
"""

from __future__ import annotations
import json
import os
import statistics
import time

from unlearning_meta.run_baseline_comparison import build_base_config, run_condition
from unlearning_meta.models.base_model import MLP
from unlearning_meta.data.dataset_utils import (
    load_wine_instance_forgetting,
    compute_retrain_baseline_forget_acc,
)
from unlearning_meta.main import set_seed, pretrain

SEEDS = [20000, 20001, 20002, 20003, 20004]  # fresh block, not used elsewhere
INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM = 13, [32, 16], 3
BATCH_SIZE = 8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "seed_results", "results_wine_exploratory.jsonl")


def model_factory():
    return MLP(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM)


def main() -> None:
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    rows = []
    t0 = time.time()

    with open(RESULTS_PATH, "w") as out:
        for seed in SEEDS:
            set_seed(seed)
            fl, rl, tl, full = load_wine_instance_forgetting(
                forget_fraction=0.10, batch_size=BATCH_SIZE, seed=seed,
            )

            cfg = build_base_config()
            cfg.model.input_dim, cfg.model.hidden_dims, cfg.model.output_dim = (
                INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM
            )
            cfg.training.seed = seed
            cfg.training.batch_size = BATCH_SIZE

            base_model = model_factory()
            pretrain(base_model, full, epochs=15, lr=0.002, device="cpu", verbose=False)
            baseline = compute_retrain_baseline_forget_acc(
                fl, rl, model_factory=model_factory, epochs=15, lr=0.002,
                device="cpu", seed=seed,
            )

            a = run_condition("A", base_model, cfg, baseline, fl, rl, tl,
                              use_experience_engine=False)
            b = run_condition("B", base_model, cfg, baseline, fl, rl, tl,
                              use_experience_engine=True)

            row = {
                "seed": seed, "target_forget_acc": baseline,
                "a_mean_score": a["mean_score"], "a_min_retain_acc": a["min_retain_acc"],
                "b_mean_score": b["mean_score"], "b_min_retain_acc": b["min_retain_acc"],
                "b_minus_a": b["mean_score"] - a["mean_score"],
                "n_rules_extracted": b["n_rules_extracted"],
            }
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()

            elapsed = time.time() - t0
            print(f"seed={seed}: A_mean={a['mean_score']:.4f} (min_retain={a['min_retain_acc']:.4f})  "
                  f"B_mean={b['mean_score']:.4f} (min_retain={b['min_retain_acc']:.4f})  "
                  f"B-A={row['b_minus_a']:+.4f}  [{elapsed:.0f}s]")

    diffs = [r["b_minus_a"] for r in rows]
    n = len(diffs)
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs) if n > 1 else 0.0
    favor_b = sum(1 for d in diffs if d > 0)

    print(f"\nn={n}  mean(B-A)={mean_d:+.4f}  std={std_d:.4f}  "
          f"[{favor_b}/{n} seeds favor B]")
    if std_d > 0:
        print(f"Informal Cohen's d = {mean_d/std_d:+.4f} "
              f"(exploratory, n={n} -- descriptive only, not a significance claim)")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
