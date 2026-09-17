"""
run_mean_normalization_comparison.py
========================================
Pre-registered test of mean-preserving normalization (IDEAS.md Idea 10),
per precommit/commitment_mean_normalization.txt, written before any of
this script's own (N_noise/N_ascent) data was collected.

Reuses U (uu_score), R_noise (su_score), and R_ascent (us_score) directly
from seed_results/results_class_level_selective.jsonl (§26) rather than
re-running them. Collects only N_noise (selective_noise_normalization=
"mean") and N_ascent (selective_ascent_normalization="mean") -- two new
conditions per seed, holding the other mechanism at uniform in each.

Usage:
    python -m unlearning_meta.run_mean_normalization_comparison --seeds 9000 9001 ...
"""

from __future__ import annotations
import argparse
import copy
import json
import math
import os
import statistics
import time

from scipy import stats as _scipy_stats

from unlearning_meta.run_baseline_comparison import build_base_config
from unlearning_meta.models.base_model import MLP
from unlearning_meta.data.dataset_utils import (
    load_digits_unlearning_datasets,
    compute_retrain_baseline_forget_acc,
)
from unlearning_meta.engines.unlearning_engine import UnlearningEngine
from unlearning_meta.engines.experience_engine import ExperienceEngine
from unlearning_meta.main import set_seed, pretrain
from unlearning_meta.utils.win_compat import enable_console_compat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXISTING_PATH = os.path.join(BASE_DIR, "seed_results", "results_class_level_selective.jsonl")
RESULTS_PATH  = os.path.join(BASE_DIR, "seed_results", "results_mean_normalization.jsonl")
FORGET_CLASSES = [0]


def run_one_condition(base_model, cfg, baseline_forget_acc, fl, rl, tl,
                      selective_noise_enabled, selective_noise_normalization,
                      selective_ascent_enabled, selective_ascent_normalization):
    device = cfg.training.device
    model = copy.deepcopy(base_model).to(device)

    ue_cfg = copy.deepcopy(cfg.unlearning)
    ue_cfg.selective_noise_enabled = selective_noise_enabled
    ue_cfg.selective_noise_normalization = selective_noise_normalization
    ue_cfg.selective_ascent_enabled = selective_ascent_enabled
    ue_cfg.selective_ascent_normalization = selective_ascent_normalization
    ue = UnlearningEngine(model=model, config=ue_cfg, device=device)

    exp_cfg = copy.deepcopy(cfg.experience)
    exp_cfg.use_contextual_bandit = True
    exp_cfg.target_forget_acc = baseline_forget_acc
    exp_cfg.retain_floor = 0.90
    ee = ExperienceEngine(exp_cfg, cfg.training.score_weights, seed=cfg.training.seed)

    scores = []
    for cycle in range(cfg.training.num_cycles):
        m = ue.run_cycle(fl, rl, tl, cycle=cycle)
        proposal = ee.process(m)
        ue.apply_strategy(proposal)
        scores.append(m.composite_score(cfg.training.score_weights,
                                        target_forget_acc=baseline_forget_acc,
                                        retain_floor=0.90))
    return sum(scores) / len(scores)


def load_existing(seeds_needed):
    out = {}
    if not os.path.exists(EXISTING_PATH):
        return out
    with open(EXISTING_PATH) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["seed"] in seeds_needed:
                    out[row["seed"]] = {"u": row["uu_score"], "r_noise": row["su_score"],
                                        "r_ascent": row["us_score"]}
    return out


def already_done_seeds(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["seed"] for line in f if line.strip()}


def main() -> None:
    enable_console_compat()
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    done = already_done_seeds(RESULTS_PATH)
    seeds_to_run = [s for s in args.seeds if s not in done]

    if not seeds_to_run:
        print("All requested seeds already present.")
    else:
        t0 = time.time()
        with open(RESULTS_PATH, "a") as out:
            for seed in seeds_to_run:
                cfg = build_base_config()
                cfg.training.seed = seed
                set_seed(seed)

                fl, rl, tl, full = load_digits_unlearning_datasets(
                    forget_classes=FORGET_CLASSES, batch_size=32, seed=seed,
                )
                base_model = MLP(64, [128, 64], 10)
                pretrain(base_model, full, epochs=15, lr=0.002, device="cpu", verbose=False)
                baseline = compute_retrain_baseline_forget_acc(
                    fl, rl, model_factory=lambda: MLP(64, [128, 64], 10),
                    epochs=15, lr=0.002, device="cpu", seed=seed,
                )

                n_noise = run_one_condition(base_model, cfg, baseline, fl, rl, tl,
                                            selective_noise_enabled=True,
                                            selective_noise_normalization="mean",
                                            selective_ascent_enabled=False,
                                            selective_ascent_normalization="minmax")
                n_ascent = run_one_condition(base_model, cfg, baseline, fl, rl, tl,
                                             selective_noise_enabled=False,
                                             selective_noise_normalization="raw",
                                             selective_ascent_enabled=True,
                                             selective_ascent_normalization="mean")

                row = {"seed": seed, "n_noise_score": n_noise, "n_ascent_score": n_ascent}
                out.write(json.dumps(row) + "\n")
                out.flush()
                elapsed = time.time() - t0
                print(f"seed={seed}: N_noise={n_noise:.4f}  N_ascent={n_ascent:.4f}  [{elapsed:.0f}s]")

    # ---- Analysis ----
    new_scores = {}
    with open(RESULTS_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                new_scores[r["seed"]] = r

    existing = load_existing(set(args.seeds))
    rows = []
    for seed in args.seeds:
        if seed in existing and seed in new_scores:
            rows.append({"seed": seed, **existing[seed], **new_scores[seed]})

    if not rows:
        print("\nNo seeds with both existing U/R and new N data -- nothing to analyze.")
        return

    def paired_test(vals_a, vals_b, label):
        diffs = [a - b for a, b in zip(vals_a, vals_b)]
        n = len(diffs)
        mean_d = statistics.mean(diffs)
        std_d = statistics.stdev(diffs) if n > 1 else 0.0
        se = std_d / (n ** 0.5) if n > 0 else 0.0
        if std_d > 0:
            t_stat, p_val = _scipy_stats.ttest_1samp(diffs, 0)
            d = mean_d / std_d
            ci_low, ci_high = _scipy_stats.t.interval(0.95, df=n - 1, loc=mean_d, scale=se)
        else:
            t_stat, p_val = 0.0, 1.0
            d = 0.0 if mean_d == 0 else (float("inf") if mean_d > 0 else float("-inf"))
            ci_low, ci_high = mean_d, mean_d
        print(f"{label}: n={n}  mean={mean_d:+.4f}  std={std_d:.4f}  t({n-1})={t_stat:.4f}  "
              f"p={p_val:.4f}  Cohen's d={d:+.4f}  95% CI=[{ci_low:+.4f}, {ci_high:+.4f}]")

    print("\n" + "=" * 72)
    print(f"MEAN-NORMALIZATION COMPARISON (n={len(rows)})")
    print("=" * 72)

    n_noise = [r["n_noise_score"] for r in rows]
    r_noise = [r["r_noise"] for r in rows]
    u_noise = [r["u"] for r in rows]
    n_ascent = [r["n_ascent_score"] for r in rows]
    r_ascent = [r["r_ascent"] for r in rows]
    u_ascent = [r["u"] for r in rows]

    print("\n-- NOISE mechanism --")
    paired_test(n_noise, r_noise, "N_noise vs R_noise (does mean-norm shrink SS26's harmful effect?)")
    paired_test(n_noise, u_noise, "N_noise vs U        (does mean-norm noise beat uniform?)")

    print("\n-- ASCENT mechanism --")
    paired_test(n_ascent, r_ascent, "N_ascent vs R_ascent (does mean-norm move ascent's null?)")
    paired_test(n_ascent, u_ascent, "N_ascent vs U         (does mean-norm ascent beat uniform?)")


if __name__ == "__main__":
    main()
