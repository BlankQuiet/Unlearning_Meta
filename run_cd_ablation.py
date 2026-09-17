"""
run_cd_ablation.py
=====================
Collects Rule-Extraction-ablation (C) and Strategy-Evolution-ablation (D)
scores at the exact same 55 seeds already used for the A-vs-B comparison
in seed_results/results.jsonl (ARCHITECTURE.md §17.4), so all three
conditions are matched on identical starting models for a proper
repeated-measures comparison.

Pre-registered analysis plan: precommit/commitment_cd.txt (written before
this script was run). Appends to seed_results/results_cd.jsonl rather
than modifying results.jsonl, so the already-validated A/B data is never
touched.

Usage:
    python -m unlearning_meta.run_cd_ablation --start-index 0 --count 10
"""

from __future__ import annotations
import argparse
import json
import os
import time

from unlearning_meta.run_baseline_comparison import build_base_config, run_condition
from unlearning_meta.models.base_model import MLP
from unlearning_meta.data.dataset_utils import (
    load_digits_instance_forgetting,
    compute_retrain_baseline_forget_acc,
)
from unlearning_meta.main import set_seed, pretrain
from unlearning_meta.utils.win_compat import enable_console_compat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AB_RESULTS_PATH = os.path.join(BASE_DIR, "seed_results", "results.jsonl")
CD_RESULTS_PATH = os.path.join(BASE_DIR, "seed_results", "results_cd.jsonl")


def get_ab_seeds() -> list[int]:
    with open(AB_RESULTS_PATH) as f:
        return [json.loads(line)["seed"] for line in f if line.strip()]


def already_done_seeds(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["seed"] for line in f if line.strip()}


def run_batch(seeds: list[int]) -> None:
    os.makedirs(os.path.dirname(CD_RESULTS_PATH), exist_ok=True)
    done = already_done_seeds(CD_RESULTS_PATH)
    seeds = [s for s in seeds if s not in done]
    if not seeds:
        print("All requested seeds already have C/D data — nothing to do.")
        return

    t0 = time.time()
    with open(CD_RESULTS_PATH, "a") as f:
        for seed in seeds:
            cfg = build_base_config()
            cfg.training.seed = seed
            set_seed(seed)
            fl, rl, tl, full = load_digits_instance_forgetting(
                forget_fraction=0.10, batch_size=32, seed=seed
            )
            base_model = MLP(64, [128, 64], 10)
            pretrain(base_model, full, epochs=15, lr=0.002, device="cpu", verbose=False)
            baseline = compute_retrain_baseline_forget_acc(
                fl, rl, model_factory=lambda: MLP(64, [128, 64], 10),
                epochs=15, lr=0.002, device="cpu", seed=seed,
            )
            c = run_condition("C", base_model, cfg, baseline, fl, rl, tl,
                              use_experience_engine=True, disable_rule_extraction=True)
            d = run_condition("D", base_model, cfg, baseline, fl, rl, tl,
                              use_experience_engine=True, disable_strategy_evolution=True)
            f.write(json.dumps({
                "seed": seed, "c_score": c["mean_score"], "d_score": d["mean_score"],
            }) + "\n")
            f.flush()
            elapsed = time.time() - t0
            print(f"seed={seed}: C(no_extraction)={c['mean_score']:.4f} "
                  f"D(no_evolution)={d['mean_score']:.4f}  [{elapsed:.0f}s elapsed]")

    n_done = len(already_done_seeds(CD_RESULTS_PATH))
    print(f"\nBatch complete: {len(seeds)} new seeds in {time.time()-t0:.0f}s "
          f"(total C/D seeds so far: {n_done})")


def main() -> None:
    enable_console_compat()
    p = argparse.ArgumentParser()
    p.add_argument("--start-index", type=int, default=0,
                   help="Index into the 55-seed A/B list to start from")
    p.add_argument("--count", type=int, default=10)
    args = p.parse_args()

    ab_seeds = get_ab_seeds()
    batch = ab_seeds[args.start_index: args.start_index + args.count]
    run_batch(batch)


if __name__ == "__main__":
    main()
