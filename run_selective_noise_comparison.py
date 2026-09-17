"""
run_selective_noise_comparison.py
=====================================
Compares condition B (existing full system, selective_noise_enabled=False)
against condition F (B + selective_noise_enabled=True, §20's mechanism),
per the two-stage pre-registered plan in
precommit/commitment_selective_noise.txt.

Usage:
    python -m unlearning_meta.run_selective_noise_comparison --seeds 42 123 ...
"""

from __future__ import annotations
import argparse
import copy
import json
import os
import time

from unlearning_meta.run_baseline_comparison import build_base_config
from unlearning_meta.models.base_model import MLP
from unlearning_meta.data.dataset_utils import (
    load_digits_instance_forgetting,
    compute_retrain_baseline_forget_acc,
)
from unlearning_meta.engines.unlearning_engine import UnlearningEngine
from unlearning_meta.engines.experience_engine import ExperienceEngine
from unlearning_meta.main import set_seed, pretrain
from unlearning_meta.utils.win_compat import enable_console_compat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "seed_results", "results_selective_noise.jsonl")


def run_one_condition(base_model, cfg, baseline_forget_acc, fl, rl, tl,
                      selective_noise_enabled: bool):
    device = cfg.training.device
    model = copy.deepcopy(base_model).to(device)

    ue_cfg = copy.deepcopy(cfg.unlearning)
    ue_cfg.selective_noise_enabled = selective_noise_enabled
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
    seeds = [s for s in args.seeds if s not in done]
    if not seeds:
        print("All requested seeds already present — nothing to do.")
        return

    t0 = time.time()
    with open(RESULTS_PATH, "a") as f:
        for seed in seeds:
            cfg = build_base_config()
            cfg.training.seed = seed
            set_seed(seed)

            fl, rl, tl, full = load_digits_instance_forgetting(
                forget_fraction=0.10, batch_size=32, seed=seed,
            )
            base_model = MLP(64, [128, 64], 10)
            pretrain(base_model, full, epochs=15, lr=0.002, device="cpu", verbose=False)
            baseline = compute_retrain_baseline_forget_acc(
                fl, rl, model_factory=lambda: MLP(64, [128, 64], 10),
                epochs=15, lr=0.002, device="cpu", seed=seed,
            )

            b_score = run_one_condition(base_model, cfg, baseline, fl, rl, tl,
                                        selective_noise_enabled=False)
            f_score = run_one_condition(base_model, cfg, baseline, fl, rl, tl,
                                        selective_noise_enabled=True)
            delta = f_score - b_score

            f.write(json.dumps({
                "seed": seed, "b_score": b_score, "f_score": f_score,
                "delta": delta,
            }) + "\n")
            f.flush()
            elapsed = time.time() - t0
            print(f"seed={seed}: B={b_score:.4f}  F={f_score:.4f}  "
                  f"delta={delta:+.4f}  [{elapsed:.0f}s elapsed]")

    n_total = len(already_done_seeds(RESULTS_PATH))
    print(f"\nBatch complete: {len(seeds)} new seeds "
          f"(total in file: {n_total})")


if __name__ == "__main__":
    main()
