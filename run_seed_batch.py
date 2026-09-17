"""
run_seed_batch.py
====================
Extends the multi-seed baseline-vs-full-system comparison
(ARCHITECTURE.md §17.3) by running additional seeds and APPENDING to
seed_results/results.jsonl, rather than re-running everything from
scratch each time.

§17.3 found that the n=7 estimate's own effect-size-based guidance
("~25-30 seeds needed") was itself inflated by small-sample variance —
the properly-computed target at n=26 is closer to ~65 seeds. This script
exists to reach that incrementally, in batches, since a single run of
~65 seeds is a long-running job (~25-30 minutes on CPU at ~24s/seed).

Usage:
    python -m unlearning_meta.run_seed_batch --seeds 3000 3001 3002 ...
    python -m unlearning_meta.run_seed_batch --start 3000 --count 10

Then re-analyse the accumulated file with:
    python -m unlearning_meta.analyze_seed_results
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

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "seed_results", "results.jsonl")


def already_run_seeds(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["seed"] for line in f if line.strip()}


def run_batch(seeds: list[int]) -> None:
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    done = already_run_seeds(RESULTS_PATH)
    seeds = [s for s in seeds if s not in done]
    if not seeds:
        print("All requested seeds already present in results.jsonl — nothing to do.")
        return

    t0 = time.time()
    with open(RESULTS_PATH, "a") as f:
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
            a = run_condition("A", base_model, cfg, baseline, fl, rl, tl,
                              use_experience_engine=False)
            b = run_condition("B", base_model, cfg, baseline, fl, rl, tl,
                              use_experience_engine=True)
            delta = b["mean_score"] - a["mean_score"]
            f.write(json.dumps({
                "seed": seed, "a_score": a["mean_score"],
                "b_score": b["mean_score"], "delta": delta,
            }) + "\n")
            f.flush()
            elapsed = time.time() - t0
            print(f"seed={seed}: A={a['mean_score']:.4f} B={b['mean_score']:.4f} "
                  f"delta={delta:+.4f} {'B wins' if delta > 0 else 'A wins'}  "
                  f"[{elapsed:.0f}s elapsed]")

    print(f"\nBatch complete: {len(seeds)} new seeds in {time.time()-t0:.0f}s "
          f"(total in file: {len(already_run_seeds(RESULTS_PATH))})")


def main() -> None:
    enable_console_compat()
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", help="Explicit list of seeds to run")
    p.add_argument("--start", type=int, help="Start of a seed range")
    p.add_argument("--count", type=int, default=10, help="How many seeds from --start")
    args = p.parse_args()

    if args.seeds:
        seeds = args.seeds
    elif args.start is not None:
        seeds = list(range(args.start, args.start + args.count))
    else:
        p.error("Provide either --seeds or --start")
        return

    run_batch(seeds)


if __name__ == "__main__":
    main()
