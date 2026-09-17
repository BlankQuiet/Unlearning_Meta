"""
run_fisher_forgetting_confirmatory.py
=========================================
Pre-registered two-stage confirmatory design testing whether per-layer
epsilon normalization (per_layer_normalize=True) improves AVERAGE
performance over the original Fisher Forgetting baseline
(per_layer_normalize=False), on seeds NOT selected for containing the
known seed=1004 failure case. Plan fixed in advance in
precommit/commitment_fisher_forgetting_per_layer.txt, before any of
this script's data was collected. See IDEAS.md #1.

Same paired-by-seed design as run_fisher_forgetting_robustness_check.py
(§22's exploratory, reused-seed check), but:
  - takes an explicit --seeds argument, so Stage 1 and Stage 2 use
    disjoint, purpose-drawn fresh seed blocks rather than one fixed list
  - appends to (rather than overwrites) its results file, with dedup,
    so Stage 2 can be run as a separate invocation without re-running
    Stage 1's seeds
  - reports Cohen's d for the paired score differences -- Stage 1's
    ONLY job; not analyzed for significance at Stage 1, per the
    pre-registration

Usage:
    python -m unlearning_meta.run_fisher_forgetting_confirmatory --seeds 7000 7001 ... 7014
"""

from __future__ import annotations
import argparse
import copy
import json
import os
import statistics
import time

from unlearning_meta.run_baseline_comparison import build_base_config
from unlearning_meta.models.base_model import MLP
from unlearning_meta.data.dataset_utils import (
    load_digits_instance_forgetting,
    compute_retrain_baseline_forget_acc,
)
from unlearning_meta.modules.unlearning.fisher_forgetting_baseline import (
    FisherForgettingBaseline,
)
from unlearning_meta.modules.unlearning.evaluator import UnlearningEvaluator
from unlearning_meta.main import set_seed, pretrain
from unlearning_meta.utils.win_compat import enable_console_compat

NOISE_SCALE = 3e-6
EPSILON = 1e-10
PER_LAYER_EPS_FRACTION = 0.01
RETAIN_FLOOR = 0.90

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "seed_results", "results_fisher_forgetting_confirmatory.jsonl")


def score_condition(scrubbed_model, base_model, cfg, baseline_forget_acc, fl, rl, tl):
    evaluator = UnlearningEvaluator(scrubbed_model, device="cpu")
    metrics = evaluator.evaluate(fl, rl, tl, reference_model=base_model, cycle=0)
    score = metrics.composite_score(
        cfg.training.score_weights,
        target_forget_acc=baseline_forget_acc,
        retain_floor=RETAIN_FLOOR,
    )
    return score, metrics.forget_acc, metrics.retain_acc


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
        print("All requested seeds already present -- nothing to do.")
    else:
        t0 = time.time()
        with open(RESULTS_PATH, "a") as out:
            for seed in seeds:
                cfg = build_base_config()
                cfg.training.seed = seed
                set_seed(seed)

                fl, rl, tl, full = load_digits_instance_forgetting(
                    forget_fraction=0.10, batch_size=32, seed=seed,
                )
                base_model = MLP(64, [128, 64], 10)
                pretrain(base_model, full, epochs=15, lr=0.002, device="cpu", verbose=False)
                baseline_forget_acc = compute_retrain_baseline_forget_acc(
                    fl, rl, model_factory=lambda: MLP(64, [128, 64], 10),
                    epochs=15, lr=0.002, device="cpu", seed=seed,
                )

                ff_e = FisherForgettingBaseline(noise_scale=NOISE_SCALE, epsilon=EPSILON,
                                                device="cpu", per_layer_normalize=False)
                scrubbed_e = ff_e.scrub(copy.deepcopy(base_model), rl)
                e_score, e_forget, e_retain = score_condition(
                    scrubbed_e, base_model, cfg, baseline_forget_acc, fl, rl, tl)

                ff_ep = FisherForgettingBaseline(noise_scale=NOISE_SCALE, epsilon=EPSILON,
                                                 device="cpu", per_layer_normalize=True,
                                                 per_layer_eps_fraction=PER_LAYER_EPS_FRACTION)
                scrubbed_ep = ff_ep.scrub(copy.deepcopy(base_model), rl)
                ep_score, ep_forget, ep_retain = score_condition(
                    scrubbed_ep, base_model, cfg, baseline_forget_acc, fl, rl, tl)

                row = {
                    "seed": seed,
                    "e_score": e_score, "e_retain_acc": e_retain,
                    "e_catastrophic": e_score == -10.0,
                    "ep_score": ep_score, "ep_retain_acc": ep_retain,
                    "ep_catastrophic": ep_score == -10.0,
                    "delta": ep_score - e_score,
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                elapsed = time.time() - t0
                print(f"seed={seed:>6}  E={e_score:8.4f}  E'={ep_score:8.4f}  "
                      f"delta={row['delta']:+.4f}  [{elapsed:.0f}s]")

    # ---- Analysis over exactly the requested seed block ----
    all_rows = []
    with open(RESULTS_PATH) as f:
        for line in f:
            if line.strip():
                all_rows.append(json.loads(line))

    requested_set = set(args.seeds)
    rows = [r for r in all_rows if r["seed"] in requested_set]

    deltas = [r["delta"] for r in rows]
    mean_d = statistics.mean(deltas)
    sample_std_d = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    if sample_std_d > 0:
        cohens_d = mean_d / sample_std_d
    else:
        cohens_d = 0.0 if mean_d == 0 else (float("inf") if mean_d > 0 else float("-inf"))

    e_fail  = sum(r["e_catastrophic"]  for r in rows)
    ep_fail = sum(r["ep_catastrophic"] for r in rows)

    print("\n" + "=" * 72)
    print(f"n = {len(rows)} seeds: {sorted(r['seed'] for r in rows)}")
    print(f"mean(E' - E) = {mean_d:+.4f}   sample std = {sample_std_d:.4f}   Cohen's d = {cohens_d:+.4f}")
    print(f"catastrophic failures: E={e_fail}/{len(rows)}   E'={ep_fail}/{len(rows)}  "
          f"(descriptive only, not power-analyzed at this n)")

    if len(rows) <= 15:
        print("\nThis is Stage 1 (exploratory) -- NOT a significance test, per pre-registration.")
        if sample_std_d == 0:
            print("Zero variance in deltas -- cannot compute a Stage 2 n from this.")
        else:
            n_needed = ((1.96 + 0.84) / abs(cohens_d)) ** 2
            print(f"Implied Stage 2 n from |d|={abs(cohens_d):.4f}: {n_needed:.1f}")
            if n_needed > 80:
                n_stage2 = None
                print("Exceeds the pre-registered ceiling of 80 -> per the plan's own advance")
                print("instruction, this is 'no practically meaningful effect detected.'")
                print("Stage 2 should NOT be run.")
            else:
                n_stage2 = max(30, round(n_needed))
                print(f"Within bounds -> Stage 2 n = {n_stage2} (floor 30 applied if needed).")
                print("Stage 2 should use a fresh, non-overlapping seed block (e.g. 8000+).")

    print(f"\nResults file: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
