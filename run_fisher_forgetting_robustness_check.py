"""
run_fisher_forgetting_robustness_check.py
==============================================
Exploratory check (n=15, NOT pre-registered, one fixed calibration point --
same status as ARCHITECTURE.md §19.2 itself) of whether per-layer epsilon
normalization closes the robustness gap found in §19.3: Fisher Forgetting's
single global epsilon triggered the retain_floor guardrail on seed=1004
(catastrophic retain collapse) while the other 14 of the same 15 seeds were
unaffected, at the identical noise_scale=3e-6.

Design
------
Paired by seed, identical pretrained starting model for both conditions,
isolating ONLY the per_layer_normalize toggle (noise_scale, epsilon, and
every other hyperparameter held fixed at §19's original calibrated values,
matching this project's established one-toggle-at-a-time convention):

  E  (original):  FisherForgettingBaseline(noise_scale=3e-6, epsilon=1e-10,
                   per_layer_normalize=False)   -- exactly §19's condition
  E' (per-layer):  same, but per_layer_normalize=True, per_layer_eps_fraction=0.01

This is explicitly a first exploratory look, not a confirmatory claim --
if it looks promising, the natural next step is a proper pre-registered
two-stage design, exactly as §20 -> §21's precommit/Stage-1/Stage-2 pattern.

Usage:
    python -m unlearning_meta.run_fisher_forgetting_robustness_check
"""

from __future__ import annotations
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

SEEDS = [42, 123, 777, 2024, 31415, 8, 99, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
# Identical to §18/§19.2's seed set (same seeds, same order) -- so the
# seed=1004 failure is being checked against itself, not a fresh draw.

NOISE_SCALE = 3e-6   # §19.1's calibrated operating point; held fixed --
                      # this check isolates ONLY per_layer_normalize.
EPSILON = 1e-10
RETAIN_FLOOR = 0.90

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "seed_results", "results_fisher_forgetting_per_layer.jsonl")


def score_condition(scrubbed_model, base_model, cfg, baseline_forget_acc, fl, rl, tl):
    evaluator = UnlearningEvaluator(scrubbed_model, device="cpu")
    metrics = evaluator.evaluate(fl, rl, tl, reference_model=base_model, cycle=0)
    score = metrics.composite_score(
        cfg.training.score_weights,
        target_forget_acc=baseline_forget_acc,
        retain_floor=RETAIN_FLOOR,
    )
    return score, metrics.forget_acc, metrics.retain_acc


def main() -> None:
    enable_console_compat()
    print("=" * 72)
    print("FISHER FORGETTING ROBUSTNESS CHECK -- per-layer epsilon normalization")
    print("Exploratory, n=15, NOT pre-registered (matches SS19.2's own status)")
    print("=" * 72 + "\n")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    rows = []
    t0 = time.time()

    with open(RESULTS_PATH, "w") as out:
        for seed in SEEDS:
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

            # Condition E: original (single global epsilon) -- §19's condition
            ff_e = FisherForgettingBaseline(noise_scale=NOISE_SCALE, epsilon=EPSILON,
                                            device="cpu", per_layer_normalize=False)
            scrubbed_e = ff_e.scrub(copy.deepcopy(base_model), rl)
            e_score, e_forget, e_retain = score_condition(
                scrubbed_e, base_model, cfg, baseline_forget_acc, fl, rl, tl)

            # Condition E': per-layer normalized epsilon (same noise_scale --
            # isolates ONLY the per_layer_normalize toggle)
            ff_ep = FisherForgettingBaseline(noise_scale=NOISE_SCALE, epsilon=EPSILON,
                                             device="cpu", per_layer_normalize=True,
                                             per_layer_eps_fraction=0.01)
            scrubbed_ep = ff_ep.scrub(copy.deepcopy(base_model), rl)
            ep_score, ep_forget, ep_retain = score_condition(
                scrubbed_ep, base_model, cfg, baseline_forget_acc, fl, rl, tl)

            row = {
                "seed": seed,
                "e_score": e_score, "e_forget_acc": e_forget, "e_retain_acc": e_retain,
                "e_catastrophic": e_score == -10.0,
                "ep_score": ep_score, "ep_forget_acc": ep_forget, "ep_retain_acc": ep_retain,
                "ep_catastrophic": ep_score == -10.0,
                "delta": ep_score - e_score,
            }
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()

            elapsed = time.time() - t0
            flag_e = " <-- CATASTROPHIC" if row["e_catastrophic"] else ""
            flag_ep = " <-- CATASTROPHIC" if row["ep_catastrophic"] else ""
            print(f"seed={seed:>6}  E={e_score:8.4f}{flag_e:<18}  "
                  f"E'={ep_score:8.4f}{flag_ep:<18}  [{elapsed:.0f}s]")

    # ---- Summary ----
    e_scores  = [r["e_score"]  for r in rows]
    ep_scores = [r["ep_score"] for r in rows]
    e_fail  = sum(r["e_catastrophic"]  for r in rows)
    ep_fail = sum(r["ep_catastrophic"] for r in rows)

    print("\n" + "=" * 72)
    print(f"E  (original, global eps):     mean={statistics.mean(e_scores):.4f}  "
          f"std={statistics.pstdev(e_scores):.4f}  catastrophic={e_fail}/{len(rows)}")
    print(f"E' (per-layer normalized):     mean={statistics.mean(ep_scores):.4f}  "
          f"std={statistics.pstdev(ep_scores):.4f}  catastrophic={ep_fail}/{len(rows)}")

    e_excl  = [r["e_score"]  for r in rows if not r["e_catastrophic"]]
    ep_excl = [r["ep_score"] for r in rows if not r["ep_catastrophic"]]
    if e_excl:
        print(f"E  excluding its own failures:  mean={statistics.mean(e_excl):.4f}  "
              f"std={statistics.pstdev(e_excl):.4f}  n={len(e_excl)}")
    if ep_excl:
        print(f"E' excluding its own failures:  mean={statistics.mean(ep_excl):.4f}  "
              f"std={statistics.pstdev(ep_excl):.4f}  n={len(ep_excl)}")

    seed_1004 = next((r for r in rows if r["seed"] == 1004), None)
    if seed_1004:
        print(f"\nseed=1004 specifically (the original E failure):")
        print(f"  E  score={seed_1004['e_score']:.4f}  retain_acc={seed_1004['e_retain_acc']:.4f}  "
              f"catastrophic={seed_1004['e_catastrophic']}")
        print(f"  E' score={seed_1004['ep_score']:.4f}  retain_acc={seed_1004['ep_retain_acc']:.4f}  "
              f"catastrophic={seed_1004['ep_catastrophic']}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
