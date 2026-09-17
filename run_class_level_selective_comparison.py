"""
run_class_level_selective_comparison.py
===========================================
Pre-registered 2x2 factorial testing selective noise/ascent on
CLASS-level forgetting, per precommit/commitment_class_level_selective.txt,
written before any of this script's data was collected. See IDEAS.md
Idea 9 Phase 2.

Unlike run_selective_ascent_comparison.py (§24), no existing data can be
reused here -- this benchmark has never been tested in this project's
pre-registered comparison format before -- so all four conditions
(UU/SU/US/SS) are collected fresh, per seed.

Also records cycles_to_target (first cycle where forget_acc is within
0.02 of target_forget_acc) per condition -- descriptive only, motivated
by the pilot observation that retain_acc ceilings early on this task,
so convergence speed is expected to be where conditions differ.

Usage:
    python -m unlearning_meta.run_class_level_selective_comparison --seeds 9000 9001 ...
"""

from __future__ import annotations
import argparse
import copy
import json
import math
import os
import statistics
import time

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
RESULTS_PATH = os.path.join(BASE_DIR, "seed_results", "results_class_level_selective.jsonl")
FORGET_CLASSES = [0]
TARGET_TOLERANCE = 0.02


def run_one_condition(base_model, cfg, baseline_forget_acc, fl, rl, tl,
                      selective_noise_enabled: bool, selective_ascent_enabled: bool):
    device = cfg.training.device
    model = copy.deepcopy(base_model).to(device)

    ue_cfg = copy.deepcopy(cfg.unlearning)
    ue_cfg.selective_noise_enabled  = selective_noise_enabled
    ue_cfg.selective_ascent_enabled = selective_ascent_enabled
    ue = UnlearningEngine(model=model, config=ue_cfg, device=device)

    exp_cfg = copy.deepcopy(cfg.experience)
    exp_cfg.use_contextual_bandit = True
    exp_cfg.target_forget_acc = baseline_forget_acc
    exp_cfg.retain_floor = 0.90
    ee = ExperienceEngine(exp_cfg, cfg.training.score_weights, seed=cfg.training.seed)

    scores = []
    cycles_to_target = cfg.training.num_cycles  # capped default: never reached
    reached = False
    for cycle in range(cfg.training.num_cycles):
        m = ue.run_cycle(fl, rl, tl, cycle=cycle)
        proposal = ee.process(m)
        ue.apply_strategy(proposal)
        scores.append(m.composite_score(cfg.training.score_weights,
                                        target_forget_acc=baseline_forget_acc,
                                        retain_floor=0.90))
        if not reached and abs(m.forget_acc - baseline_forget_acc) <= TARGET_TOLERANCE:
            cycles_to_target = cycle
            reached = True
    return sum(scores) / len(scores), cycles_to_target


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

                row = {"seed": seed, "target_forget_acc": baseline}
                for label, sn, sa in (("uu", False, False), ("su", True, False),
                                      ("us", False, True),  ("ss", True, True)):
                    score, ctt = run_one_condition(base_model, cfg, baseline, fl, rl, tl,
                                                   selective_noise_enabled=sn,
                                                   selective_ascent_enabled=sa)
                    row[f"{label}_score"] = score
                    row[f"{label}_cycles_to_target"] = ctt

                out.write(json.dumps(row) + "\n")
                out.flush()
                elapsed = time.time() - t0
                print(f"seed={seed}: UU={row['uu_score']:.4f} SU={row['su_score']:.4f} "
                      f"US={row['us_score']:.4f} SS={row['ss_score']:.4f}  [{elapsed:.0f}s]")

    # ---- Analysis ----
    all_rows = {}
    with open(RESULTS_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                all_rows[r["seed"]] = r

    rows = [all_rows[s] for s in args.seeds if s in all_rows]
    if not rows:
        print("\nNo data to analyze.")
        return

    for r in rows:
        uu, su, us, ss = r["uu_score"], r["su_score"], r["us_score"], r["ss_score"]
        r["effect_ascent"] = ((us - uu) + (ss - su)) / 2
        r["effect_noise"]  = ((su - uu) + (ss - us)) / 2
        r["interaction"]   = (ss - us) - (su - uu)

    def summarize(key, label):
        vals = [r[key] for r in rows]
        mean_v = statistics.mean(vals)
        std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if std_v > 0:
            d = mean_v / std_v
        else:
            d = 0.0 if mean_v == 0 else (float("inf") if mean_v > 0 else float("-inf"))
        print(f"{label}: mean={mean_v:+.4f}  sample_std={std_v:.4f}  Cohen's d={d:+.4f}  n={len(vals)}")
        return d

    print("\n" + "=" * 72)
    print(f"CLASS-LEVEL 2x2 FACTORIAL (n={len(rows)} seeds, forget_classes={FORGET_CLASSES})")
    print("=" * 72)
    print(f"{'seed':>6} {'UU':>8} {'SU':>8} {'US':>8} {'SS':>8}   "
          f"{'ctt_UU':>7} {'ctt_SU':>7} {'ctt_US':>7} {'ctt_SS':>7}")
    for r in rows:
        print(f"{r['seed']:>6} {r['uu_score']:>8.4f} {r['su_score']:>8.4f} "
              f"{r['us_score']:>8.4f} {r['ss_score']:>8.4f}   "
              f"{r['uu_cycles_to_target']:>7} {r['su_cycles_to_target']:>7} "
              f"{r['us_cycles_to_target']:>7} {r['ss_cycles_to_target']:>7}")

    print()
    d_ascent = summarize("effect_ascent", "Effect_ascent (PRIMARY)")
    d_noise  = summarize("effect_noise",  "Effect_noise  (PRIMARY)")
    summarize("interaction", "Interaction   (descriptive only)")

    print(f"\nMean cycles_to_target: UU={statistics.mean(r['uu_cycles_to_target'] for r in rows):.1f}  "
          f"SU={statistics.mean(r['su_cycles_to_target'] for r in rows):.1f}  "
          f"US={statistics.mean(r['us_cycles_to_target'] for r in rows):.1f}  "
          f"SS={statistics.mean(r['ss_cycles_to_target'] for r in rows):.1f}  (descriptive only)")

    print("\nStage 1 (exploratory) -- NOT a significance test, per pre-registration.")
    for name, d in (("Effect_ascent", d_ascent), ("Effect_noise", d_noise)):
        if abs(d) > 0 and math.isfinite(d):
            n_needed = ((1.96 + 0.84) / abs(d)) ** 2
            verdict = ("exceeds ceiling of 80 -> 'no practically meaningful effect detected', "
                       "Stage 2 NOT run" if n_needed > 80 else
                       f"within bounds -> Stage 2 n = {max(30, round(n_needed))}, fresh seeds")
            print(f"{name}: implied Stage 2 n from |d|={abs(d):.4f} = {n_needed:.1f} -> {verdict}")


if __name__ == "__main__":
    main()
