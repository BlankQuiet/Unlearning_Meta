"""
run_selective_ascent_comparison.py
======================================
Pre-registered 2x2 factorial testing selective ascent (IDEAS.md Idea 3)
against selective noise (§20/§21), per
precommit/commitment_selective_ascent.txt, written before any of this
script's own (US/SS) data was collected.

  UU: selective_noise_enabled=False, selective_ascent_enabled=False  (reused from §21)
  SU: selective_noise_enabled=True,  selective_ascent_enabled=False  (reused from §21)
  US: selective_noise_enabled=False, selective_ascent_enabled=True   (NEW)
  SS: selective_noise_enabled=True,  selective_ascent_enabled=True   (NEW)

UU/SU are read from seed_results/results_selective_noise.jsonl
(b_score/f_score respectively) rather than re-run -- same seeds, same
config, already collected in §21. This script collects US and SS only,
paired against those existing scores by seed, then reports Effect_ascent
(primary), Effect_noise and Interaction (secondary/descriptive).

Usage:
    python -m unlearning_meta.run_selective_ascent_comparison --seeds 6000 6001 ... 6014
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
    load_digits_instance_forgetting,
    compute_retrain_baseline_forget_acc,
)
from unlearning_meta.engines.unlearning_engine import UnlearningEngine
from unlearning_meta.engines.experience_engine import ExperienceEngine
from unlearning_meta.main import set_seed, pretrain
from unlearning_meta.utils.win_compat import enable_console_compat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXISTING_PATH = os.path.join(BASE_DIR, "seed_results", "results_selective_noise.jsonl")
RESULTS_PATH  = os.path.join(BASE_DIR, "seed_results", "results_selective_ascent.jsonl")


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
    for cycle in range(cfg.training.num_cycles):
        m = ue.run_cycle(fl, rl, tl, cycle=cycle)
        proposal = ee.process(m)
        ue.apply_strategy(proposal)
        scores.append(m.composite_score(cfg.training.score_weights,
                                        target_forget_acc=baseline_forget_acc,
                                        retain_floor=0.90))
    return sum(scores) / len(scores)


def load_existing_uu_su(seeds_needed):
    """UU/SU per seed, from §21's already-collected data."""
    out = {}
    if not os.path.exists(EXISTING_PATH):
        return out
    with open(EXISTING_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["seed"] in seeds_needed:
                out[row["seed"]] = {"uu": row["b_score"], "su": row["f_score"]}
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

    existing = load_existing_uu_su(set(args.seeds))
    missing_existing = [s for s in args.seeds if s not in existing]
    if missing_existing:
        print(f"WARNING: no existing UU/SU data for seeds {missing_existing} "
              f"in {EXISTING_PATH} -- these seeds cannot be analyzed as part "
              f"of the 2x2 factorial (US/SS would be collected but orphaned). "
              f"Proceeding to collect US/SS for them anyway, but they will be "
              f"excluded from the Effect_ascent/Effect_noise/Interaction analysis.")

    done = already_done_seeds(RESULTS_PATH)
    seeds_to_run = [s for s in args.seeds if s not in done]

    if not seeds_to_run:
        print("All requested seeds already present in results_selective_ascent.jsonl.")
    else:
        t0 = time.time()
        with open(RESULTS_PATH, "a") as out:
            for seed in seeds_to_run:
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

                us_score = run_one_condition(base_model, cfg, baseline, fl, rl, tl,
                                             selective_noise_enabled=False,
                                             selective_ascent_enabled=True)
                ss_score = run_one_condition(base_model, cfg, baseline, fl, rl, tl,
                                             selective_noise_enabled=True,
                                             selective_ascent_enabled=True)

                row = {"seed": seed, "us_score": us_score, "ss_score": ss_score}
                out.write(json.dumps(row) + "\n")
                out.flush()
                elapsed = time.time() - t0
                print(f"seed={seed}: US={us_score:.4f}  SS={ss_score:.4f}  [{elapsed:.0f}s]")

    # ---- Join with existing UU/SU and analyze ----
    new_scores = {}
    with open(RESULTS_PATH) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                new_scores[row["seed"]] = row

    existing = load_existing_uu_su(set(args.seeds))
    rows = []
    for seed in args.seeds:
        if seed in existing and seed in new_scores:
            uu, su = existing[seed]["uu"], existing[seed]["su"]
            us, ss = new_scores[seed]["us_score"], new_scores[seed]["ss_score"]
            rows.append({
                "seed": seed, "uu": uu, "su": su, "us": us, "ss": ss,
                "effect_ascent": ((us - uu) + (ss - su)) / 2,
                "effect_noise":  ((su - uu) + (ss - us)) / 2,
                "interaction":   (ss - us) - (su - uu),
            })

    if not rows:
        print("\nNo seeds with both existing UU/SU and new US/SS data -- nothing to analyze.")
        return

    def summarize(key, label):
        vals = [r[key] for r in rows]
        mean_v = statistics.mean(vals)
        std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if std_v > 0:
            d = mean_v / std_v
        else:
            d = 0.0 if mean_v == 0 else (float("inf") if mean_v > 0 else float("-inf"))
        print(f"{label}: mean={mean_v:+.4f}  sample_std={std_v:.4f}  Cohen's d={d:+.4f}  n={len(vals)}")
        return mean_v, std_v, d

    print("\n" + "=" * 72)
    print(f"2x2 FACTORIAL ANALYSIS (n={len(rows)} paired seeds)")
    print("=" * 72)
    print(f"{'seed':>6} {'UU':>8} {'SU':>8} {'US':>8} {'SS':>8}")
    for r in rows:
        print(f"{r['seed']:>6} {r['uu']:>8.4f} {r['su']:>8.4f} {r['us']:>8.4f} {r['ss']:>8.4f}")
    print()
    _, _, d_ascent = summarize("effect_ascent", "Effect_ascent (PRIMARY)")
    summarize("effect_noise", "Effect_noise   (secondary)")
    summarize("interaction",  "Interaction    (descriptive only, underpowered at this n)")

    print("\nThis is Stage 1 (exploratory) for Effect_ascent -- NOT a significance")
    print("test, per pre-registration.")
    if abs(d_ascent) > 0 and math.isfinite(d_ascent):
        n_needed = ((1.96 + 0.84) / abs(d_ascent)) ** 2
        print(f"Implied Stage 2 n from |d|={abs(d_ascent):.4f}: {n_needed:.1f}")
        if n_needed > 80:
            print("Exceeds the pre-registered ceiling of 80 -> 'no practically")
            print("meaningful effect detected'. Stage 2 should NOT be run.")
        else:
            print(f"Within bounds -> Stage 2 n = {max(30, round(n_needed))} "
                  f"(floor 30 applied if needed), fresh seeds, all 4 conditions.")


if __name__ == "__main__":
    main()
