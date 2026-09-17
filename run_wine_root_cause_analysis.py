"""
run_wine_root_cause_analysis.py
===================================
Diagnostic follow-up to run_wine_exploratory_comparison.py's result
(B underperforms A on wine, opposite of §17.4's digits finding).
Explicitly exploratory analysis, not a new pre-registered comparison and
not a mechanism change -- captures per-cycle detail that
run_condition()'s summary-only return doesn't expose, to distinguish:

  (a) adaptation itself is harmful on wine (even starting fresh)
  (b) negative transfer from prior experience
  (c) excessive/noisy exploration given wine's small, simple task

NOTE on (b): StrategyMemory (modules/experience/strategy_memory.py) has
no disk persistence -- every ExperienceEngine() instantiation starts
with empty active/dormant/archive rule lists, confirmed by reading its
__init__ directly. Every condition-B run in this project, on any
dataset, starts from zero prior rules. There is no cross-run memory
transfer in the current architecture at all, so (b) in the
"pre-existing-experience-carries-over" sense cannot be what's happening
here.

IMPORTANT METHODOLOGICAL NOTE: this script runs condition A (via the
same run_condition() used everywhere else in this project) BEFORE the
instrumented condition B, even though only B's per-cycle detail is
needed. set_seed() resets the RNG once per seed, and every downstream
call (pretrain, baseline computation, condition A, condition B) shares
that ONE continuous stream. Running instrumented B without first running
A would start B from a DIFFERENT point in that shared stream than
run_wine_exploratory_comparison.py's own B did for "the same seed" --
silently producing different numbers, not because of a logic bug in
either implementation, but purely because of this ordering. Confirmed
directly during this analysis's own development: calling run_condition
('B', ...) and an early, A-skipping version of this file's instrumented
routine back-to-back from identical starting state gave DIFFERENT
results for nominally the same seed (3.7331 vs 3.6770). Every run in
this file follows the exploratory script's exact A-then-B sequence for
that reason.

Captures per cycle: full UnlearningMetrics (forget_acc, retain_acc,
accuracy, privacy, hallucination, stability), composite_score, the
proposed StrategyUpdate (action, delta, confidence, source_rule -- None
means the proposal came from raw bandit exploration, not an extracted
rule), the full hyperparameter config snapshot (dataclasses.asdict), and
StrategyMemory's active/dormant/archive rule counts.

Usage:
    python -m unlearning_meta.run_wine_root_cause_analysis
"""

from __future__ import annotations
import copy
import dataclasses
import json
import os

from unlearning_meta.run_baseline_comparison import build_base_config, run_condition
from unlearning_meta.engines.unlearning_engine import UnlearningEngine
from unlearning_meta.engines.experience_engine import ExperienceEngine
from unlearning_meta.models.base_model import MLP
from unlearning_meta.data.dataset_utils import (
    load_wine_instance_forgetting,
    load_digits_instance_forgetting,
    compute_retrain_baseline_forget_acc,
)
from unlearning_meta.main import set_seed, pretrain
from unlearning_meta.schemas import StrategyState

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WINE_SEEDS   = [20000, 20001, 20002, 20003, 20004]  # identical to the exploratory check
DIGITS_SEEDS = [30000, 30001]  # fresh, small contrast sample -- not testing digits, just
                                # instrumenting the SAME diagnostic on the known-positive
                                # case for comparison


def instrumented_condition_b(base_model, cfg, baseline_forget_acc, fl, rl, tl):
    device = cfg.training.device
    model = copy.deepcopy(base_model).to(device)

    ue_cfg = copy.deepcopy(cfg.unlearning)
    ue = UnlearningEngine(model=model, config=ue_cfg, device=device)

    exp_cfg = copy.deepcopy(cfg.experience)
    exp_cfg.target_forget_acc = baseline_forget_acc
    exp_cfg.retain_floor = 0.90
    ee = ExperienceEngine(exp_cfg, cfg.training.score_weights, seed=cfg.training.seed)

    cycles = []
    for cycle in range(cfg.training.num_cycles):
        m = ue.run_cycle(fl, rl, tl, cycle=cycle)
        proposal = ee.process(m)
        applied = ue.apply_strategy(proposal)
        score = m.composite_score(cfg.training.score_weights,
                                  target_forget_acc=baseline_forget_acc,
                                  retain_floor=0.90)

        cycles.append({
            "cycle": cycle,
            "forget_acc": m.forget_acc, "retain_acc": m.retain_acc,
            "accuracy": m.accuracy, "privacy": m.privacy,
            "hallucination": m.hallucination, "stability": m.stability,
            "score": score,
            "action": proposal.action.value, "delta": proposal.delta,
            "confidence": proposal.confidence, "source_rule": proposal.source_rule,
            "applied": applied,
            "n_active_rules": sum(1 for r in ee.memory.records.values()
                                  if r.state == StrategyState.ACTIVE),
            "n_dormant_rules": sum(1 for r in ee.memory.records.values()
                                   if r.state == StrategyState.DORMANT),
            "n_archive_rules": sum(1 for r in ee.memory.records.values()
                                   if r.state == StrategyState.ARCHIVE),
            "config_snapshot": {
                k: v for k, v in dataclasses.asdict(ue_cfg).items()
                if isinstance(v, (int, float, bool))
            },
        })

    final_config = {k: v for k, v in dataclasses.asdict(ue_cfg).items()
                    if isinstance(v, (int, float, bool))}
    return cycles, final_config


def run_one(seed, loader_fn, input_dim, hidden_dims, output_dim, batch_size, label):
    set_seed(seed)
    fl, rl, tl, full = loader_fn(forget_fraction=0.10, batch_size=batch_size, seed=seed)

    cfg = build_base_config()
    cfg.model.input_dim, cfg.model.hidden_dims, cfg.model.output_dim = (
        input_dim, hidden_dims, output_dim
    )
    cfg.training.seed = seed
    cfg.training.batch_size = batch_size

    model_factory = lambda: MLP(input_dim, hidden_dims, output_dim)
    base_model = model_factory()
    pretrain(base_model, full, epochs=15, lr=0.002, device="cpu", verbose=False)
    baseline = compute_retrain_baseline_forget_acc(
        fl, rl, model_factory=model_factory, epochs=15, lr=0.002, device="cpu", seed=seed,
    )

    # Run condition A first via the same established run_condition() used
    # everywhere else in this project -- see this file's module docstring
    # for why this ordering matters (shared RNG stream with condition B).
    a = run_condition("A", base_model, cfg, baseline, fl, rl, tl,
                      use_experience_engine=False)

    cycles, final_config = instrumented_condition_b(base_model, cfg, baseline, fl, rl, tl)

    n_rule_guided = sum(1 for c in cycles if c["source_rule"] is not None)
    n_applied = sum(1 for c in cycles if c["applied"])
    actions_used = sorted(set(c["action"] for c in cycles))
    b_mean_score = sum(c["score"] for c in cycles) / len(cycles)
    n_floor_violations = sum(1 for c in cycles if c["score"] == -10.0)

    print(f"\n[{label}] seed={seed}: target_forget_acc={baseline:.4f}")
    print(f"  A: mean_score={a['mean_score']:.4f}  min_retain_acc={a['min_retain_acc']:.4f}")
    print(f"  B: mean_score={b_mean_score:.4f}  min_retain_acc={min(c['retain_acc'] for c in cycles):.4f}  "
          f"(B-A={b_mean_score - a['mean_score']:+.4f})")
    print(f"  B retain_floor penalty cycles (score==-10): {n_floor_violations}/25")
    print(f"  B final: forget_acc={cycles[-1]['forget_acc']:.4f} retain_acc={cycles[-1]['retain_acc']:.4f} "
          f"score={cycles[-1]['score']:.4f}")
    print(f"  B proposals applied: {n_applied}/25   rule-guided (source_rule != None): {n_rule_guided}/25")
    print(f"  B rules by end: active={cycles[-1]['n_active_rules']} dormant={cycles[-1]['n_dormant_rules']} "
          f"archive={cycles[-1]['n_archive_rules']}")
    print(f"  distinct actions used: {actions_used}")
    print(f"  min retain_acc across cycles: {min(c['retain_acc'] for c in cycles):.4f} "
          f"(at cycle {min(range(len(cycles)), key=lambda i: cycles[i]['retain_acc'])})")

    return {"seed": seed, "label": label, "target_forget_acc": baseline,
            "a_summary": a, "b_mean_score": b_mean_score,
            "cycles": cycles, "final_config": final_config,
            "n_rule_guided": n_rule_guided, "n_applied": n_applied}


def main():
    all_results = []
    print("=" * 72)
    print("WINE (the underperforming case)")
    print("=" * 72)
    for seed in WINE_SEEDS:
        r = run_one(seed, load_wine_instance_forgetting, 13, [32, 16], 3, 8, "wine")
        all_results.append(r)

    print("\n" + "=" * 72)
    print("DIGITS (known-positive contrast case, same instrumentation)")
    print("=" * 72)
    for seed in DIGITS_SEEDS:
        r = run_one(seed, load_digits_instance_forgetting, 64, [128, 64], 10, 32, "digits")
        all_results.append(r)

    out_path = os.path.join(BASE_DIR, "seed_results", "wine_root_cause_analysis.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nFull per-cycle detail written to {out_path}")


if __name__ == "__main__":
    main()
