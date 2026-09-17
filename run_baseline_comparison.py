"""
run_baseline_comparison.py
=============================
Baselines and ablations on the instance-level forgetting benchmark,
using the now-fully-calibrated evaluation (§16 target-aware scoring) and
strategy-selection mechanism (§13-15 grad-clip fix + unified bandit +
momentum). This has been on the external-review backlog (§11.5 item 5)
since round 2 and flagged as top priority in every review since — this
script finally executes it.

Conditions compared
----------------------
A) Fixed hyperparameters  — no ExperienceEngine at all; the "dumbest"
   baseline any unlearning paper needs. Same hyperparameters used
   throughout §12-16, held static for all 25 cycles.
B) Full Experience Engine — bandit + momentum + target-aware scoring;
   the proposed method.
C) No Rule Extraction     — Experience Engine active, bandit active, but
   RuleExtractor.extract() is never called, so StrategyMemory stays
   empty and rule_priors are always 0 (the bandit reduces to pure
   UCB + momentum, no rule-based prior term at all).
D) No Strategy Evolution  — Rule Extraction runs normally, but
   StrategyEvolution.evolve() is never called, so rules accumulate
   without ever being merged/generalised.

All four conditions start from an IDENTICAL deep-copied pretrained model
(same seed) and see the identical forget/retain split, so differences are
attributable only to the strategy-selection mechanism under test.

Run:
    python -m unlearning_meta.run_baseline_comparison
"""

from __future__ import annotations
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unlearning_meta.config import Config
from unlearning_meta.models.base_model import MLP
from unlearning_meta.data.dataset_utils import (
    load_digits_instance_forgetting,
    compute_retrain_baseline_forget_acc,
)
from unlearning_meta.engines.unlearning_engine import UnlearningEngine
from unlearning_meta.engines.experience_engine import ExperienceEngine
from unlearning_meta.utils.win_compat import enable_console_compat
from unlearning_meta.main import set_seed, pretrain


def build_base_config() -> Config:
    cfg = Config()
    cfg.model.input_dim, cfg.model.hidden_dims, cfg.model.output_dim = 64, [128, 64], 10
    cfg.training.num_cycles      = 25
    cfg.training.batch_size      = 32
    cfg.training.pretrain_epochs = 15
    cfg.training.pretrain_lr     = 0.002
    cfg.training.device          = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.seed            = 42

    cfg.unlearning.forget_lr                = 0.006
    cfg.unlearning.forget_epochs            = 2
    cfg.unlearning.gradient_ascent_strength = 0.5
    cfg.unlearning.noise_scale              = 0.004
    cfg.unlearning.ewc_lambda               = 300.0
    cfg.unlearning.fisher_samples           = 150
    cfg.unlearning.dream_replay_epochs      = 2
    cfg.unlearning.correction_epochs        = 3
    cfg.unlearning.eval_batch_size          = 64

    cfg.experience.exploration_rate       = 0.15
    cfg.experience.min_observations       = 2
    cfg.experience.rule_confidence_thresh = 0.50
    cfg.experience.pattern_window         = 15
    cfg.experience.reflection_window      = 8
    cfg.experience.use_contextual_bandit  = True   # §14/15's fixed mechanism
    return cfg


def run_condition(
    name:               str,
    base_model:         torch.nn.Module,
    cfg:                Config,
    baseline_forget_acc: float,
    forget_loader, retain_loader, test_loader,
    use_experience_engine: bool,
    disable_rule_extraction: bool = False,
    disable_strategy_evolution: bool = False,
) -> dict:
    """Run 25 cycles for one baseline/ablation condition."""
    device = cfg.training.device
    model = copy.deepcopy(base_model).to(device)

    ue_cfg = copy.deepcopy(cfg.unlearning)
    ue = UnlearningEngine(model=model, config=ue_cfg, device=device)

    ee = None
    if use_experience_engine:
        exp_cfg = copy.deepcopy(cfg.experience)
        exp_cfg.target_forget_acc = baseline_forget_acc
        exp_cfg.retain_floor      = 0.90
        ee = ExperienceEngine(exp_cfg, cfg.training.score_weights, seed=cfg.training.seed)
        if disable_rule_extraction:
            # Neutralise extraction rather than deleting the module, so
            # everything else (journal, reflection, bandit) behaves
            # identically — only the rule-prior pathway is cut off.
            ee.extractor.min_observations = 10 ** 9
        if disable_strategy_evolution:
            ee.evolver.max_rounds = 0

    forget_accs, retain_accs, scores = [], [], []
    for cycle in range(cfg.training.num_cycles):
        m = ue.run_cycle(forget_loader, retain_loader, test_loader, cycle=cycle)
        if ee is not None:
            proposal = ee.process(m)
            ue.apply_strategy(proposal)
            score = m.composite_score(cfg.training.score_weights,
                                      target_forget_acc=baseline_forget_acc,
                                      retain_floor=0.90)
        else:
            score = m.composite_score(cfg.training.score_weights,
                                      target_forget_acc=baseline_forget_acc,
                                      retain_floor=0.90)
        forget_accs.append(m.forget_acc)
        retain_accs.append(m.retain_acc)
        scores.append(score)

    deviations = [abs(f - baseline_forget_acc) for f in forget_accs]
    return {
        "name": name,
        "final_forget_acc": forget_accs[-1],
        "min_deviation":    min(deviations),
        "max_deviation":    max(deviations),
        "mean_deviation":   sum(deviations) / len(deviations),
        "min_retain_acc":   min(retain_accs),
        "final_retain_acc": retain_accs[-1],
        "mean_score":       sum(scores) / len(scores),
        "final_score":      scores[-1],
        "n_rules_extracted": len(ee.extractor.extracted_rules) if ee else 0,
        "n_merges":          len(ee.evolver.merge_log) if ee else 0,
    }


def main() -> None:
    enable_console_compat()
    cfg = build_base_config()
    set_seed(cfg.training.seed)
    device = cfg.training.device

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  BASELINES & ABLATIONS — instance-level benchmark            ║")
    print("║  §11.5 backlog item, run on the §16-calibrated evaluation    ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    forget_loader, retain_loader, test_loader, full_loader = \
        load_digits_instance_forgetting(forget_fraction=0.10,
                                         batch_size=cfg.training.batch_size,
                                         seed=cfg.training.seed)

    print("Pre-training the shared starting model (identical θ_0 for all conditions)...")
    base_model = MLP(cfg.model.input_dim, cfg.model.hidden_dims, cfg.model.output_dim,
                      cfg.model.dropout_rate, cfg.model.use_batch_norm).to(device)
    pretrain(base_model, full_loader, epochs=cfg.training.pretrain_epochs,
             lr=cfg.training.pretrain_lr, device=device, verbose=False)

    baseline_forget_acc = compute_retrain_baseline_forget_acc(
        forget_loader, retain_loader,
        model_factory=lambda: MLP(64, [128, 64], 10),
        epochs=cfg.training.pretrain_epochs, lr=cfg.training.pretrain_lr,
        device=device, seed=cfg.training.seed,
    )
    print(f"Retrain-from-scratch gold standard: {baseline_forget_acc:.4f}\n")

    conditions = [
        dict(name="A_fixed_hyperparameters", use_experience_engine=False),
        dict(name="B_full_experience_engine", use_experience_engine=True),
        dict(name="C_no_rule_extraction", use_experience_engine=True,
             disable_rule_extraction=True),
        dict(name="D_no_strategy_evolution", use_experience_engine=True,
             disable_strategy_evolution=True),
    ]

    results = []
    for cond in conditions:
        print(f"Running {cond['name']}...")
        r = run_condition(base_model=base_model, cfg=cfg,
                          baseline_forget_acc=baseline_forget_acc,
                          forget_loader=forget_loader, retain_loader=retain_loader,
                          test_loader=test_loader, **cond)
        results.append(r)
        print(f"    mean_score={r['mean_score']:.4f}  "
              f"mean_deviation={r['mean_deviation']:.4f}  "
              f"min_retain={r['min_retain_acc']:.4f}  "
              f"rules={r['n_rules_extracted']}  merges={r['n_merges']}")

    print("\n" + "═" * 88)
    print("COMPARISON")
    print("═" * 88)
    print(f"  Retrain-baseline forget_acc: {baseline_forget_acc:.4f} (target)\n")
    header = (f"  {'Condition':<26} {'mean_score':>10} {'mean_dev':>9} "
              f"{'min_dev':>8} {'min_retain':>10} {'rules':>6} {'merges':>7}")
    print(header)
    print("  " + "-" * 84)
    for r in sorted(results, key=lambda x: -x["mean_score"]):
        print(f"  {r['name']:<26} {r['mean_score']:>10.4f} {r['mean_deviation']:>9.4f} "
              f"{r['min_deviation']:>8.4f} {r['min_retain_acc']:>10.4f} "
              f"{r['n_rules_extracted']:>6} {r['n_merges']:>7}")

    print("\n" + "═" * 88)
    print("INTERPRETATION")
    print("═" * 88)
    by_name = {r["name"]: r for r in results}
    a, b = by_name["A_fixed_hyperparameters"], by_name["B_full_experience_engine"]
    c, d = by_name["C_no_rule_extraction"], by_name["D_no_strategy_evolution"]

    print(f"  Full system vs fixed-hyperparameter baseline: "
          f"mean_score {b['mean_score']:+.4f} vs {a['mean_score']:+.4f} "
          f"(Δ={b['mean_score']-a['mean_score']:+.4f})")
    print(f"  Full system vs no-rule-extraction ablation:   "
          f"mean_score {b['mean_score']:+.4f} vs {c['mean_score']:+.4f} "
          f"(Δ={b['mean_score']-c['mean_score']:+.4f})")
    print(f"  Full system vs no-strategy-evolution ablation: "
          f"mean_score {b['mean_score']:+.4f} vs {d['mean_score']:+.4f} "
          f"(Δ={b['mean_score']-d['mean_score']:+.4f})")

    return results


if __name__ == "__main__":
    main()
