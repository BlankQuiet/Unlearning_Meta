"""
run_ablation_study.py
=======================
2x2 factorial ablation study, directly following up on ARCHITECTURE.md
§12.3's mechanistic hypothesis for why instance-level forgetting showed
persistent under-forgetting (forget_acc stuck near 1.0 for 25 cycles
despite increase_forgetting_strength being applied 18/25 times).

Hypothesis under test
-----------------------
Correction Replay and/or EWC Consolidation — both tuned against the easy
whole-class benchmark, where forget-set and retain-set gradients point in
largely orthogonal directions — are strong enough on the entangled
instance-level task to fully neutralise each cycle's gradient-ascent
forgetting progress.

Design
-------
A single pretrained model (fixed seed) is deep-copied into four identical
starting points, then each runs 25 cycles of UnlearningEngine.run_cycle()
with FIXED hyperparameters (no ExperienceEngine / no strategy adaptation)
so the comparison isolates the mechanism's effect, not confounded by
strategy-selection dynamics:

  A) Full system         : correction_replay=ON,  ewc=ON   (replicates §12)
  B) No correction replay: correction_replay=OFF, ewc=ON
  C) No EWC              : correction_replay=ON,  ewc=OFF
  D) Neither              : correction_replay=OFF, ewc=OFF

Run:
    python -m unlearning_meta.run_ablation_study
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
from unlearning_meta.utils.win_compat import enable_console_compat
from unlearning_meta.main import set_seed, pretrain


CONDITIONS = [
    ("A_full_system",       True,  True),
    ("B_no_correction",     False, True),
    ("C_no_ewc",            True,  False),
    ("D_neither",           False, False),
]


def build_base_config() -> Config:
    cfg = Config()
    cfg.model.input_dim, cfg.model.hidden_dims, cfg.model.output_dim = 64, [128, 64], 10
    cfg.training.num_cycles      = 25
    cfg.training.batch_size      = 32
    cfg.training.pretrain_epochs = 15
    cfg.training.pretrain_lr     = 0.002
    cfg.training.device          = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.seed            = 42

    # Identical to run_instance_level_experiment.py's hyperparameters —
    # only correction_replay_enabled / ewc_enabled vary across conditions.
    cfg.unlearning.forget_lr                = 0.006
    cfg.unlearning.forget_epochs            = 2
    cfg.unlearning.gradient_ascent_strength = 0.5
    cfg.unlearning.noise_scale              = 0.004
    cfg.unlearning.ewc_lambda               = 300.0
    cfg.unlearning.fisher_samples           = 150
    cfg.unlearning.dream_replay_epochs      = 2
    cfg.unlearning.correction_epochs        = 3
    cfg.unlearning.eval_batch_size          = 64
    return cfg


def run_condition(
    name: str,
    correction_enabled: bool,
    ewc_enabled: bool,
    base_model: torch.nn.Module,
    forget_loader, retain_loader, test_loader,
    cfg: Config,
) -> dict:
    """Run 25 fixed-hyperparameter cycles for one ablation condition."""
    device = cfg.training.device
    model = copy.deepcopy(base_model).to(device)   # identical starting point every time

    condition_cfg = copy.deepcopy(cfg.unlearning)
    condition_cfg.correction_replay_enabled = correction_enabled
    condition_cfg.ewc_enabled               = ewc_enabled

    engine = UnlearningEngine(model=model, config=condition_cfg, device=device)

    forget_traj, retain_traj = [], []
    for cycle in range(cfg.training.num_cycles):
        m = engine.run_cycle(forget_loader, retain_loader, test_loader, cycle=cycle)
        forget_traj.append(m.forget_acc)
        retain_traj.append(m.retain_acc)
        # NOTE: no experience_engine.process() / apply_strategy() call —
        # hyperparameters are held fixed throughout, deliberately, so this
        # isolates the mechanism's effect rather than confounding it with
        # strategy-adaptation dynamics.

    return {
        "name": name,
        "correction_enabled": correction_enabled,
        "ewc_enabled": ewc_enabled,
        "forget_traj": forget_traj,
        "retain_traj": retain_traj,
        "final_forget_acc": forget_traj[-1],
        "min_forget_acc": min(forget_traj),
        "final_retain_acc": retain_traj[-1],
        "min_retain_acc": min(retain_traj),
    }


def main() -> None:
    enable_console_compat()
    cfg = build_base_config()
    set_seed(cfg.training.seed)
    device = cfg.training.device

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  ABLATION STUDY — isolating the under-forgetting cause      ║")
    print("║  2×2: Correction Replay × EWC, on instance-level forgetting ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    # ── Shared setup: one pretrained model, one forget/retain split ─────────
    forget_loader, retain_loader, test_loader, full_loader = \
        load_digits_instance_forgetting(forget_fraction=0.10,
                                         batch_size=cfg.training.batch_size,
                                         seed=cfg.training.seed)

    print("Pre-training the shared starting model (θ_0 for all 4 conditions)...")
    base_model = MLP(cfg.model.input_dim, cfg.model.hidden_dims, cfg.model.output_dim,
                      cfg.model.dropout_rate, cfg.model.use_batch_norm).to(device)
    pretrain(base_model, full_loader, epochs=cfg.training.pretrain_epochs,
             lr=cfg.training.pretrain_lr, device=device, verbose=False)

    base_model.eval()
    with torch.no_grad():
        correct = total = 0
        for x, y in forget_loader:
            x, y = x.to(device), y.to(device)
            correct += (base_model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    pre_forget_acc = correct / max(total, 1)

    baseline = compute_retrain_baseline_forget_acc(
        forget_loader, retain_loader,
        model_factory=lambda: MLP(64, [128, 64], 10),
        epochs=cfg.training.pretrain_epochs, lr=cfg.training.pretrain_lr,
        device=device, seed=cfg.training.seed,
    )

    print(f"Pre-unlearning forget_acc (shared θ_0)  : {pre_forget_acc:.4f}")
    print(f"Retrain-from-scratch gold standard       : {baseline:.4f}\n")

    # ── Run all 4 conditions from the identical starting point ──────────────
    results = []
    for name, corr, ewc in CONDITIONS:
        print(f"Running condition {name}  "
              f"(correction_replay={'ON' if corr else 'OFF'}, "
              f"EWC={'ON' if ewc else 'OFF'})...")
        r = run_condition(name, corr, ewc, base_model,
                          forget_loader, retain_loader, test_loader, cfg)
        results.append(r)
        print(f"    -> final forget_acc={r['final_forget_acc']:.4f}  "
              f"min forget_acc={r['min_forget_acc']:.4f}  "
              f"final retain_acc={r['final_retain_acc']:.4f}  "
              f"min retain_acc={r['min_retain_acc']:.4f}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("ABLATION RESULTS")
    print("═" * 78)
    print(f"  Pre-unlearning forget_acc : {pre_forget_acc:.4f}")
    print(f"  Retrain-baseline forget_acc: {baseline:.4f}  (target — NOT 0.0)\n")
    print(f"  {'Condition':<18} {'Correction':>10} {'EWC':>6} "
          f"{'final F':>9} {'min F':>8} {'ΔF (drop)':>10} {'min R':>7}")
    print("  " + "-" * 74)
    for r in results:
        delta = pre_forget_acc - r["min_forget_acc"]
        print(f"  {r['name']:<18} {str(r['correction_enabled']):>10} "
              f"{str(r['ewc_enabled']):>6} {r['final_forget_acc']:>9.4f} "
              f"{r['min_forget_acc']:>8.4f} {delta:>10.4f} {r['min_retain_acc']:>7.4f}")

    # ── Interpretation ────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("INTERPRETATION")
    print("═" * 78)
    by_name = {r["name"]: r for r in results}
    drop_full   = pre_forget_acc - by_name["A_full_system"]["min_forget_acc"]
    drop_nocorr = pre_forget_acc - by_name["B_no_correction"]["min_forget_acc"]
    drop_noewc  = pre_forget_acc - by_name["C_no_ewc"]["min_forget_acc"]
    drop_neither= pre_forget_acc - by_name["D_neither"]["min_forget_acc"]

    print("  Forgetting progress (drop in forget_acc from pre-unlearning):")
    print(f"    Full system (A)        : {drop_full:.4f}")
    print(f"    No correction (B)      : {drop_nocorr:.4f}  "
          f"({'+' if drop_nocorr > drop_full else ''}{drop_nocorr-drop_full:+.4f} vs A)")
    print(f"    No EWC (C)             : {drop_noewc:.4f}  "
          f"({'+' if drop_noewc > drop_full else ''}{drop_noewc-drop_full:+.4f} vs A)")
    print(f"    Neither (D)            : {drop_neither:.4f}  "
          f"({'+' if drop_neither > drop_full else ''}{drop_neither-drop_full:+.4f} vs A)")

    contributors = []
    if drop_nocorr > drop_full + 0.02:
        contributors.append("Correction Replay")
    if drop_noewc > drop_full + 0.02:
        contributors.append("EWC")
    if contributors:
        print(f"\n  -> Removing {' and '.join(contributors)} measurably increased "
              f"forgetting progress relative to the full system, supporting "
              f"the §12.3 hypothesis that it neutralises gradient-ascent progress.")
    else:
        msg = (
            "\n  -> Neither ablation meaningfully increased forgetting progress "
            "on its own. The §12.3 hypothesis is NOT well supported by this "
            "result — the flat forget_acc trajectory likely has a different "
            "cause (e.g. gradient_ascent_strength / forget_lr too weak for "
            "this task's tiny 139-sample forget set relative to the "
            "1,298-sample retain set, independent of correction/EWC)."
        )
        print(msg)

    return results


if __name__ == "__main__":
    main()
