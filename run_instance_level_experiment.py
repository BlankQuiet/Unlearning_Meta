"""
run_instance_level_experiment.py
==================================
Validation run on the HARDER instance-level forgetting benchmark
(external review round 2, Q5), rather than the whole-class benchmark
used in run_real_experiment.py.

Key methodological difference from run_real_experiment.py
-------------------------------------------------------------
For whole-class forgetting, forget_acc -> 0 is the correct target.
For instance-level forgetting it is NOT: a model that never saw the
forgotten instances still classifies most of them correctly from general
class knowledge alone (see the retrain-from-scratch baseline computed
below). This script therefore reports forget_acc *relative to that
baseline*, not relative to 0, and explicitly flags over-forgetting
(driving forget_acc below the baseline) as a failure mode alongside
under-forgetting (forget_acc staying too close to the pre-unlearning
value).

Run:
    python -m unlearning_meta.run_instance_level_experiment
"""

from __future__ import annotations
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
from unlearning_meta.evaluation.metrics import MetricsTracker
from unlearning_meta.evaluation.mia_evaluator import MIAEvaluator
from unlearning_meta.utils.logger import ExperimentLogger
from unlearning_meta.utils.win_compat import enable_console_compat
from unlearning_meta.main import set_seed, pretrain


def build_config() -> Config:
    cfg = Config()
    cfg.model.input_dim, cfg.model.hidden_dims, cfg.model.output_dim = 64, [128, 64], 10

    cfg.training.num_cycles      = 25
    cfg.training.batch_size      = 32
    cfg.training.pretrain_epochs = 15
    cfg.training.pretrain_lr     = 0.002
    cfg.training.device          = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.seed            = 42
    cfg.training.save_dir        = "./outputs_instance"
    cfg.training.log_dir         = "./logs_instance"

    # Same forgetting hyperparameters as the whole-class run for direct
    # comparability — the point of this script is to isolate the effect of
    # task difficulty, not to also change the unlearning hyperparameters.
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

    return cfg


def main() -> None:
    enable_console_compat()
    cfg = build_config()
    set_seed(cfg.training.seed)
    os.makedirs(cfg.training.save_dir, exist_ok=True)
    os.makedirs(cfg.training.log_dir, exist_ok=True)
    device = cfg.training.device

    logger = ExperimentLogger(log_dir=cfg.training.log_dir, verbose=True,
                               name="instance_level_exp")

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  INSTANCE-LEVEL FORGETTING — harder benchmark (review Q5)  ║")
    print("║  Forgets 10% of EVERY class, not one whole class            ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  device={device}  cycles={cfg.training.num_cycles}\n")

    # ── 1. Load the harder, entangled benchmark ─────────────────────────────
    logger.section("Data Loading (instance-level, 10% of every class)")
    forget_loader, retain_loader, test_loader, full_loader = \
        load_digits_instance_forgetting(
            forget_fraction=0.10, batch_size=cfg.training.batch_size,
            seed=cfg.training.seed,
        )

    # ── 2. Retrain-from-scratch gold standard ────────────────────────────────
    logger.section("Computing Retrain-From-Scratch Gold Standard")
    print("  Training a model that NEVER sees the forget set, for comparison...")
    baseline_forget_acc = compute_retrain_baseline_forget_acc(
        forget_loader, retain_loader,
        model_factory=lambda: MLP(cfg.model.input_dim, cfg.model.hidden_dims,
                                   cfg.model.output_dim, cfg.model.dropout_rate,
                                   cfg.model.use_batch_norm),
        epochs=cfg.training.pretrain_epochs, lr=cfg.training.pretrain_lr,
        device=device, seed=cfg.training.seed,
    )
    print(f"  → Gold-standard forget_acc = {baseline_forget_acc:.4f}  "
          f"(the TARGET for a good unlearning method — NOT 0.0)")

    # ARCHITECTURE.md §16: wire the just-computed baseline into the actual
    # decision-making machinery. Without this, every score-driven mechanism
    # (bandit momentum, rule outcome tracking, meta UCB updates) silently
    # targets forget_acc=0 regardless of what this script prints — the
    # baseline being computed and displayed above is not the same thing as
    # it being used. retain_floor is a companion guardrail: catastrophic
    # retain damage should never be "worth it" for a forget-score gain.
    cfg.experience.target_forget_acc = baseline_forget_acc
    cfg.experience.retain_floor      = 0.90
    tracker = MetricsTracker(
        score_weights      = cfg.training.score_weights,
        target_forget_acc  = baseline_forget_acc,
        retain_floor       = 0.90,
    )

    # ── 3. Pre-train the actual model to be unlearned ────────────────────────
    logger.section("Pre-Training (learns everything, including forget instances)")
    model = MLP(cfg.model.input_dim, cfg.model.hidden_dims, cfg.model.output_dim,
                cfg.model.dropout_rate, cfg.model.use_batch_norm).to(device)
    pretrain(model, full_loader, epochs=cfg.training.pretrain_epochs,
             lr=cfg.training.pretrain_lr, device=device, verbose=True)

    model.eval()
    with torch.no_grad():
        correct = total = 0
        for x, y in forget_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    pre_forget_acc = correct / max(total, 1)
    print(f"  → Pre-unlearning forget_acc = {pre_forget_acc:.4f}  "
          f"(model has memorised the forget instances specifically)")

    # ── 4. Dual-engine unlearning loop ────────────────────────────────────────
    logger.section("Dual-Engine Unlearning Loop")
    unlearning_engine = UnlearningEngine(model=model, config=cfg.unlearning, device=device)
    experience_engine = ExperienceEngine(config=cfg.experience,
                                          score_weights=cfg.training.score_weights,
                                          seed=cfg.training.seed)

    print(f"\n  {'Cyc':>4} {'forget_acc':>10} {'vs_baseline':>11} {'retain':>7} "
          f"{'acc':>6} {'priv':>6}  Strategy")
    print(f"  {'─'*80}")

    for cycle in range(cfg.training.num_cycles):
        metrics  = unlearning_engine.run_cycle(forget_loader, retain_loader,
                                                test_loader, cycle=cycle)
        proposal = experience_engine.process(metrics)
        applied  = unlearning_engine.apply_strategy(proposal)

        rec = tracker.record(metrics, proposal, unlearning_engine.config_snapshot())
        gap = metrics.forget_acc - baseline_forget_acc
        gap_str = f"{gap:+.3f}"
        action_str = proposal.action.value[:22] if not proposal.is_noop() else "noop"
        print(f"  {cycle:>4} {metrics.forget_acc:>10.4f} {gap_str:>11} "
              f"{metrics.retain_acc:>7.3f} {metrics.accuracy:>6.3f} "
              f"{metrics.privacy:>6.3f}  {action_str}")

        expl = experience_engine.journal.last_n(1)[0].explanation \
               if experience_engine.journal.last_n(1) else ""
        logger.log_cycle(cycle, metrics, proposal, rec.score, explanation=expl,
                          extra={"strategy_applied": applied,
                                 "baseline_gap": gap,
                                 "cumulative_drift": metrics.extra.get("cumulative_drift")})

    # ── 5. Final analysis: did it converge to the RIGHT target? ───────────────
    logger.section("Results: Convergence Toward Gold-Standard Baseline")
    final = tracker.records[-1].metrics
    final_gap = final.forget_acc - baseline_forget_acc

    print(f"  Gold-standard baseline forget_acc : {baseline_forget_acc:.4f}")
    print(f"  Pre-unlearning forget_acc         : {pre_forget_acc:.4f}")
    print(f"  Final forget_acc (this run)       : {final.forget_acc:.4f}")
    print(f"  Final gap vs. gold standard       : {final_gap:+.4f}")
    print()
    if final.forget_acc < baseline_forget_acc - 0.10:
        print("  ⚠ OVER-FORGETTING: final forget_acc is well below what a "
              "retrain-from-scratch model would achieve. The system is "
              "suppressing correct predictions it has every right to make "
              "from general class knowledge — this is a failure mode "
              "specific to instance-level forgetting that whole-class "
              "forgetting cannot even exhibit.")
    elif final.forget_acc > pre_forget_acc - 0.10:
        print("  ⚠ UNDER-FORGETTING: forget_acc barely moved from its "
              "pre-unlearning value — the specific forgotten instances "
              "are still largely being recognised.")
    else:
        print("  ✓ Converged into a reasonable range between the "
              "pre-unlearning value and the gold-standard baseline.")

    print()
    print("  Did headline forget_acc alone reach 0 within a few cycles, the "
          "way it did on the whole-class benchmark? "
          f"{'YES' if final.forget_acc < 0.05 else 'NO'} "
          f"(min across run: {min(r.metrics.forget_acc for r in tracker.records):.4f})")

    # ── 6. MIA — the metric that actually matters for this benchmark ─────────
    logger.section("MIA Evaluation (the primary signal for instance-level forgetting)")
    mia = MIAEvaluator(device=device, max_samples=200)
    mia_report = mia.evaluate(model, forget_loader, retain_loader, test_loader)
    print(f"  Confidence-MIA AUC={mia_report['confidence_mia']['auc']:.4f}  "
          f"privacy={mia_report['confidence_mia']['privacy_score']:.4f}")
    print(f"  Shadow-MIA      AUC={mia_report['shadow_mia']['auc']:.4f}  "
          f"privacy={mia_report['shadow_mia']['privacy_score']:.4f}")
    print(f"  → Final privacy score = {mia_report['final_privacy']:.4f}")

    tracker.print_table(last_n=25)

    # ── 7. Save ────────────────────────────────────────────────────────────
    tracker.to_csv(os.path.join(cfg.training.save_dir, "metrics_instance.csv"))
    tracker.to_jsonl(os.path.join(cfg.training.save_dir, "metrics_instance.jsonl"))
    unlearning_engine.save_checkpoint(
        os.path.join(cfg.training.save_dir, "final_model_instance.pt")
    )
    logger.close()
    print("\n  === INSTANCE-LEVEL VALIDATION RUN COMPLETE ===")
    return tracker, baseline_forget_acc, pre_forget_acc


if __name__ == "__main__":
    main()
