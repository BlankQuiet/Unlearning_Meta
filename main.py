"""
main.py
=======
Entry point for the Unlearning-MetaLearning Research Prototype.

Run::
    python -m unlearning_meta.main

Architecture overview::

    ┌─────────────────────────────────────────────────────────────┐
    │  Training Loop                                               │
    │                                                             │
    │  for cycle in 1..N:                                         │
    │    metrics  = UnlearningEngine.run_cycle(...)    ← weights  │
    │    proposal = ExperienceEngine.process(metrics)  ← no wts  │
    │    UnlearningEngine.apply_strategy(proposal)                │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from unlearning_meta.config import Config
from unlearning_meta.models.base_model import MLP
from unlearning_meta.data.dataset_utils import load_unlearning_datasets
from unlearning_meta.engines.unlearning_engine import UnlearningEngine
from unlearning_meta.engines.experience_engine import ExperienceEngine
from unlearning_meta.evaluation.metrics import MetricsTracker
from unlearning_meta.evaluation.mia_evaluator import MIAEvaluator
from unlearning_meta.utils.logger import ExperimentLogger
from unlearning_meta.utils.visualization import generate_dashboard
from unlearning_meta.utils.win_compat import enable_console_compat


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────────────────────────────────────
# Pre-training
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(
    model:       nn.Module,
    full_loader: DataLoader,
    epochs:      int   = 5,
    lr:          float = 0.001,
    device:      str   = "cpu",
    verbose:     bool  = True,
) -> nn.Module:
    """Train the model on all data (forget + retain) before unlearning."""
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        correct = total = 0
        for x, y in full_loader:
            if x.size(0) < 2:
                # BatchNorm1d requires ≥2 samples per batch in train() mode.
                continue
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)

        if verbose:
            print(f"  Pre-train epoch {epoch}/{epochs}  acc={correct/total:.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(config: Config) -> MetricsTracker:
    """
    Full experiment: pre-train → unlearning loop → evaluation → report.

    Returns:
        MetricsTracker with all cycle records.
    """
    # Must run before any print()/logging call below. Placed here (not
    # only in the CLI __main__ guard) so this also protects the
    # documented direct-import usage pattern in README.md's Quick Start
    # (`from unlearning_meta.main import run_experiment`), which never
    # passes through __main__ at all.
    enable_console_compat()

    set_seed(config.training.seed)
    os.makedirs(config.training.save_dir, exist_ok=True)
    os.makedirs(config.training.log_dir,  exist_ok=True)

    device = config.training.device
    logger = ExperimentLogger(
        log_dir = config.training.log_dir,
        verbose = config.training.verbose,
        name    = "unlearning_exp",
    )
    tracker = MetricsTracker(score_weights=config.training.score_weights)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.section("Data Loading")
    forget_loader, retain_loader, test_loader, full_loader = \
        load_unlearning_datasets(
            data_dir       = config.training.data_dir,
            forget_classes = config.training.forget_classes,
            batch_size     = config.training.batch_size,
            seed           = config.training.seed,
        )

    # ── 2. Build and pre-train model ─────────────────────────────────────────
    logger.section("Pre-Training")
    model = MLP(
        input_dim   = config.model.input_dim,
        hidden_dims = config.model.hidden_dims,
        output_dim  = config.model.output_dim,
        dropout_rate = config.model.dropout_rate,
        use_batch_norm = config.model.use_batch_norm,
    ).to(device)

    print(f"  Model: {model}")
    pretrain(
        model, full_loader,
        epochs  = config.training.pretrain_epochs,
        lr      = config.training.pretrain_lr,
        device  = device,
        verbose = config.training.verbose,
    )

    # ── 3. Initialise engines ─────────────────────────────────────────────────
    logger.section("Initialising Engines")
    unlearning_engine = UnlearningEngine(
        model  = model,
        config = config.unlearning,
        device = device,
    )
    experience_engine = ExperienceEngine(
        config        = config.experience,
        score_weights = config.training.score_weights,
        seed          = config.training.seed,
    )
    print(f"  UnlearningEngine ready  device={device}")
    print(f"  ExperienceEngine ready  exploration_rate={config.experience.exploration_rate}")

    # ── 4. Unlearning loop ────────────────────────────────────────────────────
    logger.section("Unlearning Loop")
    print(
        f"\n  {'Cycle':>6} {'Score':>7} {'Forget':>7} {'Retain':>7} "
        f"{'Acc':>7} {'Priv':>7} {'Hall':>7} {'Stab':>7}  Strategy\n"
        f"  {'─'*72}"
    )

    proposal = None  # No proposal for cycle 0

    for cycle in range(config.training.num_cycles):
        t0 = time.time()

        # ── 4a. Unlearning Engine: run one cycle ───────────────────────────
        metrics = unlearning_engine.run_cycle(
            forget_loader = forget_loader,
            retain_loader = retain_loader,
            test_loader   = test_loader,
            cycle         = cycle,
        )

        # ── 4b. Experience Engine: process metrics, emit proposal ──────────
        # CRITICAL: Experience Engine receives ONLY metrics (no model access)
        proposal = experience_engine.process(metrics)

        # ── 4c. Unlearning Engine: apply strategy proposal ─────────────────
        applied = unlearning_engine.apply_strategy(proposal)

        # ── 4d. Record & log ───────────────────────────────────────────────
        cfg_snap = unlearning_engine.config_snapshot()
        rec      = tracker.record(metrics, proposal, cfg_snap)

        logger.log_cycle(
            cycle       = cycle,
            metrics     = metrics,
            strategy    = proposal,
            score       = rec.score,
            explanation = experience_engine.journal.last_n(1)[0].explanation
                          if experience_engine.journal.last_n(1) else "",
            extra       = {
                "cycle_duration_s": round(time.time() - t0, 2),
                "strategy_applied": applied,
            },
        )

        # ── 4e. Periodic checkpoint ───────────────────────────────────────
        if cycle > 0 and cycle % 10 == 0:
            ckpt_path = os.path.join(
                config.training.save_dir, f"checkpoint_{cycle:03d}.pt"
            )
            unlearning_engine.save_checkpoint(ckpt_path)
            logger.log_event("CHECKPOINT", f"Saved → {ckpt_path}")

    # ── 5. Final evaluation ───────────────────────────────────────────────────
    logger.section("Final MIA Evaluation")
    mia_evaluator = MIAEvaluator(device=device)
    mia_report = mia_evaluator.evaluate(
        model          = model,
        forget_loader  = forget_loader,
        retain_loader  = retain_loader,
        test_loader    = test_loader,
    )
    print(f"  Confidence MIA  privacy={mia_report['confidence_mia']['privacy_score']:.4f}  "
          f"AUC={mia_report['confidence_mia']['auc']:.4f}")
    print(f"  Shadow MIA      privacy={mia_report['shadow_mia']['privacy_score']:.4f}  "
          f"AUC={mia_report['shadow_mia']['auc']:.4f}")
    print(f"  Final privacy   score  ={mia_report['final_privacy']:.4f}")

    # ── 6. Summary table ──────────────────────────────────────────────────────
    logger.section("Results Summary")
    tracker.print_table(last_n=15)
    summary = tracker.summary()
    print(f"\n  Best cycle      : {summary['best_cycle']}")
    print(f"  Best score      : {summary['best_score']:.4f}")
    print(f"  Convergence at  : cycle {tracker.convergence_cycle()}")

    # ── 7. Experience Engine report ───────────────────────────────────────────
    exp_report = experience_engine.report()
    logger.log_report(exp_report)

    print("\n  Top-5 Extracted Rules:")
    for r in experience_engine.top_rules(5):
        print(f"    [{r['id']}] {r['rule'][:70]}  "
              f"success={r['success_rate']:.2f}  conf={r['confidence']:.2f}")

    # ── 8. Save artefacts ─────────────────────────────────────────────────────
    logger.section("Saving Artefacts")
    tracker.to_csv(os.path.join(config.training.save_dir, "metrics.csv"))
    tracker.to_jsonl(os.path.join(config.training.save_dir, "metrics.jsonl"))

    # ── 9. Generate plots ─────────────────────────────────────────────────────
    try:
        plot_paths = generate_dashboard(
            tracker      = tracker,
            memory_stats = exp_report["memory_stats"],
            save_dir     = config.training.save_dir,
        )
        for p in plot_paths:
            logger.log_event("PLOT", p)
    except Exception as e:
        logger.log_event("PLOT_WARN", f"Visualisation failed: {e}")

    # ── 10. Final save ─────────────────────────────────────────────────────────
    final_ckpt = os.path.join(config.training.save_dir, "final_model.pt")
    unlearning_engine.save_checkpoint(final_ckpt)
    logger.log_event("FINAL", f"Model saved → {final_ckpt}")

    logger.close()
    return tracker


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unlearning-MetaLearning Research Prototype"
    )
    p.add_argument("--cycles",         type=int,   default=25,
                   help="Number of unlearning cycles")
    p.add_argument("--forget-classes", type=int,   nargs="+", default=[0],
                   help="Classes to forget (e.g. 0 1)")
    p.add_argument("--ewc-lambda",     type=float, default=400.0)
    p.add_argument("--ga-strength",    type=float, default=1.0,
                   help="Gradient ascent strength")
    p.add_argument("--device",         type=str,   default=None,
                   help="cpu | cuda (auto-detect if omitted)")
    p.add_argument("--save-dir",       type=str,   default="./outputs")
    p.add_argument("--data-dir",       type=str,   default="./data")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--quiet",          action="store_true")
    p.add_argument("--threads",        type=int,   default=None,
                   help="torch.set_num_threads() override. On a modest "
                        "CPU-only laptop, leaving this unset (PyTorch "
                        "auto-detects core count) is usually best; set "
                        "explicitly (e.g. --threads 4) only if you "
                        "notice CPU contention with other running apps.")
    p.add_argument("--low-memory",     action="store_true",
                   help="Use Config.for_low_memory_cpu() preset (smaller "
                        "model/batch size, tuned for ~8GB-RAM GPU-less "
                        "machines). See config.py for the documented "
                        "memory budget this preset targets.")
    return p.parse_args()


def main() -> None:
    enable_console_compat()   # before the very first print() below
    args = parse_args()
    cfg  = Config.for_low_memory_cpu() if args.low_memory else Config()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    # Apply CLI overrides (on top of whichever base preset was selected)
    cfg.training.num_cycles     = args.cycles
    cfg.training.forget_classes = args.forget_classes
    cfg.training.save_dir       = args.save_dir
    cfg.training.data_dir       = args.data_dir
    cfg.training.seed           = args.seed
    cfg.training.verbose        = not args.quiet

    cfg.unlearning.ewc_lambda              = args.ewc_lambda
    cfg.unlearning.gradient_ascent_strength = args.ga_strength

    if args.device:
        cfg.training.device = args.device

    print("╔═══════════════════════════════════════════════════════╗")
    print("║  Unlearning-MetaLearning Research Prototype           ║")
    print("║  Layer 1: Unlearning Engine  (owns all weight updates) ║")
    print("║  Layer 2: Experience Engine  (proposals only, no wts) ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print(f"  device={cfg.training.device}  cycles={cfg.training.num_cycles}  "
          f"forget={cfg.training.forget_classes}  "
          f"preset={'low_memory_cpu' if args.low_memory else 'default'}\n")

    run_experiment(cfg)


if __name__ == "__main__":
    main()
