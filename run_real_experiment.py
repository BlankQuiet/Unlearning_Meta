"""
run_real_experiment.py
========================
End-to-end VALIDATION run on real structured data (sklearn digits, 8x8
handwritten digit images — a network-free MNIST substitute).

Unlike the smoke tests in main.py's docstring examples, this script
exercises the FULL dual-engine pipeline against genuinely learnable
patterns, so the resulting metrics demonstrate real unlearning dynamics:
  • forget_acc should fall from ~high (post pre-train) toward ~chance
  • retain_acc / accuracy should stay high throughout
  • privacy should rise as forget/test confidence distributions converge
  • the Experience Engine should extract genuine rules and evolve them

Run:
    python -m unlearning_meta.run_real_experiment
"""

from __future__ import annotations
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unlearning_meta.config import Config
from unlearning_meta.models.base_model import MLP
from unlearning_meta.data.dataset_utils import load_digits_unlearning_datasets
from unlearning_meta.engines.unlearning_engine import UnlearningEngine
from unlearning_meta.engines.experience_engine import ExperienceEngine
from unlearning_meta.evaluation.metrics import MetricsTracker
from unlearning_meta.evaluation.mia_evaluator import MIAEvaluator
from unlearning_meta.utils.logger import ExperimentLogger
from unlearning_meta.utils.visualization import generate_dashboard
from unlearning_meta.utils.win_compat import enable_console_compat
from unlearning_meta.main import set_seed, pretrain


def build_config() -> Config:
    cfg = Config()

    # Model sized for 8x8=64-dim digit images (not 784-dim MNIST)
    cfg.model.input_dim    = 64
    cfg.model.hidden_dims  = [128, 64]
    cfg.model.output_dim   = 10
    cfg.model.dropout_rate = 0.2

    cfg.training.num_cycles      = 25
    cfg.training.forget_classes  = [7]          # forget digit "7"
    cfg.training.batch_size      = 32
    cfg.training.pretrain_epochs = 15
    cfg.training.pretrain_lr     = 0.002
    cfg.training.device          = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.seed            = 42
    cfg.training.save_dir        = "./outputs_real"
    cfg.training.log_dir         = "./logs_real"

    # Gentler initial forgetting than the first validation pass: the previous
    # run showed forget_lr=0.02/strength=1.0 drives forget_acc to 0% within a
    # single cycle on this small dataset, leaving the Experience Engine no
    # gradient of forgetting-quality signal to learn from (it only ever sees
    # "already forgotten") and over-relies on repeated noise injection, which
    # spiked hallucination to 0.84-0.90. Slower forgetting gives the rule
    # extractor real threshold-crossing observations (forget_acc passing
    # through 0.40 → 0.25 → 0.15 across multiple cycles) and lets us observe
    # the engine learning to taper off noise/forgetting-strength once
    # forget_acc is already low — i.e. genuine increase→decrease adaptation.
    cfg.unlearning.forget_lr                = 0.006
    cfg.unlearning.forget_epochs            = 2
    cfg.unlearning.gradient_ascent_strength = 0.5
    cfg.unlearning.noise_scale              = 0.004
    cfg.unlearning.ewc_lambda               = 300.0
    cfg.unlearning.fisher_samples           = 150
    cfg.unlearning.dream_replay_epochs      = 2
    cfg.unlearning.correction_epochs        = 3
    cfg.unlearning.eval_batch_size          = 64

    cfg.experience.exploration_rate    = 0.15
    cfg.experience.min_observations    = 2
    cfg.experience.rule_confidence_thresh = 0.50
    cfg.experience.pattern_window      = 15
    cfg.experience.reflection_window   = 8

    return cfg


def main() -> None:
    enable_console_compat()   # before the very first print() below
    cfg = build_config()
    set_seed(cfg.training.seed)
    os.makedirs(cfg.training.save_dir, exist_ok=True)
    os.makedirs(cfg.training.log_dir, exist_ok=True)

    device = cfg.training.device
    logger = ExperimentLogger(log_dir=cfg.training.log_dir, verbose=True,
                               name="real_digits_exp")
    tracker = MetricsTracker(score_weights=cfg.training.score_weights)

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  REAL END-TO-END VALIDATION RUN                            ║")
    print("║  Data : sklearn digits (8x8 real handwritten digits)        ║")
    print("║  Task  : forget digit '7', retain digits 0-6,8,9            ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  device={device}  cycles={cfg.training.num_cycles}\n")

    # ── 1. Load real structured data ────────────────────────────────────────
    logger.section("Data Loading (sklearn digits — network-free)")
    forget_loader, retain_loader, test_loader, full_loader = \
        load_digits_unlearning_datasets(
            forget_classes = cfg.training.forget_classes,
            batch_size     = cfg.training.batch_size,
            seed           = cfg.training.seed,
        )

    # ── 2. Build + pre-train model ──────────────────────────────────────────
    logger.section("Pre-Training (learn ALL digits incl. forget class)")
    model = MLP(
        input_dim      = cfg.model.input_dim,
        hidden_dims    = cfg.model.hidden_dims,
        output_dim     = cfg.model.output_dim,
        dropout_rate   = cfg.model.dropout_rate,
        use_batch_norm = cfg.model.use_batch_norm,
    ).to(device)
    print(f"  {model}")

    pretrain(model, full_loader, epochs=cfg.training.pretrain_epochs,
             lr=cfg.training.pretrain_lr, device=device, verbose=True)

    # Pre-unlearning baseline accuracy on forget class specifically
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for x, y in forget_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    pre_forget_acc = correct / max(total, 1)
    print(f"  → Pre-unlearning forget_acc (digit 7) = {pre_forget_acc:.4f}  "
          f"(should be high — model knows digit 7 well)")

    # ── 3. Initialise dual engines ──────────────────────────────────────────
    logger.section("Initialising Dual-Engine Architecture")
    unlearning_engine = UnlearningEngine(model=model, config=cfg.unlearning, device=device)
    experience_engine = ExperienceEngine(
        config=cfg.experience, score_weights=cfg.training.score_weights,
        seed=cfg.training.seed,
    )

    # ── 4. Run the dual-engine unlearning loop ──────────────────────────────
    logger.section("Dual-Engine Unlearning Loop")
    print(f"\n  {'Cyc':>4} {'Score':>6} {'Forget':>7} {'Retain':>7} {'Acc':>6} "
          f"{'Priv':>6} {'Hall':>6} {'Stab':>6}  Strategy")
    print(f"  {'─'*84}")

    for cycle in range(cfg.training.num_cycles):
        metrics  = unlearning_engine.run_cycle(forget_loader, retain_loader,
                                                test_loader, cycle=cycle)
        proposal = experience_engine.process(metrics)      # NO weight access
        applied  = unlearning_engine.apply_strategy(proposal)

        rec = tracker.record(metrics, proposal, unlearning_engine.config_snapshot())
        expl = experience_engine.journal.last_n(1)[0].explanation \
               if experience_engine.journal.last_n(1) else ""
        logger.log_cycle(cycle, metrics, proposal, rec.score, explanation=expl,
                          extra={"strategy_applied": applied,
                                 "cumulative_drift": metrics.extra.get("cumulative_drift")})

    # ── 5. Final MIA evaluation ─────────────────────────────────────────────
    logger.section("Final MIA Evaluation (real data)")
    mia = MIAEvaluator(device=device, max_samples=300)
    mia_report = mia.evaluate(model, forget_loader, retain_loader, test_loader)
    print(f"  Confidence-MIA  AUC={mia_report['confidence_mia']['auc']:.4f}  "
          f"privacy={mia_report['confidence_mia']['privacy_score']:.4f}")
    print(f"  Shadow-MIA      AUC={mia_report['shadow_mia']['auc']:.4f}  "
          f"privacy={mia_report['shadow_mia']['privacy_score']:.4f}")
    print(f"  → Final privacy score = {mia_report['final_privacy']:.4f}")

    # ── 6. Results summary ──────────────────────────────────────────────────
    logger.section("Results Summary")
    tracker.print_table(last_n=20)
    summary = tracker.summary()

    first = tracker.records[0].metrics
    last  = tracker.records[-1].metrics
    print(f"\n  {'Metric':<14} {'Cycle 0':>10} {'Final':>10} {'Δ':>10}")
    print(f"  {'─'*46}")
    for name in ["forget_acc", "retain_acc", "accuracy", "privacy", "hallucination", "stability"]:
        v0, v1 = getattr(first, name), getattr(last, name)
        print(f"  {name:<14} {v0:>10.4f} {v1:>10.4f} {v1-v0:>+10.4f}")

    print(f"\n  Pre-unlearn forget_acc : {pre_forget_acc:.4f}")
    print(f"  Post-unlearn forget_acc: {last.forget_acc:.4f}  "
          f"(Δ = {last.forget_acc - pre_forget_acc:+.4f})")
    print(f"  Best composite score   : {summary['best_score']:.4f} @ cycle {summary['best_cycle']}")
    print(f"  Convergence cycle      : {tracker.convergence_cycle(window=4, tol=0.08)}")

    # ── 7. Experience Engine introspection ──────────────────────────────────
    logger.section("Experience Engine — Extracted Knowledge")
    exp_report = experience_engine.report()
    logger.log_report(exp_report)

    rules = experience_engine.top_rules(8)
    print(f"  Extracted rules ({len(experience_engine.extractor.extracted_rules)} total, top 8 shown):")
    if rules:
        for r in rules:
            print(f"    [{r['id']}] {r['rule'][:64]:<64} "
                  f"succ={r['success_rate']:.2f} conf={r['confidence']:.2f} "
                  f"used={r['usage']:>2} state={r['state']}")
    else:
        print("    (no rules crossed the confidence/observation threshold in this short run)")

    print(f"\n  Strategy memory lifecycle: {exp_report['memory_stats']}")
    print(f"  Rule-merge events        : {len(experience_engine.evolver.merge_log)}")
    for m in experience_engine.evolver.merge_log[:5]:
        print(f"    cycle {m['cycle']}: merged {len(m['merged'])} rules → {m['rule'][:60]}")

    strat_eff = tracker.strategy_effectiveness()
    print("\n  Per-strategy effectiveness (Δscore | improve-rate):")
    for action, stats in sorted(strat_eff.items(), key=lambda kv: -kv[1]["mean_delta"]):
        print(f"    {action:<32} n={stats['count']:>3}  "
              f"Δ={stats['mean_delta']:>+7.4f}  improve_rate={stats['improve_rate']:.2f}")

    # ── 8. Save artefacts ────────────────────────────────────────────────────
    logger.section("Saving Artefacts")
    tracker.to_csv(os.path.join(cfg.training.save_dir, "metrics_real.csv"))
    tracker.to_jsonl(os.path.join(cfg.training.save_dir, "metrics_real.jsonl"))

    try:
        plot_paths = generate_dashboard(
            tracker=tracker, memory_stats=exp_report["memory_stats"],
            save_dir=cfg.training.save_dir,
        )
        for p in plot_paths:
            logger.log_event("PLOT", p)
            print(f"  saved: {p}")
    except Exception as e:
        logger.log_event("PLOT_WARN", str(e))
        print(f"  plot generation warning: {e}")

    unlearning_engine.save_checkpoint(
        os.path.join(cfg.training.save_dir, "final_model_real.pt")
    )
    logger.close()

    print("\n  === REAL VALIDATION RUN COMPLETE ===")
    return tracker, experience_engine, mia_report


if __name__ == "__main__":
    main()
