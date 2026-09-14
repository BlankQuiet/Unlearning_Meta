# Unlearning Meta-Learning Research Prototype

A dual-engine cognitive architecture combining **Machine Unlearning** and **Meta-Learning**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Training Loop                              │
│                                                                  │
│  ┌────────────────────────────┐    UnlearningMetrics             │
│  │  Layer 1: Unlearning Engine│ ──────────────────────────────► │
│  │                            │    {forget_acc, retain_acc,      │
│  │  • Memory Trace Mapping    │     accuracy, privacy,           │
│  │  • Fisher Information      │     hallucination, stability}    │
│  │  • Gradient Ascent         │                                  │
│  │  • Dream Replay            │                    ┌──────────────────────────┐
│  │  • Correction Replay       │                    │ Layer 2: Experience Engine│
│  │  • EWC Consolidation       │                    │                          │
│  │  • Evaluation              │ ◄── StrategyUpdate─│  • Memory Journal        │
│  │                            │    {action, δ,      │  • Self Reflection       │
│  │  ✓ Owns model weights      │     rationale,      │  • Rule Extraction       │
│  └────────────────────────────┘     confidence}     │  • Strategy Memory       │
│                                                     │  • Strategy Evolution    │
│           ↑ apply_strategy()                        │  • Meta-Knowledge        │
│           (hyperparams only,                        │                          │
│            never weights)                           │  ✗ NO weight access      │
│                                                     └──────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
pip install torch torchvision numpy matplotlib
cd /path/to/project
python -m unlearning_meta.main
```

### Windows 11 / 8GB RAM Setup

This is a supported, validated target environment. Recommended setup:

```powershell
# CPU-only PyTorch build — much smaller download (~200MB vs ~2.5GB+ for
# the CUDA build) and avoids loading unused CUDA driver bindings into
# memory on a machine without a dedicated GPU.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchvision numpy matplotlib scikit-learn

# Run with the low-memory preset (smaller model/batch, tuned for ~8GB RAM)
python -m unlearning_meta.main --low-memory --cycles 20
```

A few practical notes for this environment:

* **Terminal**: use Windows Terminal or the VS Code integrated terminal
  rather than legacy `cmd.exe` — both default to UTF-8 and full ANSI
  colour support out of the box. The code also defensively reconfigures
  stdout to UTF-8 and nudges legacy consoles into ANSI mode at startup
  (`utils/win_compat.py`), so plain `cmd.exe` works too, just with
  slightly plainer output on older console hosts.
* **Memory budget**: actually profiled (`resource.getrusage`, not
  guessed) at **~789 MB peak RSS** for a full 30-cycle run of the
  `--low-memory` preset on a Linux container — see the docstring on
  `Config.for_low_memory_cpu()` in `config.py` for the full
  measurement breakdown, including confirmation that this project's
  own state (journal, strategy memory, model, optimiser) stays flat
  across cycles once the bounded-deque caps engage; nearly all of the
  789 MB is fixed one-time import overhead from torch/numpy/sklearn/
  matplotlib, not something that grows with `--cycles`. This leaves
  comfortable headroom under 8GB total system RAM even with a
  browser/IDE open alongside. (Profiled on Linux, not Windows itself —
  see the docstring for that caveat.) `--low-memory` adds extra margin
  and faster CPU iteration via a smaller hidden-layer width and batch
  size; reach for it mainly if you plan to scale up the model later
  (a CNN backbone, larger images) and want a known-good starting
  point to scale down from.
* **No GPU required**: `device` auto-detects and falls back to CPU
  automatically (`config.py::TrainingConfig.device`). All DataLoaders
  default to `num_workers=0`, which avoids a Windows-specific
  multiprocessing pitfall (worker processes are spawned, not forked, on
  Windows, which otherwise requires extra `__main__` guarding).
* **Real MNIST download**: `main.py` downloads MNIST via `torchvision`,
  which needs normal internet access. If you're on a restricted network,
  use `python -m unlearning_meta.run_real_experiment` instead — it runs
  the identical pipeline against scikit-learn's bundled `digits` dataset
  with no download required.

---

## Quick Start

```python
from unlearning_meta.config import Config
from unlearning_meta.main import run_experiment

cfg = Config()
cfg.training.num_cycles     = 25
cfg.training.forget_classes = [0]          # forget digit "0"
cfg.training.device         = "cuda"       # or "cpu"

tracker = run_experiment(cfg)
print(tracker.summary())
```

### Real-Data Validation Run (no network required)

`main.py` uses real MNIST and needs network access to download it. For a
network-free end-to-end validation against genuinely learnable data
(scikit-learn's bundled 8×8 digit images — not random tensors), run:

```bash
python -m unlearning_meta.run_real_experiment
```

This exercises the full dual-engine loop, the MIA evaluator, rule
extraction/evolution, and the strategy-memory lifecycle against data with
real class structure, and writes results to `./outputs_real/` and
`./logs_real/`. See `docs/ARCHITECTURE.md` § 10 for the validated results,
including two implementation bugs this run surfaced and fixed (a rule-
extraction off-by-one and a BatchNorm1d crash on size-1 trailing batches)
and an emergent strategy-selection limitation worth knowing about before
relying on this for a real deployment.

### Instance-Level Forgetting (harder benchmark)

Whole-class forgetting (above) is solved almost trivially — forget_acc
reaches 0 within 2–3 cycles regardless of strategy sophistication, which
makes it a poor benchmark for showing one unlearning approach is better
than another. For a harder, more realistic task (forgetting a random 10%
of *every* class, entangled with retain data rather than trivially
separable by class identity):

```bash
python -m unlearning_meta.run_instance_level_experiment
```

**Read this before assuming the system works well in general**: on this
benchmark, the same hyperparameters that succeed cleanly on whole-class
forgetting produce **persistent under-forgetting** — forget_acc stayed
within 2 points of its pre-unlearning value for all 25 cycles despite
`increase_forgetting_strength` being proposed and applied in 18 of them.
See `docs/ARCHITECTURE.md` § 12 for the full results and a mechanistic
hypothesis (retain-protection mechanisms tuned for whole-class forgetting
may be strong enough to fully neutralize gradient ascent when forget and
retain data overlap).

---

## Command-Line Interface

```bash
# Forget class 0, run for 30 cycles
python -m unlearning_meta.main --cycles 30 --forget-classes 0

# Forget classes 0 and 1, custom EWC lambda
python -m unlearning_meta.main --forget-classes 0 1 --ewc-lambda 800

# Stronger gradient ascent
python -m unlearning_meta.main --ga-strength 2.0 --device cuda

# Quiet mode (no stdout, log to file only)
python -m unlearning_meta.main --quiet
```

---

## Module Map

```
unlearning_meta/
├── config.py                   Master configuration
├── schemas.py                   Shared dataclasses
├── main.py                     Entry point + training loop
│
├── models/
│   └── base_model.py           MLP backbone
│
├── data/
│   └── dataset_utils.py        MNIST split utilities
│
├── modules/
│   ├── unlearning/
│   │   ├── memory_trace.py     Neuron selectivity mapping
│   │   ├── fisher_information.py  Diagonal Fisher IM
│   │   ├── gradient_ascent.py  GA forgetting + noise
│   │   ├── dream_replay.py     Pseudo-sample retention
│   │   ├── correction_replay.py   Real-data retention
│   │   ├── ewc_consolidation.py   EWC regularisation
│   │   └── evaluator.py        6-metric evaluation
│   │
│   └── experience/
│       ├── memory_journal.py   Chronological cycle log
│       ├── self_reflection.py  Failure analysis
│       ├── self_explanation.py Natural language logs
│       ├── rule_extraction.py  If-then rule mining
│       ├── strategy_memory.py  4-state lifecycle memory
│       ├── strategy_evolution.py  Rule merging
│       └── meta_knowledge.py   UCB + action statistics
│
├── engines/
│   ├── unlearning_engine.py    Layer 1 orchestrator
│   └── experience_engine.py    Layer 2 orchestrator
│
├── evaluation/
│   ├── mia_evaluator.py        Confidence + Shadow MIA
│   └── metrics.py              Tracker + CSV/JSONL export
│
└── utils/
    ├── logger.py               Colour stdout + file logging
    └── visualization.py        Matplotlib dashboards
```

---

## Outputs

After running, the `./outputs/` directory contains:

| File | Description |
|------|-------------|
| `metrics.csv` | Per-cycle metrics table |
| `metrics.jsonl` | Structured JSON Lines log |
| `metric_trajectories.png` | 7-panel time series |
| `strategy_effectiveness.png` | Per-action score-delta bar chart |
| `memory_lifecycle.png` | Strategy memory pie chart |
| `score_heatmap.png` | Metric × cycle heatmap |
| `final_model.pt` | Final model checkpoint |
| `checkpoint_*.pt` | Mid-run checkpoints |

Logs in `./logs/`:

| File | Description |
|------|-------------|
| `unlearning_exp.log` | Human-readable timestamped log |
| `unlearning_exp.jsonl` | Machine-readable structured log |

---

## Composite Score

```
Ω = F + P + A + R - H + 0.5·S

where:
  F = 1 - forget_acc    (forgetting quality;    ↑ better)
  P = privacy_score     (MIA resistance;        ↑ better)
  A = test_accuracy     (overall accuracy;      ↑ better)
  R = retain_accuracy   (retention quality;     ↑ better)
  H = hallucination     (hallucination rate;    ↓ better, penalised)
  S = stability         (parameter stability;   ↑ better)

Theoretical maximum ≈ 5.5  (all perfect, H=0)
Practical target    ≥ 4.0
```

---

## Critical Design Constraint

> **The Experience Engine NEVER accesses or modifies model weights.**
>
> It receives only `UnlearningMetrics` and emits `StrategyUpdate` proposals.  
> The Unlearning Engine decides whether and how to apply each proposal.  
> All parameter updates happen exclusively inside `UnlearningEngine.run_cycle()`.

---

## Testing

```bash
pip install -r requirements.txt pytest
python -m pytest tests/
```

73 tests, converted from ad-hoc verification performed during this
project's development (ARCHITECTURE.md documents the bugs each one
guards against — the BatchNorm1d crash on size-1 batches, the
exploration-rate independent-resampling bug, the gradient-clip
saturation, the Fisher Forgetting epsilon-calibration bug, the Shadow
MIA evaluation-data leakage, the AUC tie-handling bug, and the project's
core architectural constraint that the Experience Engine never touches
model weights).

## Literature Baseline

`modules/unlearning/fisher_forgetting_baseline.py` implements Fisher
Forgetting (Golatkar et al., CVPR 2020) for comparison against this
project's dual-engine system — see `ARCHITECTURE.md` § 19 for an
exploratory comparison and a genuine robustness difference it surfaced
(the literature baseline's fixed calibration failed catastrophically on
1 of 15 seeds; this project's adaptive system did not fail on any).
§ 22 implements and checks a fix (per-layer epsilon normalization,
opt-in via `per_layer_normalize=True`) against that exact failure case;
whether it holds up as a general average-performance improvement is
still open — see `docs/IDEAS.md` #1.

## Project Documents

- `docs/ARCHITECTURE.md` — design, implementation, and results for
  everything that has actually been built and run. Every claim in it is
  backed by a completed result.
- `docs/IDEAS.md` — proposed mechanisms, open questions, and
  pre-registered-but-not-yet-executed study designs. Kept separate so a
  reader of ARCHITECTURE.md is never left guessing whether something is
  a finding or a plan.
- `docs/external_reviews/` — raw archived text of external AI review
  rounds (ChatGPT, Gemini); IDEAS.md summarizes and assesses each round,
  this directory preserves the original wording.

## Dependencies & License

```bash
pip install -r requirements.txt
```

MIT licensed — see `LICENSE`.

---

## Citation

```bibtex
@misc{unlearning_meta_2025,
  title   = {Dual-Engine Cognitive Architecture for Adaptive Machine Unlearning},
  year    = {2025},
  note    = {Research Prototype: Machine Unlearning + Meta-Learning},
  url     = {https://github.com/your-org/unlearning-meta}
}
```
