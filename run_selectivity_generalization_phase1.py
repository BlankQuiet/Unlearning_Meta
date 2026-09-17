"""
run_selectivity_generalization_phase1.py
=============================================
IDEAS.md Idea 9, Phase 1 (diagnostic only -- no performance claim).

Measures MemoryTraceMapper's raw selectivity distribution (mean/median/
std/min/max/p10/p90) under two task structures, same model architecture,
same seeds, same measurement code -- the only thing that differs is
whether forget/retain are different SAMPLES of the same classes
(instance-level, §24.1's benchmark) or structurally different CLASSES
(class-level, this project's original, easier benchmark, §10 and
earlier, superseded by instance-level from §12 onward specifically
because class-level was found too easy for FORGETTING performance --
whether it is also different for SELECTIVITY specifically is exactly
what this script checks, since that's a different question).

This does not run noise or ascent, does not compute a composite_score,
and makes no performance claim -- purely characterizes the upstream
signal, per Idea 9's two-phase design (round-4 external review).

Usage:
    python -m unlearning_meta.run_selectivity_generalization_phase1
"""

from __future__ import annotations
import statistics

import torch

from unlearning_meta.models.base_model import MLP
from unlearning_meta.data.dataset_utils import (
    load_digits_instance_forgetting,
    load_digits_unlearning_datasets,
)
from unlearning_meta.modules.unlearning.memory_trace import MemoryTraceMapper
from unlearning_meta.main import set_seed, pretrain

SEEDS = [42, 123, 777]  # identical to §24.1's original instance-level calibration


def measure_selectivity(all_sel: torch.Tensor) -> dict:
    n = all_sel.numel()
    return {
        "n": n,
        "mean":   all_sel.mean().item(),
        "median": all_sel.median().item(),
        "std":    all_sel.std().item(),
        "min":    all_sel.min().item(),
        "max":    all_sel.max().item(),
        "p10":    all_sel.kthvalue(max(1, int(0.1 * n))).values.item(),
        "p90":    all_sel.kthvalue(max(1, int(0.9 * n))).values.item(),
    }


def instance_level_selectivity(seed: int) -> dict:
    set_seed(seed)
    fl, rl, tl, full = load_digits_instance_forgetting(
        forget_fraction=0.10, batch_size=32, seed=seed,
    )
    model = MLP(64, [128, 64], 10)
    pretrain(model, full, epochs=15, lr=0.002, device="cpu", verbose=False)

    trace = MemoryTraceMapper(top_k=50, device="cpu")
    trace.compute(model, fl, rl)
    all_sel = torch.cat([s.flatten() for s in trace.selectivity.values()])
    return measure_selectivity(all_sel)


def class_level_selectivity(seed: int, forget_classes=(0,)) -> dict:
    set_seed(seed)
    fl, rl, tl, full = load_digits_unlearning_datasets(
        forget_classes=list(forget_classes), batch_size=32, seed=seed,
    )
    model = MLP(64, [128, 64], 10)
    pretrain(model, full, epochs=15, lr=0.002, device="cpu", verbose=False)

    trace = MemoryTraceMapper(top_k=50, device="cpu")
    trace.compute(model, fl, rl)
    all_sel = torch.cat([s.flatten() for s in trace.selectivity.values()])
    return measure_selectivity(all_sel)


def main() -> None:
    print("=" * 72)
    print("IDEA 9 PHASE 1 -- selectivity distribution, instance- vs class-level")
    print("Diagnostic only. No noise/ascent run, no composite_score, no")
    print("performance claim -- characterizing MemoryTraceMapper's raw")
    print("output under two task structures, same model/seeds/measurement code.")
    print("=" * 72 + "\n")

    instance_results, class_results = [], []

    for seed in SEEDS:
        inst = instance_level_selectivity(seed)
        instance_results.append(inst)
        print(f"[instance-level] seed={seed}: n={inst['n']}  mean={inst['mean']:.4f}  "
              f"median={inst['median']:.4f}  std={inst['std']:.4f}  "
              f"min={inst['min']:.4f}  max={inst['max']:.4f}  "
              f"p10={inst['p10']:.4f}  p90={inst['p90']:.4f}")

    for seed in SEEDS:
        cls = class_level_selectivity(seed, forget_classes=(0,))
        class_results.append(cls)
        print(f"[class-level]    seed={seed}: n={cls['n']}  mean={cls['mean']:.4f}  "
              f"median={cls['median']:.4f}  std={cls['std']:.4f}  "
              f"min={cls['min']:.4f}  max={cls['max']:.4f}  "
              f"p10={cls['p10']:.4f}  p90={cls['p90']:.4f}")

    print("\n" + "=" * 72)
    print("SUMMARY (mean across 3 seeds)")
    print("=" * 72)
    for label, results in (("instance-level", instance_results), ("class-level", class_results)):
        mean_of_means = statistics.mean(r["mean"] for r in results)
        mean_of_stds  = statistics.mean(r["std"]  for r in results)
        mean_of_range = statistics.mean(r["max"] - r["min"] for r in results)
        print(f"{label:>15}: mean(mean)={mean_of_means:.4f}  mean(std)={mean_of_stds:.4f}  "
              f"mean(max-min)={mean_of_range:.4f}")

    inst_std = statistics.mean(r["std"] for r in instance_results)
    cls_std  = statistics.mean(r["std"] for r in class_results)
    ratio = cls_std / inst_std if inst_std > 0 else float("inf")
    print(f"\nclass-level std / instance-level std = {ratio:.2f}x")
    print("(Idea 9's own framing: notably >1 supports the task-structure")
    print("hypothesis and motivates Phase 2; close to 1 points at")
    print("MemoryTraceMapper's own design instead.)")


if __name__ == "__main__":
    main()
