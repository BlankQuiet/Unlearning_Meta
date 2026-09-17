"""
schemas.py
==========
Shared dataclasses used across the entire system.

Critical design rule
--------------------
  StrategyUpdate contains NO reference to model weights or parameters.
  The Experience Engine produces StrategyUpdate; only the Unlearning
  Engine executes weight changes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time


# ─────────────────────────────────────────────────────────────────────────────
# Unlearning metrics  (Layer 1 → Layer 2 interface)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UnlearningMetrics:
    """Output of one Unlearning Engine cycle."""
    forget_acc: float       # accuracy on forget set  (↓ better)
    retain_acc: float       # accuracy on retain set  (↑ better)
    accuracy: float         # overall test accuracy   (↑ better)
    privacy: float          # MIA-based privacy score (↑ better, max 1.0)
    hallucination: float    # wrong-but-confident rate(↓ better)
    stability: float        # parameter stability     (↑ better)
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def composite_score(
        self,
        weights:           Optional[Dict[str, float]] = None,
        target_forget_acc: Optional[float] = None,
        retain_floor:      Optional[float] = None,
    ) -> float:
        """
        Compute F + P + A + R - H (+stability bonus).

        Args:
            weights:           Per-term weights (defaults as below).
            target_forget_acc: If None (default), forget quality targets
                               0 — correct for whole-class forgetting,
                               where "no residual accuracy on the forgotten
                               concept" is the unambiguous goal. If set
                               (e.g. to a retrain-from-scratch gold-standard
                               baseline for instance-level forgetting — see
                               ARCHITECTURE.md §12.1/§16), forget quality
                               instead targets *closeness* to this value,
                               since driving forget_acc all the way to 0 on
                               an instance-level task means suppressing
                               correct predictions the model has every
                               right to make from general class knowledge —
                               over-forgetting, not success. Without this,
                               every score-driven mechanism in the system
                               (bandit momentum, rule outcome tracking, meta
                               UCB updates) silently optimises toward the
                               wrong target on that benchmark; see §16 for
                               the audit that found this.
            retain_floor:      If set and self.retain_acc falls below it,
                               return a heavily penalised score regardless
                               of other terms — a hard guardrail (external
                               review round 2 Q2 / round 4's connection to
                               §12.1) rather than a soft linear trade-off,
                               so catastrophic retain damage can never be
                               "worth it" for gains elsewhere in the score.
        """
        w = weights or {
            "forget": 1.0, "privacy": 1.0, "accuracy": 1.0,
            "retention": 1.0, "hallucination": -1.0, "stability": 0.5,
        }

        if retain_floor is not None and self.retain_acc < retain_floor:
            return -10.0   # guardrail: dominates any other term combination

        if target_forget_acc is None:
            # forget_quality = 1 - forget_acc  (lower forget_acc = better forgetting)
            forget_quality = 1.0 - self.forget_acc
        else:
            forget_quality = 1.0 - abs(self.forget_acc - target_forget_acc)

        return (
            w.get("forget", 1.0)       * forget_quality
            + w.get("privacy", 1.0)    * self.privacy
            + w.get("accuracy", 1.0)   * self.accuracy
            + w.get("retention", 1.0)  * self.retain_acc
            + w.get("hallucination", -1.0) * self.hallucination  # negative
            + w.get("stability", 0.5)  * self.stability
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "forget_acc":    self.forget_acc,
            "retain_acc":    self.retain_acc,
            "accuracy":      self.accuracy,
            "privacy":       self.privacy,
            "hallucination": self.hallucination,
            "stability":     self.stability,
        }

    def __repr__(self) -> str:
        s = self.composite_score()
        return (
            f"[Cycle {self.cycle}] "
            f"forget={self.forget_acc:.3f} retain={self.retain_acc:.3f} "
            f"acc={self.accuracy:.3f} priv={self.privacy:.3f} "
            f"hall={self.hallucination:.3f} stab={self.stability:.3f} "
            f"→ score={s:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy proposals  (Layer 2 → Layer 1 interface — proposals only)
# ─────────────────────────────────────────────────────────────────────────────

class StrategyAction(str, Enum):
    """
    All strategy actions the Experience Engine may propose.

    The first 7 (through NOOP) are the actions specified in the original
    project brief. INCREASE_GRAD_CLIP / DECREASE_GRAD_CLIP were added
    afterward, motivated by a precisely-isolated empirical finding
    (ARCHITECTURE.md §12.3-12.4): on a harder instance-level forgetting
    benchmark, increase_forgetting_strength was correctly and repeatedly
    selected (18/25 cycles in one run) but had almost no effect, because
    gradient-norm clipping (max_grad_norm, hardcoded at 1.0, untouched by
    any of the original 7 actions) silently capped the resulting gradient
    step regardless of how high gradient_ascent_strength climbed. This is
    a genuine capability gap in the original action set, not a renaming or
    reframing of anything the brief specified — flagged explicitly here
    since it extends beyond what was originally asked for.
    """
    INCREASE_FORGETTING_STRENGTH  = "increase_forgetting_strength"
    DECREASE_FORGETTING_STRENGTH  = "decrease_forgetting_strength"
    INCREASE_NOISE                = "increase_noise"
    DECREASE_NOISE                = "decrease_noise"
    INCREASE_REPLAY               = "increase_replay"
    INCREASE_CORRECTION_REPLAY    = "increase_correction_replay"
    INCREASE_EWC                  = "increase_ewc"
    DECREASE_EWC                  = "decrease_ewc"
    INCREASE_GRAD_CLIP            = "increase_grad_clip"   # relax the clip ceiling
    DECREASE_GRAD_CLIP            = "decrease_grad_clip"   # tighten it back
    NOOP                          = "noop"

    # Mapping to config attribute & delta sign for easy application
    @property
    def config_attr(self) -> Optional[str]:
        return {
            "increase_forgetting_strength":  "gradient_ascent_strength",
            "decrease_forgetting_strength":  "gradient_ascent_strength",
            "increase_noise":               "noise_scale",
            "decrease_noise":               "noise_scale",
            "increase_replay":              "dream_replay_epochs",
            "increase_correction_replay":   "correction_epochs",
            "increase_grad_clip":           "max_grad_norm",
            "decrease_grad_clip":           "max_grad_norm",
            "increase_ewc":                 "ewc_lambda",
            "decrease_ewc":                 "ewc_lambda",
            "noop":                         None,
        }.get(self.value)

    @property
    def sign(self) -> int:
        return -1 if self.value.startswith("decrease") else +1

    @property
    def opposite(self) -> Optional["StrategyAction"]:
        """
        The 'inverse direction' action on the same underlying parameter, if
        one exists (ARCHITECTURE.md §15 — the non-stationary/paired-arm
        bandit fix).

        Deliberately a static, explicit mapping rather than inferring pairs
        by string-prefix matching ("increase_" / "decrease_"): prefix
        inference is implicit and fragile — it would silently misbehave for
        NOOP (no prefix at all) and for increase_replay /
        increase_correction_replay, which have NO decrease_ counterpart in
        the current action space at all (see schemas.py's config_attr map)
        and would need special-casing either way. An explicit mapping is
        self-documenting and fails loudly (KeyError-free None, not a wrong
        guess) if the action space changes in the future.
        """
        return {
            "increase_forgetting_strength": StrategyAction.DECREASE_FORGETTING_STRENGTH,
            "decrease_forgetting_strength": StrategyAction.INCREASE_FORGETTING_STRENGTH,
            "increase_noise":               StrategyAction.DECREASE_NOISE,
            "decrease_noise":               StrategyAction.INCREASE_NOISE,
            "increase_ewc":                 StrategyAction.DECREASE_EWC,
            "decrease_ewc":                 StrategyAction.INCREASE_EWC,
            "increase_grad_clip":           StrategyAction.DECREASE_GRAD_CLIP,
            "decrease_grad_clip":           StrategyAction.INCREASE_GRAD_CLIP,
            # increase_replay, increase_correction_replay, noop: no opposite
        }.get(self.value)


@dataclass
class StrategyUpdate:
    """
    Proposal emitted by the Experience Engine.

    IMPORTANT: This object contains NO model weights or gradients.
    It only encodes *what hyperparameter to change* and *by how much*.
    The Unlearning Engine decides whether to apply it and does the actual work.
    """
    action: StrategyAction
    delta: float                    # magnitude of change (always positive; sign from action)
    rationale: str = ""             # natural language explanation
    confidence: float = 0.5         # 0–1 confidence in this proposal
    source_rule: Optional[str] = None  # rule ID that generated this
    cycle: int = 0

    def is_noop(self) -> bool:
        return self.action == StrategyAction.NOOP

    def __repr__(self) -> str:
        return (
            f"StrategyUpdate({self.action.value}, δ={self.delta:.4f}, "
            f"conf={self.confidence:.2f}) — {self.rationale[:60]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Journal Entry  (internal to Experience Engine)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JournalEntry:
    cycle: int
    metrics: UnlearningMetrics
    strategy_applied: Optional[StrategyUpdate]
    composite_score: float
    failure_types: List[str] = field(default_factory=list)
    explanation: str = ""
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Record  (Strategy Memory)
# ─────────────────────────────────────────────────────────────────────────────

class StrategyState(str, Enum):
    ACTIVE   = "active"
    DORMANT  = "dormant"
    ARCHIVE  = "archive"
    DELETED  = "deleted"


@dataclass
class StrategyRecord:
    """
    One entry in the Long-Term Strategy Memory.

    Fields per spec:
        rule, success_rate, generalization_score,
        usage_frequency, last_used, confidence
    """
    id: str
    rule: str                       # natural language rule description
    action: StrategyAction
    condition_key: str              # metric name  (e.g. "forget_acc")
    condition_op: str               # ">" or "<"
    condition_thresh: float         # threshold value
    success_rate: float = 0.0
    generalization_score: float = 0.5
    usage_frequency: int = 0
    last_used: int = 0              # cycle number
    confidence: float = 0.5
    state: StrategyState = StrategyState.ACTIVE
    created_at: int = 0
    update_history: List[Dict] = field(default_factory=list)

    def importance(
        self,
        current_cycle: int,
        w_success: float = 0.35,
        w_gen: float = 0.25,
        w_recency: float = 0.20,
        w_usage: float = 0.20,
        max_freq: int = 100,
    ) -> float:
        """
        importance = 0.35·success + 0.25·generalization
                   + 0.20·recency  + 0.20·usage_frequency
        """
        recency = max(0.0, 1.0 - (current_cycle - self.last_used) / max(current_cycle, 1))
        usage   = min(1.0, self.usage_frequency / max(max_freq, 1))
        return (
            w_success  * self.success_rate
            + w_gen    * self.generalization_score
            + w_recency * recency
            + w_usage  * usage
        )

    def matches(self, metrics: UnlearningMetrics) -> bool:
        """Check if this strategy's condition is satisfied by given metrics."""
        value = getattr(metrics, self.condition_key, None)
        if value is None:
            return False
        if self.condition_op == ">":
            return value > self.condition_thresh
        elif self.condition_op == "<":
            return value < self.condition_thresh
        return False
