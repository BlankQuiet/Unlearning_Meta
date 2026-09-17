"""
data/dataset_utils.py
=====================
Dataset helpers for machine unlearning experiments.

Splits MNIST into:
  • forget_loader  – data the model must forget
  • retain_loader  – data the model must preserve
  • test_loader    – evaluation set (never used for training)

Note on num_workers: all DataLoaders below default to num_workers=0.
This is a deliberate choice, not an oversight — on Windows, DataLoader
worker processes are created via spawn (no fork()), which requires every
script using num_workers > 0 to be wrapped in an
`if __name__ == "__main__":` guard or it will recursively re-import and
re-execute the whole script in each worker. num_workers=0 runs the
loader in the main process and sidesteps this Windows-specific pitfall
entirely; for the model/dataset sizes used in this project (small MLPs,
MNIST/digits-scale data) the throughput cost of single-process loading is
negligible relative to the forward/backward passes themselves.
"""

from __future__ import annotations
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as T
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

def get_transform(augment: bool = False) -> T.Compose:
    transforms = [T.ToTensor(), T.Normalize(MNIST_MEAN, MNIST_STD)]
    if augment:
        transforms = [T.RandomRotation(10), T.RandomHorizontalFlip()] + transforms
    return T.Compose(transforms)


# ─────────────────────────────────────────────────────────────────────────────
# Core split function (real MNIST — requires network access)
# ─────────────────────────────────────────────────────────────────────────────

def load_unlearning_datasets(
    data_dir: str = "./data",
    forget_classes: Optional[List[int]] = None,
    batch_size: int = 128,
    retain_fraction: float = 1.0,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST split into four DataLoaders.

    Args:
        data_dir:        Root directory for dataset download.
        forget_classes:  Class labels to be forgotten (default: [0]).
        batch_size:      Mini-batch size for all loaders.
        retain_fraction: Fraction of retain samples to use (for speed).
        seed:            Random seed for reproducibility.
        num_workers:     DataLoader worker processes. Defaults to 0 — see
                         module docstring for why this matters on Windows.

    Returns:
        (forget_loader, retain_loader, test_loader, full_train_loader)
    """
    if forget_classes is None:
        forget_classes = [0]

    rng = np.random.default_rng(seed)

    train_ds = torchvision.datasets.MNIST(
        data_dir, train=True,  download=True, transform=get_transform()
    )
    test_ds  = torchvision.datasets.MNIST(
        data_dir, train=False, download=True, transform=get_transform()
    )

    targets = np.array(train_ds.targets)

    forget_mask = np.isin(targets, forget_classes)
    retain_mask = ~forget_mask

    forget_idx = np.where(forget_mask)[0].tolist()
    retain_idx = np.where(retain_mask)[0].tolist()

    # Sub-sample retain set for speed
    if retain_fraction < 1.0:
        n_retain = int(len(retain_idx) * retain_fraction)
        retain_idx = rng.choice(retain_idx, size=n_retain, replace=False).tolist()

    forget_ds = Subset(train_ds, forget_idx)
    retain_ds = Subset(train_ds, retain_idx)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)

    forget_loader = DataLoader(forget_ds, shuffle=True,  **kw)
    retain_loader = DataLoader(retain_ds, shuffle=True,  **kw)
    test_loader   = DataLoader(test_ds,   shuffle=False, **kw)
    full_loader   = DataLoader(train_ds,  shuffle=True,  **kw)

    print(
        f"[Dataset] forget={len(forget_ds):,}  retain={len(retain_ds):,}  "
        f"test={len(test_ds):,}  forget_classes={forget_classes}"
    )
    return forget_loader, retain_loader, test_loader, full_loader


# ─────────────────────────────────────────────────────────────────────────────
# Network-free alternative (sklearn digits — no download required)
# ─────────────────────────────────────────────────────────────────────────────

def load_digits_unlearning_datasets(
    forget_classes: Optional[List[int]] = None,
    batch_size: int = 32,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Network-free alternative to ``load_unlearning_datasets`` using
    scikit-learn's bundled ``digits`` dataset (1,797 real 8×8 handwritten
    digit images, 10 classes). No external download required — useful for
    offline environments, CI, or quick validation runs.

    Interface mirrors ``load_unlearning_datasets`` exactly so it can be
    swapped in without touching any downstream code (model input_dim
    must be set to 64 instead of 784).

    Args:
        forget_classes: Class labels to be forgotten (default: [0]).
        batch_size:     Mini-batch size for all loaders.
        test_fraction:  Fraction of data held out as the test set.
        seed:           Random seed for reproducibility.

    Returns:
        (forget_loader, retain_loader, test_loader, full_train_loader)
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    if forget_classes is None:
        forget_classes = [0]

    digits = load_digits()
    X = digits.images.astype("float32") / 16.0   # pixel range [0,16] → [0,1]
    y = digits.target.astype("int64")

    # Normalise like MNIST (mean/std) for consistent training dynamics
    mean, std = X.mean(), X.std()
    X = (X - mean) / (std + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )

    X_train_t = torch.from_numpy(X_train).unsqueeze(1)   # [N,1,8,8]
    y_train_t = torch.from_numpy(y_train)
    X_test_t  = torch.from_numpy(X_test).unsqueeze(1)
    y_test_t  = torch.from_numpy(y_test)

    forget_mask = np.isin(y_train, forget_classes)
    retain_mask = ~forget_mask

    forget_ds = TensorDataset(X_train_t[forget_mask], y_train_t[forget_mask])
    retain_ds = TensorDataset(X_train_t[retain_mask], y_train_t[retain_mask])
    test_ds   = TensorDataset(X_test_t, y_test_t)
    full_ds   = TensorDataset(X_train_t, y_train_t)

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    forget_loader = DataLoader(forget_ds, shuffle=True,  **kw)
    retain_loader = DataLoader(retain_ds, shuffle=True,  **kw)
    test_loader   = DataLoader(test_ds,   shuffle=False, **kw)
    full_loader   = DataLoader(full_ds,   shuffle=True,  **kw)

    print(
        f"[Dataset:digits] forget={len(forget_ds):,}  retain={len(retain_ds):,}  "
        f"test={len(test_ds):,}  forget_classes={forget_classes}  "
        f"(network-free sklearn substitute, input_dim=64)"
    )
    return forget_loader, retain_loader, test_loader, full_loader


# ─────────────────────────────────────────────────────────────────────────────
# Instance-level forgetting (harder benchmark — Q5 from external review round 2)
# ─────────────────────────────────────────────────────────────────────────────

def load_digits_instance_forgetting(
    forget_fraction: float = 0.10,
    batch_size:      int   = 32,
    test_fraction:   float = 0.2,
    seed:            int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Harder unlearning benchmark: forget a random FRACTION of *every* class
    instead of one whole class.

    Why this is harder (and why it matters)
    -----------------------------------------
    Whole-class forgetting (``load_digits_unlearning_datasets``) lets the
    network solve "forgetting" by refusing to recognise an entire concept
    it never needs again — the forget and retain sets are trivially
    separable by class identity alone, and gradient ascent on the forget
    set doesn't have to disturb the decision boundary for any digit the
    model still needs to recognise. Empirically (see ARCHITECTURE.md §10)
    this converges to forget_acc=0.000 within 2–3 cycles regardless of
    unlearning strategy sophistication, which makes it a poor benchmark
    for showing that a *smarter* strategy layer outperforms a naive one:
    everything succeeds equally fast at the headline metric.

    Instance-level forgetting draws the forget set from the SAME class
    distributions as the retain set (e.g. forgetting these 14 particular
    "3"s while retaining the other 164 "3"s). The network cannot solve
    this by refusing a whole concept — the general decision boundary for
    "3" is still needed for the 164 retained examples, so forgetting must
    target something closer to memorisation of the *specific* instances,
    which is a much closer analogue of the actual GDPR-style
    "right to be forgotten" (one user's specific records, not a whole
    category) that motivates machine unlearning research in the first
    place.

    IMPORTANT — forget_acc is not the right target here
    -------------------------------------------------------
    For whole-class forgetting, forget_acc→0 is the unambiguous goal. For
    instance-level forgetting it is NOT: a model retrained from scratch on
    retain-only data would still often classify a "forgotten" instance
    correctly just because it resembles other same-class examples that
    were never removed. Driving forget_acc all the way to 0 here would
    mean actively suppressing correct predictions the model has every
    right to make from general class knowledge alone — that's over-forgetting,
    not success. See ``compute_retrain_baseline`` below: the correct
    per-sample target is "behave like a model that never saw this sample,"
    not "get it wrong." This is also why MIA-based privacy evaluation
    (already implemented in evaluation/mia_evaluator.py) becomes the
    primary success signal for this benchmark rather than a secondary
    check — raw forget_acc alone isn't a well-defined optimisation target.

    Args:
        forget_fraction: Fraction of EACH class's samples to forget
                         (e.g. 0.10 = forget 10% of every digit, not one
                         whole digit).
        batch_size:      Mini-batch size for all loaders.
        test_fraction:   Fraction of data held out as the test set.
        seed:            Random seed for reproducibility.

    Returns:
        (forget_loader, retain_loader, test_loader, full_train_loader)
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.images.astype("float32") / 16.0
    y = digits.target.astype("int64")
    mean, std = X.mean(), X.std()
    X = (X - mean) / (std + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )

    rng = np.random.default_rng(seed)
    forget_mask = np.zeros(len(y_train), dtype=bool)

    # Stratified per-class sampling: forget_fraction of EACH class,
    # not forget_fraction of the dataset as a whole (which could by chance
    # concentrate in a few classes and silently degrade back into
    # near-whole-class forgetting for the unlucky classes).
    for c in np.unique(y_train):
        class_idx = np.where(y_train == c)[0]
        n_forget = max(1, int(len(class_idx) * forget_fraction))
        forget_idx_c = rng.choice(class_idx, size=n_forget, replace=False)
        forget_mask[forget_idx_c] = True

    retain_mask = ~forget_mask

    X_train_t = torch.from_numpy(X_train).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train)
    X_test_t  = torch.from_numpy(X_test).unsqueeze(1)
    y_test_t  = torch.from_numpy(y_test)

    forget_ds = TensorDataset(X_train_t[forget_mask], y_train_t[forget_mask])
    retain_ds = TensorDataset(X_train_t[retain_mask], y_train_t[retain_mask])
    test_ds   = TensorDataset(X_test_t, y_test_t)
    full_ds   = TensorDataset(X_train_t, y_train_t)

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    forget_loader = DataLoader(forget_ds, shuffle=True,  **kw)
    retain_loader = DataLoader(retain_ds, shuffle=True,  **kw)
    test_loader   = DataLoader(test_ds,   shuffle=False, **kw)
    full_loader   = DataLoader(full_ds,   shuffle=True,  **kw)

    counts = np.bincount(y_train[forget_mask])
    print(
        f"[Dataset:digits-instance] forget={forget_mask.sum():,} "
        f"({forget_fraction:.0%} of every class, per-class counts={list(counts)})  "
        f"retain={retain_mask.sum():,}  test={len(test_ds):,}"
    )
    return forget_loader, retain_loader, test_loader, full_loader


def compute_retrain_baseline_forget_acc(
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    model_factory,
    epochs: int = 15,
    lr:     float = 0.003,
    device: str = "cpu",
    seed:   int = 42,
) -> float:
    """
    Gold-standard reference for instance-level forgetting: train a FRESH
    model on retain-only data (never sees the forget set at all) and
    measure ITS accuracy on the forget set.

    This is the "exact unlearning" baseline (full retrain, excluding the
    deleted data) that the literature treats as ground truth for what
    "successfully forgotten" should look like. For instance-level
    forgetting this number is typically well above 0 — the model still
    gets many forgotten instances right purely from general class
    knowledge — which is exactly why raw forget_acc=0 is the wrong target
    for this benchmark (see module docstring above). A good unlearning
    method should drive forget_acc *towards this baseline*, not towards 0.

    Args:
        forget_loader: DataLoader for the forget set (used only for eval).
        retain_loader: DataLoader for the retain set (used for training).
        model_factory: Zero-arg callable returning a fresh, untrained model
                       (e.g. ``lambda: MLP(64, [64,32], 10)``).
        epochs, lr:    Training hyperparameters for the fresh model.
        device:        torch device.
        seed:          Random seed.

    Returns:
        forget_acc achieved by a model that never saw the forget set.
    """
    import torch.nn as nn
    import torch.optim as optim
    torch.manual_seed(seed)

    model = model_factory().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in retain_loader:
            if x.size(0) < 2:   # BatchNorm1d guard, same rationale as elsewhere
                continue
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in forget_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: sample a small batch from a DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def sample_batch(
    loader: DataLoader,
    n: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the first n samples from loader as a single tensor."""
    xs, ys = [], []
    collected = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        collected += x.size(0)
        if collected >= n:
            break
    x_all = torch.cat(xs, dim=0)[:n].to(device)
    y_all = torch.cat(ys, dim=0)[:n].to(device)
    return x_all, y_all


def load_wine_instance_forgetting(
    forget_fraction: float = 0.10,
    batch_size:      int   = 8,
    test_fraction:   float = 0.2,
    seed:            int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Instance-level forgetting on a genuinely different dataset (IDEAS.md
    Idea 11's first exploratory check, mirroring §18's own low-key,
    non-pre-registered style for a first generalization look) -- sklearn's
    bundled wine dataset (13 tabular features, 3 classes, 178 samples
    total), chosen specifically because it differs from the digits
    benchmark on every axis this project has never varied before: tabular
    not image data, 3 classes not 10, ~178 samples not ~1,800 (roughly
    10x smaller), and a completely different feature scale/semantics
    (chemical assay measurements, not pixel intensities). Same
    instance-level design as load_digits_instance_forgetting (stratified
    per-class forgetting, not whole-class) for a like-for-like comparison
    of the same benchmark *type* on different data — see that function's
    docstring for why instance-level (not whole-class) is used throughout
    this project.

    default batch_size=8, not 32: with ~142 training samples (178 * 0.8)
    and forget_fraction=0.10, the forget set is only ~14 samples --
    batch_size=32 would put the entire forget set in a single batch,
    unlike every other loader in this project. 8 keeps multiple batches
    per epoch even for the smallest split.

    Returns:
        (forget_loader, retain_loader, test_loader, full_train_loader)
    """
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split

    wine = load_wine()
    X = wine.data.astype("float32")
    y = wine.target.astype("int64")
    mean, std = X.mean(axis=0), X.std(axis=0)
    X = (X - mean) / (std + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )

    rng = np.random.default_rng(seed)
    forget_mask = np.zeros(len(y_train), dtype=bool)

    for c in np.unique(y_train):
        class_idx = np.where(y_train == c)[0]
        n_forget = max(1, int(len(class_idx) * forget_fraction))
        forget_idx_c = rng.choice(class_idx, size=n_forget, replace=False)
        forget_mask[forget_idx_c] = True

    retain_mask = ~forget_mask

    X_train_t = torch.from_numpy(X_train)  # [N, 13] -- flat, no image dims to preserve
    y_train_t = torch.from_numpy(y_train)
    X_test_t  = torch.from_numpy(X_test)
    y_test_t  = torch.from_numpy(y_test)

    forget_ds = TensorDataset(X_train_t[forget_mask], y_train_t[forget_mask])
    retain_ds = TensorDataset(X_train_t[retain_mask], y_train_t[retain_mask])
    test_ds   = TensorDataset(X_test_t, y_test_t)
    full_ds   = TensorDataset(X_train_t, y_train_t)

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    forget_loader = DataLoader(forget_ds, shuffle=True,  **kw)
    retain_loader = DataLoader(retain_ds, shuffle=True,  **kw)
    test_loader   = DataLoader(test_ds,   shuffle=False, **kw)
    full_loader   = DataLoader(full_ds,   shuffle=True,  **kw)

    counts = np.bincount(y_train[forget_mask])
    print(
        f"[Dataset:wine-instance] forget={forget_mask.sum():,} "
        f"({forget_fraction:.0%} of every class, per-class counts={list(counts)})  "
        f"retain={retain_mask.sum():,}  test={len(test_ds):,}"
    )
    return forget_loader, retain_loader, test_loader, full_loader
