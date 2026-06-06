"""
Qualitative visualisation of predictions on unseen validation images.

visualize_random_samples  — inline side-by-side figure (original / GT / pred).
export_validation_panels  — saves caption-free individual files for IEEE figures.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
import torch

import config
from src.data import CariesDataset, ImageCache
from src.metrics import confusion_counts, metrics_from_counts
from src.model import build_model, device, unwrap


def _load_val_dataset(
    encoder: str, tag: str, cache: ImageCache, file_names: np.ndarray
) -> tuple[CariesDataset, np.ndarray]:
    """Recover the same run-0 validation fold used during training."""
    kf = RepeatedKFold(n_splits=config.K_FOLDS, n_repeats=config.N_REPEATS,
                       random_state=config.SEED)
    _, va_idx = next(iter(kf.split(file_names)))
    val_files = file_names[va_idx]
    ds = CariesDataset(cache, val_files, augmentation=False, use_clahe=(tag == "clahe"))
    return ds, val_files


def visualize_random_samples(
    encoder: str,
    cache: ImageCache,
    file_names: np.ndarray,
    tag: str = "clahe",
    n: int = 2,
    seed: int = 0,
) -> None:
    """Display n random unseen validation samples with GT and prediction overlays."""
    ckpt = os.path.join(config.CKPT_DIR, f"{encoder}_{tag}_run0.pth")
    if not os.path.exists(ckpt):
        print(f"Checkpoint {ckpt} not found — run training first.")
        return

    ds, val_files = _load_val_dataset(encoder, tag, cache, file_names)
    model = build_model(encoder, weights=None)
    unwrap(model).load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    print(f"{encoder} [{tag}] showing unseen val files: {[val_files[i] for i in picks]}")

    fig, axes = plt.subplots(len(picks), 3, figsize=(18, 5 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for r, i in enumerate(picks):
            img, gt = ds[i]
            pred = torch.sigmoid(model(img[None].to(device))).cpu().numpy()[0, 0]
            dsc = metrics_from_counts(*confusion_counts(pred, gt.numpy()))["dsc"]
            img_np, gt_np = img[0].numpy(), gt[0].numpy()

            axes[r, 0].imshow(img_np, cmap="gray")
            axes[r, 0].set_title("Original X-ray")
            axes[r, 1].imshow(img_np, cmap="gray")
            axes[r, 1].imshow(gt_np, cmap="Reds", alpha=0.5)
            axes[r, 1].set_title("Ground Truth")
            axes[r, 2].imshow(img_np, cmap="gray")
            axes[r, 2].imshow((pred > 0.5), cmap="Greens", alpha=0.5)
            axes[r, 2].set_title(f"Prediction  DSC={dsc:.3f}")
            for col in range(3):
                axes[r, col].axis("off")

    plt.tight_layout()
    plt.show()


def _overlay(
    gray: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.5
) -> np.ndarray:
    rgb = np.repeat(np.clip(gray, 0, 1)[..., None], 3, axis=2).astype(float)
    m = mask.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(color, float)
    return np.clip(rgb, 0, 1)


def export_validation_panels(
    encoder: str,
    cache: ImageCache,
    file_names: np.ndarray,
    tag: str = "clahe",
    n: int = 2,
    seed: int = 0,
    dpi: int = 300,
    fmt: str = "png",
) -> list[dict]:
    """
    Save each validation sample as three separate image files (no titles/axes)
    suitable for IEEE subfigures:
        <FIG_DIR>/sample{k}_{encoder}_{tag}_original.{fmt}
        <FIG_DIR>/sample{k}_{encoder}_{tag}_gt.{fmt}
        <FIG_DIR>/sample{k}_{encoder}_{tag}_pred.{fmt}

    Returns a list of dicts with keys: file, dsc, stem.
    """
    os.makedirs(config.FIG_DIR, exist_ok=True)
    ckpt = os.path.join(config.CKPT_DIR, f"{encoder}_{tag}_run0.pth")
    if not os.path.exists(ckpt):
        print(f"Checkpoint {ckpt} not found — run training first.")
        return []

    ds, val_files = _load_val_dataset(encoder, tag, cache, file_names)
    model = build_model(encoder, weights=None)
    unwrap(model).load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    saved = []

    with torch.no_grad():
        for k, i in enumerate(picks, 1):
            img, gt = ds[i]
            pred = torch.sigmoid(model(img[None].to(device))).cpu().numpy()[0, 0]
            dsc = metrics_from_counts(*confusion_counts(pred, gt.numpy()))["dsc"]
            gray, gt_np, pred_np = img[0].numpy(), gt[0].numpy(), (pred > 0.5)
            stem = os.path.join(config.FIG_DIR, f"sample{k}_{encoder}_{tag}")

            plt.imsave(f"{stem}_original.{fmt}", gray, cmap="gray", vmin=0, vmax=1, dpi=dpi)
            plt.imsave(f"{stem}_gt.{fmt}",   _overlay(gray, gt_np,   (1, 0, 0)), dpi=dpi)
            plt.imsave(f"{stem}_pred.{fmt}", _overlay(gray, pred_np, (0, 1, 0)), dpi=dpi)

            saved.append({"file": str(val_files[i]), "dsc": float(dsc), "stem": stem})
            print(f"  sample{k}: {val_files[i]}  DSC={dsc:.3f}  ->  {stem}_*.{fmt}")

    print(f"\nSaved {len(saved) * 3} panel files to {config.FIG_DIR}/")
    return saved
