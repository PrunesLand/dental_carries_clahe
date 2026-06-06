"""
Training routines: LR probe and the main repeated k-fold CV loop.

Both functions are fully resumable: partial results are checkpointed
after every run to a JSON file in CKPT_DIR, and skipped on re-entry.
"""

import json
import os

import numpy as np
import torch
from sklearn.model_selection import RepeatedKFold
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
from src.data import CariesDataset, ImageCache
from src.metrics import confusion_counts, metrics_from_counts
from src.model import build_model, make_criterion, free_cuda, device, unwrap


def _progress_path(encoder: str, tag: str) -> str:
    return os.path.join(config.CKPT_DIR, f"{encoder}_{tag}_progress.json")


# ---------------------------------------------------------------------------
# LR probe
# ---------------------------------------------------------------------------

def quick_train_iou(
    encoder: str,
    lr: float,
    batch_size: int,
    cache: ImageCache,
    file_names: np.ndarray,
    epochs: int = config.LR_PROBE_EPOCHS,
) -> float:
    """One-fold short baseline run; returns best validation IoU at this LR."""
    kf = RepeatedKFold(n_splits=config.K_FOLDS, n_repeats=1, random_state=config.SEED)
    train_idx, val_idx = next(iter(kf.split(file_names)))

    tr = CariesDataset(cache, file_names[train_idx], augmentation=True, use_clahe=False)
    va = CariesDataset(cache, file_names[val_idx], augmentation=False, use_clahe=False)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True,
                    num_workers=config.NUM_WORKERS, pin_memory=True)
    vl = DataLoader(va, batch_size=batch_size * config.EVAL_BATCH_MULT, shuffle=False,
                    num_workers=config.NUM_WORKERS, pin_memory=True)

    model = build_model(encoder)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = make_criterion()
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    best = 0.0

    epoch_bar = tqdm(range(epochs), desc=f"probe {encoder} lr={lr:.0e}",
                     leave=False, unit="ep")
    for _ in epoch_bar:
        model.train()
        for x, y in tl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = crit(model(x), y)
                loss.backward()
                opt.step()

        model.eval()
        TP = FP = FN = TN = 0
        with torch.no_grad():
            for x, y in vl:
                x = x.to(device, non_blocking=True)
                pred = torch.sigmoid(model(x)).cpu().numpy()
                tp, fp, fn, tn = confusion_counts(pred, y.numpy())
                TP += tp; FP += fp; FN += fn; TN += tn
        best = max(best, metrics_from_counts(TP, FP, FN, TN)["iou"])
        epoch_bar.set_postfix(best_iou=f"{best:.4f}")

    del model, opt
    if scaler:
        del scaler
    free_cuda()
    return best


def run_lr_probe(
    cache: ImageCache,
    file_names: np.ndarray,
    encoder_batch: dict[str, int],
) -> dict[str, float]:
    """
    For each encoder, pick the LR from LR_CANDIDATES that gives the best
    one-fold validation IoU.  Results are cached in LR_FILE.
    """
    if os.path.exists(config.LR_FILE):
        with open(config.LR_FILE) as fh:
            lr_map = json.load(fh)
        print("Loaded cached LRs:", lr_map)
        return lr_map

    lr_map: dict[str, float] = {}
    for enc in config.ENCODERS:
        bs = encoder_batch[enc]
        best_lr, best_iou = config.LR_CANDIDATES[0], -1.0
        for lr in config.LR_CANDIDATES:
            iou = quick_train_iou(enc, lr, bs, cache, file_names)
            print(f"  [{enc}] lr={lr:.0e} -> val IoU {iou:.4f}")
            if iou > best_iou:
                best_iou, best_lr = iou, lr
        lr_map[enc] = best_lr
        print(f"  => [{enc}] chosen LR = {best_lr:.0e}\n")

    with open(config.LR_FILE, "w") as fh:
        json.dump(lr_map, fh)
    return lr_map


# ---------------------------------------------------------------------------
# Main repeated k-fold CV
# ---------------------------------------------------------------------------

def run_kfold_cv(
    encoder: str,
    use_clahe: bool,
    lr: float,
    batch_size: int,
    tag: str,
    cache: ImageCache,
    file_names: np.ndarray,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Run K_FOLDS x N_REPEATS cross-validation for one (encoder, condition) pair.

    Args:
        encoder:    SMP encoder name.
        use_clahe:  Whether to use CLAHE-preprocessed images.
        lr:         Learning rate chosen by the LR probe.
        batch_size: Training batch size chosen by the batch-size finder.
        tag:        "baseline" or "clahe" — used for checkpoint naming.
        cache:      Pre-loaded ImageCache.
        file_names: Full array of image filenames.

    Returns:
        per_run:  Dict of metric lists, one value per CV run (length = K*N).
        history:  Dict with "loss" and "iou" epoch curves (averaged over runs).
    """
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    kf = RepeatedKFold(n_splits=config.K_FOLDS, n_repeats=config.N_REPEATS,
                       random_state=config.SEED)
    n_runs = config.K_FOLDS * config.N_REPEATS
    crit = make_criterion()
    pp = _progress_path(encoder, tag)

    # Resume from partial run if available
    if os.path.exists(pp):
        with open(pp) as fh:
            state = json.load(fh)
        per_run   = state["per_run"]
        hist_loss = np.array(state["hist_loss"])
        hist_iou  = np.array(state["hist_iou"])
        done      = state["completed_runs"]
        print(f"  [{encoder}|{tag}] resuming from run {done}/{n_runs}")
    else:
        per_run   = {m: [] for m in ["dsc", "iou", "recall", "precision", "f1", "accuracy"]}
        hist_loss = np.zeros((n_runs, config.EPOCHS))
        hist_iou  = np.zeros((n_runs, config.EPOCHS))
        done      = 0

    for run, (tr_idx, va_idx) in enumerate(kf.split(file_names)):
        if run < done:
            continue

        tr = CariesDataset(cache, file_names[tr_idx], augmentation=True, use_clahe=use_clahe)
        va = CariesDataset(cache, file_names[va_idx], augmentation=False, use_clahe=use_clahe)
        tl = DataLoader(tr, batch_size=batch_size, shuffle=True,
                        num_workers=config.NUM_WORKERS, pin_memory=True,
                        persistent_workers=config.NUM_WORKERS > 0)
        vl = DataLoader(va, batch_size=batch_size * config.EVAL_BATCH_MULT, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True,
                        persistent_workers=config.NUM_WORKERS > 0)

        model = build_model(encoder)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5
        )
        scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
        best_iou, best_metrics = -1.0, None

        epoch_bar = tqdm(range(config.EPOCHS),
                         desc=f"{encoder}|{tag} run {run + 1}/{n_runs}",
                         leave=False, unit="ep")
        for ep in epoch_bar:
            # --- train ---
            model.train()
            for x, y in tl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad()
                if scaler:
                    with torch.amp.autocast("cuda"):
                        loss = crit(model(x), y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss = crit(model(x), y)
                    loss.backward()
                    opt.step()

            # --- validate ---
            model.eval()
            TP = FP = FN = TN = 0
            vloss = 0.0
            nb = 0
            with torch.no_grad():
                for x, y in vl:
                    x = x.to(device, non_blocking=True)
                    logits = model(x)
                    vloss += crit(logits, y.to(device)).item()
                    nb += 1
                    pred = torch.sigmoid(logits).cpu().numpy()
                    tp, fp, fn, tn = confusion_counts(pred, y.numpy())
                    TP += tp; FP += fp; FN += fn; TN += tn

            m = metrics_from_counts(TP, FP, FN, TN)
            sched.step(m["iou"])
            hist_loss[run, ep] = vloss / max(nb, 1)
            hist_iou[run, ep] = m["iou"]
            if m["iou"] > best_iou:
                best_iou, best_metrics = m["iou"], m
            epoch_bar.set_postfix(val_loss=f"{hist_loss[run, ep]:.4f}",
                                  iou=f"{m['iou']:.4f}",
                                  best=f"{best_iou:.4f}")

        for k in per_run:
            per_run[k].append(best_metrics[k])

        # Save run-0 checkpoint for qualitative visualisation
        if run == 0:
            ckpt_path = os.path.join(config.CKPT_DIR, f"{encoder}_{tag}_run0.pth")
            torch.save(unwrap(model).state_dict(), ckpt_path)

        del tl, vl  # shut down persistent workers immediately before next run
        del model, opt
        if scaler:
            del scaler
        free_cuda()

        # Persist progress so the run is resumable after interruption
        with open(pp, "w") as fh:
            json.dump({
                "completed_runs": run + 1,
                "per_run": per_run,
                "hist_loss": hist_loss.tolist(),
                "hist_iou": hist_iou.tolist(),
            }, fh)
        print(f"  [{encoder}|{tag}] run {run + 1}/{n_runs}  best IoU {best_iou:.4f}")

    history = {
        "loss": hist_loss.mean(0).tolist(),
        "iou":  hist_iou.mean(0).tolist(),
    }
    return per_run, history
