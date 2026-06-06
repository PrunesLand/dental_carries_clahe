"""
Model construction, criterion, and GPU batch-size auto-tuning.
"""

import gc
import json
import os

import torch
import segmentation_models_pytorch as smp

import config

device = "cuda" if torch.cuda.is_available() else "cpu"


def _build_unet(encoder: str, weights: str | None) -> torch.nn.Module:
    """Raw, uncompiled U-Net on the target device."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=1,
        classes=1,
    ).to(device)


def build_model(encoder: str, weights: str | None = "imagenet") -> torch.nn.Module:
    model = _build_unet(encoder, weights)
    if config.USE_COMPILE and device == "cuda":
        try:
            model = torch.compile(model)
        except Exception as e:  # never let a compile issue break the run
            print(f"  [warn] torch.compile unavailable ({e}); running eager.")
    return model


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, peeling off a torch.compile() wrapper.

    Checkpoints are saved/loaded through this so they stay compatible whether or
    not USE_COMPILE is on (a compiled model's state_dict has '_orig_mod.' keys).
    """
    return getattr(model, "_orig_mod", model)


def make_criterion() -> torch.nn.Module:
    """Tversky loss with alpha=0.3, beta=0.7 (penalises false negatives more)."""
    return smp.losses.TverskyLoss(mode="binary", from_logits=True, alpha=0.3, beta=0.7)


def free_cuda() -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def find_max_batch(
    encoder: str,
    cap: int = config.BATCH_CAP,
    safety: float = config.BATCH_SAFETY,
) -> int:
    """
    Binary-search for the largest batch size that fits in GPU memory by
    running a single forward + backward pass at increasing sizes.
    Falls back to 8 on CPU.
    """
    if device != "cuda":
        return 8

    H, W = config.IMAGE_SIZE
    crit = make_criterion()
    bs, last_ok, model = 2, 2, None

    while bs <= cap:
        try:
            model = _build_unet(encoder, "imagenet")  # raw model: skip compile while probing
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda")
            x = torch.randn(bs, 1, H, W, device=device)
            y = torch.randint(0, 2, (bs, 1, H, W), device=device).float()
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            last_ok = bs
            del model, opt, scaler, x, y, loss
            model = None
            free_cuda()
            bs *= 2
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(
                e, torch.cuda.OutOfMemoryError
            ):
                raise
            del model
            model = None
            free_cuda()
            break

    chosen = max(2, int(last_ok * safety))
    print(f"  [{encoder}] fits batch ~{last_ok} -> using {chosen}")
    return chosen


def get_batch_sizes() -> dict[str, int]:
    """Load from cache file if available, otherwise probe and save."""
    if os.path.exists(config.BATCH_FILE):
        with open(config.BATCH_FILE) as fh:
            sizes = json.load(fh)
        print("Loaded cached batch sizes:", sizes)
        return sizes

    sizes = {enc: find_max_batch(enc) for enc in config.ENCODERS}
    with open(config.BATCH_FILE, "w") as fh:
        json.dump(sizes, fh)
    return sizes
