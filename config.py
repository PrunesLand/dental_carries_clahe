"""
All configuration variables for the CLAHE ablation study.
Edit this file to change the experiment setup before running main.py.
"""

import os

# Absolute path to this folder. Every path below is anchored to it so the
# project is fully self-contained: data, checkpoints and figures all stay
# inside clahe_ablation/ no matter what the current working directory is.
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
ARCHITECTURE = "Unet"
ENCODERS = ["resnet50", "efficientnet-b4", "vgg16_bn", "densenet121", "mobilenet_v2"]

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
K_FOLDS = 5
N_REPEATS = 3        # total runs per (encoder, condition) = K_FOLDS * N_REPEATS
EPOCHS = 30
SEED = 42

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
IMAGE_SIZE = (512, 1024)   # (H, W)
CLIP_LIMIT = 2.0           # CLAHE clip limit
GRID_SIZE = (16, 16)       # CLAHE tile grid size

# ---------------------------------------------------------------------------
# Dataset paths
# Resolution order: $DATASET_ROOT, then dataset_root/ inside this folder, then
# ../dataset_root/ (one level up). The first location that actually contains an
# images_cut/ folder wins, so the project works whether the data lives beside
# the code or in the parent directory.
# ---------------------------------------------------------------------------
def _resolve_dataset_root() -> str:
    env = os.environ.get("DATASET_ROOT")
    if env:
        return env
    candidates = [
        os.path.join(HERE, "dataset_root"),
        os.path.join(HERE, "..", "dataset_root"),
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "images_cut")):
            return cand
    return candidates[0]  # default; check_dataset() will print guidance if missing


DATASET_ROOT = _resolve_dataset_root()
IMAGE_PATH = os.path.join(DATASET_ROOT, "images_cut") + os.sep
LABEL_PATH = os.path.join(DATASET_ROOT, "labels_cut") + os.sep

# ---------------------------------------------------------------------------
# Learning-rate search (run once per encoder, results cached)
# ---------------------------------------------------------------------------
LR_CANDIDATES = [1e-3, 5e-4, 1e-4]
LR_PROBE_EPOCHS = 12

# ---------------------------------------------------------------------------
# Auto batch-size finder
# ---------------------------------------------------------------------------
BATCH_CAP = 64      # upper bound to probe
BATCH_SAFETY = 0.85  # fraction of max-fitting batch to actually use
EVAL_BATCH_MULT = 2  # validation loader uses a larger (no-grad) batch

# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
NUM_WORKERS = min(4, os.cpu_count() or 2)

# ---------------------------------------------------------------------------
# Throughput optimizations
# Pure speed; these do NOT change the experiment design or the statistics.
#   USE_TF32     - allow TF32 matmul/conv on Ampere+ (negligible precision cost,
#                  already moot under AMP/fp16). Safe, keep on.
#   USE_COMPILE  - wrap the model in torch.compile(). Biggest single-GPU win
#                  (~1.3-1.7x). Falls back to eager automatically if it errors,
#                  so it can never break a run. Set False if your PyTorch/GPU
#                  build misbehaves.
# ---------------------------------------------------------------------------
USE_TF32 = True
USE_COMPILE = True

# ---------------------------------------------------------------------------
# Multi-GPU
# When main.py is launched with >1 visible CUDA device, it splits the encoders
# across the GPUs (one independent worker process per GPU) and merges results.
# This is identical to running each encoder on its own card — no effect on the
# numbers. ENCODER_COST holds *relative* per-encoder training cost (measured),
# used only to balance the split so the GPUs finish at roughly the same time.
# Encoders not listed default to a cost of 1.0.
# ---------------------------------------------------------------------------
ENCODER_COST = {
    "vgg16_bn":        2.19,
    "efficientnet-b4": 1.55,
    "resnet50":        1.40,
    "densenet121":     1.34,
    "mobilenet_v2":    0.64,
}

# ---------------------------------------------------------------------------
# Output paths — anchored to this folder so all outputs land inside
# clahe_ablation/outputs/ regardless of the current working directory.
# ---------------------------------------------------------------------------
OUTPUT_DIR   = os.path.join(HERE, "outputs")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "clahe_unet_5encoder_results.json")
LR_FILE      = os.path.join(OUTPUT_DIR, "encoder_lr.json")
BATCH_FILE   = os.path.join(OUTPUT_DIR, "encoder_batch.json")
CKPT_DIR     = os.path.join(OUTPUT_DIR, "checkpoints")
FIG_DIR      = os.path.join(OUTPUT_DIR, "figures")
