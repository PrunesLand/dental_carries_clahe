# CLAHE Ablation Study — U-Net across 5 Encoders

Does CLAHE preprocessing improve caries segmentation, and for which encoder
architectures?

**Experiment design**

| Parameter | Value |
|-----------|-------|
| Architecture | U-Net (`segmentation_models_pytorch`) |
| Encoders | `resnet50`, `efficientnet-b4`, `vgg16_bn`, `densenet121`, `mobilenet_v2` |
| Variable | CLAHE on vs. off |
| CV scheme | 5-fold × 5-repeat = **25 paired runs** per (encoder, condition) |
| Statistics | Nadeau-Bengio corrected paired t-test + Holm correction across encoders |

---

## Project structure

```
clahe_ablation/
├── config.py          # All configuration variables — edit before running
├── main.py            # Experiment driver (entry point)
├── download_data.py   # Kaggle dataset downloader
├── requirements.txt
├── src/
│   ├── data.py        # ImageCache, CariesDataset, list_files
│   ├── metrics.py     # confusion_counts, metrics_from_counts, DiceLoss
│   ├── stats.py       # corrected_paired_ttest, corrected_ci
│   ├── model.py       # build_model, make_criterion, auto batch-size
│   ├── train.py       # run_lr_probe, run_kfold_cv
│   ├── report.py      # results table, forest plot, validation curves
│   └── visualize.py   # qualitative samples, IEEE-ready panel export
├── dataset_root/      # dataset downloaded here (git-ignored)
├── outputs/           # all generated artifacts (git-ignored)
└── README.md
```

Everything the project needs — the dataset, checkpoints, and figures — lives
**inside this folder**, so it can be dropped into a repository and run as-is.
`dataset_root/` and `outputs/` are listed in `.gitignore` and are not committed.

All outputs are written to `outputs/` inside the project folder:

```
clahe_ablation/
└── outputs/
    ├── encoder_batch.json                  # cached auto batch-size results
    ├── encoder_lr.json                     # cached LR probe results
    ├── clahe_unet_5encoder_results.json    # all CV scores (written per encoder)
    ├── checkpoints/
    │   ├── {encoder}_{tag}_progress.json  # per-run resume state (deleted on completion)
    │   └── {encoder}_{tag}_run0.pth       # saved model for qualitative figures
    └── figures/
        ├── forest_plot.png
        ├── validation_curves.png
        └── sample*_{encoder}_{tag}_{original|gt|pred}.png
```

---

## Setup

### 1. Install dependencies

```bash
# PyTorch — pick the wheel that matches your CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Everything else:
pip install -r requirements.txt
```

### 2. Get a Kaggle API key

1. Log in at <https://www.kaggle.com/settings>
2. Scroll to **API** → **Create New Token** — this downloads `kaggle.json`
3. Export the credentials in your shell:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

> **Security note:** never commit these values to git.  Use environment
> variables or a secrets manager.

### 3. Download the dataset

```bash
cd clahe_ablation/
python download_data.py
```

The dataset is extracted to `dataset_root/` inside this folder by default.
To use an existing copy elsewhere, set `DATASET_ROOT` before running:

```bash
export DATASET_ROOT=/path/to/your/data
python download_data.py
```

---

## Configuration

Open [config.py](config.py) to adjust any parameter before running:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENCODERS` | 5 encoders | List of SMP encoder names to test |
| `K_FOLDS` | 5 | Number of CV folds |
| `N_REPEATS` | 5 | Number of CV repeats (total runs = K×N) |
| `EPOCHS` | 30 | Training epochs per run |
| `SEED` | 42 | Global random seed |
| `IMAGE_SIZE` | `(512, 1024)` | Resize target (H, W) |
| `CLIP_LIMIT` | 2.0 | CLAHE clip limit |
| `GRID_SIZE` | `(16, 16)` | CLAHE tile grid size |
| `LR_CANDIDATES` | `[1e-3, 5e-4, 1e-4]` | LRs probed per encoder |
| `LR_PROBE_EPOCHS` | 12 | Short training epochs for LR probe |
| `BATCH_CAP` | 64 | Upper limit for auto batch-size search |
| `DATASET_ROOT` | `dataset_root/` (inside folder) | Path to dataset (overridable via env var) |

---

## Running

All commands must be run from **inside** the `clahe_ablation/` directory:

```bash
cd clahe_ablation/
```

### Full experiment (training + reporting + visualisation)

```bash
python main.py
```

The experiment is **fully resumable**.  If interrupted, re-running the same
command picks up from the last completed fold — no training is repeated.

### Reporting only (use saved results, skip training)

```bash
python main.py --report-only
```

Requires `clahe_unet_5encoder_results.json` to exist.

### Qualitative visualisation only

```bash
python main.py --visualize-only
```

Requires `{encoder}_clahe_run0.pth` checkpoints (produced during training).

### Download data only

```bash
python download_data.py
```

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/clahe_unet_5encoder_results.json` | All per-run DSC, IoU, recall, precision, F1, accuracy scores |
| `outputs/figures/forest_plot.png` | Delta DSC ± 95% CI per encoder |
| `outputs/figures/validation_curves.png` | Val loss + IoU curves (baseline vs CLAHE) |
| `outputs/figures/sample*_original.png` | Raw X-ray panel |
| `outputs/figures/sample*_gt.png` | Ground-truth overlay (red) |
| `outputs/figures/sample*_pred.png` | Prediction overlay (green) |

The panel images are caption-free and sized for direct use as IEEE subfigures.
See the LaTeX snippet in the original notebook for a ready-to-use `figure*`
environment.

---

## Statistical notes

- **Nadeau-Bengio corrected t-test** — adjusts the variance estimate for the
  non-independence of CV folds.  A standard paired t-test would be
  anti-conservative here.
- **Holm correction** — controls the family-wise error rate across the 5
  encoder comparisons.
- Results are framed as a *relative* CLAHE effect (no separate held-out test
  set given the ~100-image dataset).  The best-epoch selection bias is
  symmetric across conditions and cancels in the paired difference.
