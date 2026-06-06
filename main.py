"""
CLAHE Ablation Study — U-Net across 5 Encoders
===============================================

Experiment driver.  Run from inside the clahe_ablation/ directory:

    python main.py                    # full experiment
    python main.py --report-only      # load saved results and generate plots
    python main.py --visualize-only   # qualitative panels (needs *.pth checkpoints)

The experiment is fully resumable: if interrupted, re-running picks up
from the last completed fold via per-encoder progress files in checkpoints/.
"""

import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch

import config
from src.data import list_files, ImageCache
from src.model import get_batch_sizes, free_cuda
from src.train import run_lr_probe, run_kfold_cv
from src.report import build_results_table, print_results_table, plot_forest, plot_validation_curves
from src.visualize import visualize_random_samples, export_validation_panels


def seed_everything() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def check_dataset() -> None:
    if not os.path.isdir(config.IMAGE_PATH) or not os.path.isdir(config.LABEL_PATH):
        sys.exit(
            f"\nDataset not found at:\n"
            f"  images: {config.IMAGE_PATH}\n"
            f"  labels: {config.LABEL_PATH}\n\n"
            "Download it first:\n"
            "  python download_data.py\n"
            "Or set the DATASET_ROOT environment variable to point at an existing download.\n"
        )


def run_experiment(
    cache: ImageCache,
    file_names: np.ndarray,
    encoders: list[str] | None = None,
    results_file: str | None = None,
) -> dict:
    """Run the ablation (or resume partial results) and return all_results.

    encoders:     subset of config.ENCODERS to run (defaults to all). Used by the
                  per-GPU workers in multi-GPU mode.
    results_file: where to read/write results (defaults to config.RESULTS_FILE).
                  Each GPU worker writes its own shard to avoid concurrent writes.
    """
    encoders = encoders or config.ENCODERS
    results_file = results_file or config.RESULTS_FILE
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CKPT_DIR, exist_ok=True)

    encoder_batch = get_batch_sizes()
    encoder_lr = run_lr_probe(cache, file_names, encoder_batch)

    all_results: dict = {}
    if os.path.exists(results_file):
        with open(results_file) as fh:
            all_results = json.load(fh)
        print("Loaded cached results for:", list(all_results.keys()))

    for enc in encoders:
        if enc in all_results:
            print(f"Skipping {enc} (already complete).")
            continue

        bs, lr = encoder_batch[enc], encoder_lr[enc]
        print(f"\n=== {enc}  (lr={lr:.0e}, batch={bs}) ===")

        base_runs, base_hist = run_kfold_cv(enc, False, lr, bs, "baseline", cache, file_names)
        clahe_runs, clahe_hist = run_kfold_cv(enc, True,  lr, bs, "clahe",    cache, file_names)

        all_results[enc] = {
            "base":       base_runs,
            "clahe":      clahe_runs,
            "hist_base":  base_hist,
            "hist_clahe": clahe_hist,
            "lr":         lr,
            "batch":      bs,
        }

        with open(results_file, "w") as fh:
            json.dump(all_results, fh)

        # Clean up per-run progress files now that this encoder is done
        for tag in ("baseline", "clahe"):
            pp = os.path.join(config.CKPT_DIR, f"{enc}_{tag}_progress.json")
            if os.path.exists(pp):
                os.remove(pp)

        print(f"Saved {enc} -> {results_file}")

    print("\nAll encoders complete.")
    return all_results


def run_reporting(all_results: dict) -> None:
    os.makedirs(config.FIG_DIR, exist_ok=True)
    df = build_results_table(all_results)
    print_results_table(df)
    plot_forest(all_results, save_path=os.path.join(config.FIG_DIR, "forest_plot.png"))
    plot_validation_curves(all_results, save_path=os.path.join(config.FIG_DIR, "validation_curves.png"))


def run_visualization(cache: ImageCache, file_names: np.ndarray) -> None:
    enc = config.ENCODERS[0]
    visualize_random_samples(enc, cache, file_names, tag="clahe", n=2, seed=0)
    export_validation_panels(enc, cache, file_names, tag="clahe", n=2, seed=0)


# ---------------------------------------------------------------------------
# Multi-GPU orchestration
# ---------------------------------------------------------------------------

def partition_encoders(encoders: list[str], n_groups: int) -> list[list[str]]:
    """Balance encoders across n_groups by ENCODER_COST (longest-processing-time
    greedy). Each encoder runs whole on one GPU; this just evens out the makespan."""
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    loads = [0.0] * n_groups
    for enc in sorted(encoders, key=lambda e: config.ENCODER_COST.get(e, 1.0), reverse=True):
        i = loads.index(min(loads))
        groups[i].append(enc)
        loads[i] += config.ENCODER_COST.get(enc, 1.0)
    return [g for g in groups if g]


def run_multi_gpu(cache: ImageCache, file_names: np.ndarray, n_gpus: int) -> dict:
    """Split the encoders across n_gpus, one worker process per GPU, then merge.

    Identical numerically to running each encoder on its own card — the split is
    only about throughput. Shared LR/batch caches are populated once up front so
    workers never race to write them; each worker writes its own results shard.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CKPT_DIR, exist_ok=True)

    # Populate encoder_batch.json / encoder_lr.json once (skipped if cached),
    # so the parallel workers only ever read them.
    encoder_batch = get_batch_sizes()
    run_lr_probe(cache, file_names, encoder_batch)
    free_cuda()

    groups = partition_encoders(config.ENCODERS, n_gpus)
    print(f"\nSplitting {len(config.ENCODERS)} encoders across {len(groups)} GPU(s):")
    for gpu, encs in enumerate(groups):
        cost = sum(config.ENCODER_COST.get(e, 1.0) for e in encs)
        print(f"  GPU {gpu}: {encs}  (relative load {cost:.2f})")

    here = os.path.dirname(os.path.abspath(__file__))
    procs, shards, logs = [], [], []
    for gpu, encs in enumerate(groups):
        shard = os.path.join(config.OUTPUT_DIR, f"results_gpu{gpu}.json")
        logpath = os.path.join(config.OUTPUT_DIR, f"worker_gpu{gpu}.log")
        shards.append(shard)
        log = open(logpath, "w")
        logs.append(log)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        cmd = [sys.executable, os.path.abspath(__file__),
               "--encoders", ",".join(encs), "--results-file", shard]
        procs.append(subprocess.Popen(cmd, cwd=here, env=env, stdout=log,
                                      stderr=subprocess.STDOUT))
        print(f"  -> launched worker on GPU {gpu} (log: {logpath})")

    print("\nWorkers running. Follow progress with e.g.:  "
          f"tail -f {os.path.join(config.OUTPUT_DIR, 'worker_gpu0.log')}\n")
    rcs = [p.wait() for p in procs]
    for log in logs:
        log.close()
    if any(rc != 0 for rc in rcs):
        sys.exit(f"GPU worker(s) failed (return codes {rcs}). "
                 "Check outputs/worker_gpu*.log; re-run to resume.")

    # Merge worker shards into the canonical results file
    merged: dict = {}
    if os.path.exists(config.RESULTS_FILE):
        with open(config.RESULTS_FILE) as fh:
            merged = json.load(fh)
    for shard in shards:
        if os.path.exists(shard):
            with open(shard) as fh:
                merged.update(json.load(fh))
    with open(config.RESULTS_FILE, "w") as fh:
        json.dump(merged, fh)
    print(f"Merged {len(shards)} worker shards -> {config.RESULTS_FILE}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="CLAHE ablation study — U-Net 5 encoders")
    parser.add_argument("--report-only", action="store_true",
                        help="Load saved results and generate plots; skip training.")
    parser.add_argument("--visualize-only", action="store_true",
                        help="Generate qualitative panels only; skip training and stats.")
    parser.add_argument("--encoders", default=None,
                        help="Comma-separated encoder subset to run (used internally "
                             "by per-GPU workers; you normally don't set this).")
    parser.add_argument("--results-file", default=None,
                        help="Override the results file path (used internally by "
                             "per-GPU workers for their shard).")
    parser.add_argument("--single-gpu", action="store_true",
                        help="Force single-GPU even when several GPUs are visible.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    print(f"Device : {device}  (visible CUDA GPUs: {n_gpus})")
    print(f"Encoders: {config.ENCODERS}")

    if config.USE_TF32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed_everything()
    check_dataset()

    file_names = list_files()
    print(f"Found {len(file_names)} images.")

    if args.visualize_only:
        cache = ImageCache(file_names)
        run_visualization(cache, file_names)
        return

    if args.report_only:
        if not os.path.exists(config.RESULTS_FILE):
            sys.exit(f"No results file found at {config.RESULTS_FILE}. Run training first.")
        with open(config.RESULTS_FILE) as fh:
            all_results = json.load(fh)
        run_reporting(all_results)
        return

    # Worker mode: an explicit encoder subset was handed to us (one GPU each).
    if args.encoders:
        encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
        print(f"[worker] training {encoders} -> {args.results_file}")
        cache = ImageCache(file_names)
        run_experiment(cache, file_names, encoders=encoders, results_file=args.results_file)
        return

    cache = ImageCache(file_names)

    # Orchestrator mode: 2+ GPUs visible -> split encoders across them.
    if n_gpus >= 2 and not args.single_gpu:
        all_results = run_multi_gpu(cache, file_names, n_gpus)
    else:
        all_results = run_experiment(cache, file_names)

    run_reporting(all_results)
    run_visualization(cache, file_names)


if __name__ == "__main__":
    main()
