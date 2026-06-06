"""
Reporting: results table, forest plot, and validation curves.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

import config
from src.stats import corrected_paired_ttest, corrected_ci


def build_results_table(all_results: dict) -> pd.DataFrame:
    """
    Build a per-encoder summary table with Nadeau-Bengio corrected t-tests
    and Holm correction across encoders.
    """
    rows, pvals = [], []
    for enc in config.ENCODERS:
        b = all_results[enc]["base"]["dsc"]
        c = all_results[enc]["clahe"]["dsc"]
        t, p, mdiff = corrected_paired_ttest(b, c)
        lo, hi = corrected_ci(b, c)
        rows.append({
            "Encoder":      enc,
            "Baseline DSC": np.mean(b),
            "CLAHE DSC":    np.mean(c),
            "Delta":        mdiff,
            "95% CI":       f"[{lo:+.4f}, {hi:+.4f}]",
            "t":            t,
            "p (raw)":      p,
        })
        pvals.append(p)

    reject, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    df = pd.DataFrame(rows)
    df["p (Holm)"] = p_holm
    df["Significant"] = ["yes" if r else "no" for r in reject]
    return df


def print_results_table(df: pd.DataFrame) -> None:
    n_runs = config.K_FOLDS * config.N_REPEATS
    n_enc = len(config.ENCODERS)
    print("=" * 78)
    print(f"CLAHE vs BASELINE on DSC  |  U-Net  |  "
          f"{config.K_FOLDS}x{config.N_REPEATS} repeated CV (n={n_runs})")
    print(f"Corrected paired t-test (Nadeau-Bengio) + Holm across {n_enc} encoders")
    print("=" * 78)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}",
                           "display.max_columns", None,
                           "display.width", 120):
        print(df.to_string(index=False))
    print()


def plot_forest(all_results: dict, save_path: str | None = None) -> None:
    """Forest plot of the CLAHE effect size (Delta DSC ± 95% CI) per encoder."""
    fig, ax = plt.subplots(figsize=(8, 0.8 * len(config.ENCODERS) + 2))
    ys = np.arange(len(config.ENCODERS))[::-1]

    for y, enc in zip(ys, config.ENCODERS):
        b = all_results[enc]["base"]["dsc"]
        c = all_results[enc]["clahe"]["dsc"]
        _, p, mdiff = corrected_paired_ttest(b, c)
        lo, hi = corrected_ci(b, c)
        # Green = clearly positive, Red = clearly negative, Blue = straddles 0
        color = "C2" if lo > 0 else ("C3" if hi < 0 else "C0")
        ax.errorbar(mdiff, y, xerr=[[mdiff - lo], [hi - mdiff]],
                    fmt="o", capsize=4, color=color)
        ax.text(hi, y, f"  p={p:.3f}", va="center", fontsize=9)

    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(config.ENCODERS)
    ax.set_xlabel("Delta DSC  (CLAHE − Baseline)  with 95% CI")
    ax.set_ylabel("Encoder")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved forest plot -> {save_path}")
    plt.show()


def plot_validation_curves(all_results: dict, save_dir: str | None = None) -> None:
    """Per-encoder validation loss and IoU curves (baseline vs CLAHE).

    Saves one file per encoder per metric:
        {save_dir}/{encoder}_val_loss.png
        {save_dir}/{encoder}_val_iou.png
    """
    metrics = [
        ("loss", "val Tversky loss"),
        ("iou",  "val IoU"),
    ]

    for enc in config.ENCODERS:
        d = all_results[enc]
        for key, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(d["hist_base"][key], "--", label="baseline")
            ax.plot(d["hist_clahe"][key], label="clahe")
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.legend()
            plt.tight_layout()

            if save_dir:
                path = os.path.join(save_dir, f"{enc}_val_{key}.png")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                print(f"Saved -> {path}")
            plt.show()
            plt.close(fig)
