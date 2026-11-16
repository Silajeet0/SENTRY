#!/usr/bin/env python3
"""Plot Laya's binary affiliation confidence scores and threshold trade-offs.

Input predictions.csv columns:
    row,state,actual,p_indian
Input sweep.csv columns:
    threshold,tp,fp,fn,tn,precision,recall,fpr,f1,accuracy

Examples:
    python plot_laya_thresholds.py --predictions predictions.csv --sweep sweep.csv
    python plot_laya_thresholds.py --predictions predictions.csv --sweep sweep.csv --min-precision 0.99
    python plot_laya_thresholds.py --predictions predictions.csv --threshold 0.9781

Uses only saved scores. Does NOT load Laya or retrain it.
Do threshold tuning on a development set, then freeze it before reporting final test metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Works on remote/headless machines.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRED_COLS = {"state", "actual", "p_indian"}
SWEEP_COLS = {"threshold", "tp", "fp", "fn", "tn", "precision", "recall", "fpr", "f1", "accuracy"}


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = PRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = df.copy()
    df["actual"] = df["actual"].astype(str).str.strip().str.upper()
    unexpected = set(df["actual"]) - {"INDIAN", "NOT_INDIAN"}
    if unexpected:
        raise ValueError(f"Unexpected ground-truth labels: {sorted(unexpected)}")
    df["p_indian"] = pd.to_numeric(df["p_indian"], errors="raise")
    if df.empty or not np.isfinite(df["p_indian"]).all():
        raise ValueError("Predictions are empty or contain non-finite probabilities")
    if not df["p_indian"].between(0, 1).all():
        raise ValueError("p_indian values must be in [0, 1]")
    if df["actual"].nunique() != 2:
        raise ValueError("Both INDIAN and NOT_INDIAN cases are needed for threshold analysis")
    return df


def metrics_at(df: pd.DataFrame, threshold: float) -> dict:
    actual = df["actual"].eq("INDIAN").to_numpy()
    pred = df["p_indian"].to_numpy() >= threshold
    tp = int(np.sum(actual & pred))
    fp = int(np.sum(~actual & pred))
    fn = int(np.sum(actual & ~pred))
    tn = int(np.sum(~actual & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(threshold=float(threshold), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, fpr=fpr, f1=f1,
                accuracy=(tp + tn) / len(df))


def load_sweep_or_rebuild(path: Path | None, df: pd.DataFrame) -> pd.DataFrame:
    if path is not None and path.is_file():
        sweep = pd.read_csv(path)
        missing = SWEEP_COLS - set(sweep.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        sweep = sweep.sort_values("threshold").reset_index(drop=True)
        # Make sure sweep.csv is actually from the supplied predictions.csv.
        for _, r in sweep.iterrows():
            observed = metrics_at(df, float(r["threshold"]))
            for col in ("tp", "fp", "fn", "tn"):
                if int(r[col]) != observed[col]:
                    raise ValueError(f"Sweep and predictions disagree at threshold={r['threshold']} ({col}). "
                                     "Regenerate sweep from the same predictions or omit --sweep.")
        return sweep
    thresholds = sorted(set([0.0, 0.5, 1.0] + df["p_indian"].tolist()))
    return pd.DataFrame([metrics_at(df, t) for t in thresholds])


def pick_threshold(sweep: pd.DataFrame, supplied: float | None,
                   min_precision: float | None, max_fpr: float | None) -> tuple[float, str]:
    if supplied is not None:
        return supplied, "user-specified"
    eligible = sweep[sweep["tp"] > 0].copy()
    if min_precision is not None:
        eligible = eligible[eligible["precision"] >= min_precision]
    if max_fpr is not None:
        eligible = eligible[eligible["fpr"] <= max_fpr]
    if eligible.empty:
        raise ValueError("No threshold meets your precision/FPR requirements while detecting any Indian affiliations")
    if min_precision is not None or max_fpr is not None:
        # Preserve as many actual Indian records as possible, then reduce FP.
        eligible = eligible.sort_values(["recall", "fp", "threshold"], ascending=[False, True, True])
        return float(eligible.iloc[0]["threshold"]), "max recall subject to constraints"
    eligible = eligible.sort_values(["f1", "fp", "threshold"], ascending=[False, True, True])
    return float(eligible.iloc[0]["threshold"]), "best observed F1 (development set only)"


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  {path}")


def plot_scatter(df: pd.DataFrame, threshold: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.4))
    rng = np.random.default_rng(42)  # stable jitter, not model stochasticity
    actual = df["actual"].eq("INDIAN").to_numpy()
    probabilities = df["p_indian"].to_numpy()
    y = actual.astype(int) + rng.uniform(-0.12, 0.12, len(df))
    for is_indian, label in [(False, "Actual NOT_INDIAN (0)"), (True, "Actual INDIAN (1)")]:
        group = actual == is_indian
        ax.scatter(probabilities[group], y[group], alpha=0.70, s=28, label=label)
    wrong = (probabilities >= threshold) != actual
    if wrong.any():
        ax.scatter(probabilities[wrong], y[wrong], s=120, marker="o",
                   facecolors="none", edgecolors="black", linewidths=1.3,
                   label=f"Misclassified at t={threshold:.4f}")
    ax.axvline(0.5, linestyle=":", linewidth=1.2, label="argmax cutoff = 0.50")
    if abs(threshold - 0.5) > 1e-9:
        ax.axvline(threshold, linestyle="--", linewidth=1.5,
                   label=f"selected cutoff = {threshold:.4f}")
    ax.set(xlim=(-0.015, 1.015), ylim=(-0.3, 1.3),
           xlabel="Model score P(INDIAN)", ylabel="Actual class (0 = not Indian, 1 = Indian)",
           title="Laya confidence vs actual affiliation label")
    ax.set_yticks([0, 1])
    ax.grid(alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    save_plot(out)


def plot_distribution(df: pd.DataFrame, threshold: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bins = np.linspace(0, 1, 41)
    for label in ("NOT_INDIAN", "INDIAN"):
        scores = df.loc[df["actual"] == label, "p_indian"]
        ax.hist(scores, bins=bins, alpha=0.56, label=f"Actual {label} (n={len(scores)})")
    ax.axvline(threshold, linestyle="--", linewidth=1.5, label=f"t={threshold:.4f}")
    ax.set(xlabel="Model score P(INDIAN)", ylabel="Number of affiliations",
           title="Confidence score distribution by ground-truth class")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    save_plot(out)


def plot_precision_recall(sweep: pd.DataFrame, threshold: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for metric, label in [("precision", "Indian precision"), ("recall", "Indian recall"), ("f1", "Indian F1")]:
        ax.step(sweep["threshold"], sweep[metric], where="post", linewidth=1.8, label=label)
    ax.axvline(threshold, linestyle="--", label=f"t={threshold:.4f}")
    ax.set(xlim=(0, 1), ylim=(-0.02, 1.03), xlabel="P(INDIAN) threshold",
           ylabel="Score (0–1)", title="Indian precision / recall / F1 vs threshold")
    ax.grid(alpha=0.2)
    ax.legend()
    save_plot(out)


def plot_error_counts(sweep: pd.DataFrame, threshold: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.step(sweep["threshold"], sweep["fp"], where="post", linewidth=1.9, label="False positives")
    ax.step(sweep["threshold"], sweep["fn"], where="post", linewidth=1.9, label="False negatives")
    ax.axvline(threshold, linestyle="--", label=f"t={threshold:.4f}")
    ax.set(xlim=(0, 1), xlabel="P(INDIAN) threshold", ylabel="Count",
           title="False positives and false negatives vs threshold")
    ax.grid(alpha=0.2)
    ax.legend()
    save_plot(out)


def plot_zoom(sweep: pd.DataFrame, df: pd.DataFrame, threshold: float,
              low: float, high: float, out: Path) -> None:
    # Recompute at a regular dense grid for accurate step transition visualization.
    grid = np.unique(np.concatenate((np.linspace(low, high, 1200),
                                     df.loc[df["p_indian"].between(low, high), "p_indian"].to_numpy(),
                                     [threshold] if low <= threshold <= high else [])))
    fine = pd.DataFrame([metrics_at(df, float(t)) for t in grid])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.step(fine["threshold"], fine["precision"], where="post", label="Indian precision", linewidth=1.9)
    ax.step(fine["threshold"], fine["recall"], where="post", label="Indian recall", linewidth=1.9)
    ax.step(fine["threshold"], fine["f1"], where="post", label="Indian F1", linewidth=1.5)
    if low <= threshold <= high:
        ax.axvline(threshold, linestyle="--", label=f"t={threshold:.4f}")
    ax.set(xlim=(low, high), ylim=(0, 1.03),
           xlabel="P(INDIAN) threshold (zoomed)", ylabel="Score (0–1)",
           title="Zoomed threshold trade-off near high-confidence errors")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left")
    save_plot(out)


def save_error_cases(df: pd.DataFrame, threshold: float, out: Path) -> None:
    errors = df.copy()
    errors["predicted"] = np.where(errors["p_indian"] >= threshold, "INDIAN", "NOT_INDIAN")
    errors = errors[errors["predicted"] != errors["actual"]]
    errors.to_csv(out, index=False)
    print(f"  {out} ({len(errors)} errors)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--sweep", type=Path, default=None, help="Optional precomputed sweep.csv; checked against predictions")
    parser.add_argument("--out-dir", type=Path, default=Path("threshold_plots"))
    parser.add_argument("--threshold", type=float, default=None, help="Use exact threshold instead of selecting one")
    parser.add_argument("--min-precision", type=float, default=None, help="Precision constraint, e.g. 0.99")
    parser.add_argument("--max-fpr", type=float, default=None, help="False-positive-rate constraint, e.g. 0.01")
    parser.add_argument("--zoom-low", type=float, default=0.97)
    parser.add_argument("--zoom-high", type=float, default=0.985)
    args = parser.parse_args()
    for x in [args.threshold, args.min_precision, args.max_fpr]:
        if x is not None and not 0 <= x <= 1:
            parser.error("Threshold, min-precision and max-fpr must be between 0 and 1")
    if not 0 <= args.zoom_low < args.zoom_high <= 1:
        parser.error("Require 0 <= zoom-low < zoom-high <= 1")
    df = load_predictions(args.predictions)
    sweep = load_sweep_or_rebuild(args.sweep, df)
    threshold, method = pick_threshold(sweep, args.threshold, args.min_precision, args.max_fpr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    m05 = metrics_at(df, 0.5)
    mstar = metrics_at(df, threshold)
    print(f"Records: {len(df)}, Indian: {(df.actual=='INDIAN').sum()}, Not Indian: {(df.actual=='NOT_INDIAN').sum()}")
    print(f"Chosen threshold: {threshold:.5f} ({method})")
    print(f"At 0.5000: TP={m05['tp']}, FP={m05['fp']}, FN={m05['fn']}, TN={m05['tn']}; "
          f"precision={m05['precision']:.3f}, recall={m05['recall']:.3f}")
    print(f"At {threshold:.5f}: TP={mstar['tp']}, FP={mstar['fp']}, FN={mstar['fn']}, TN={mstar['tn']}; "
          f"precision={mstar['precision']:.3f}, recall={mstar['recall']:.3f}")
    plot_scatter(df, threshold, args.out_dir / "01_confidence_vs_ground_truth.png")
    plot_distribution(df, threshold, args.out_dir / "02_confidence_distribution.png")
    plot_precision_recall(sweep, threshold, args.out_dir / "03_precision_recall_f1_vs_threshold.png")
    plot_error_counts(sweep, threshold, args.out_dir / "04_fp_fn_vs_threshold.png")
    plot_zoom(sweep, df, threshold, args.zoom_low, args.zoom_high,
              args.out_dir / "05_precision_recall_zoom.png")
    save_error_cases(df, 0.5, args.out_dir / "errors_at_0.50.csv")
    save_error_cases(df, threshold, args.out_dir / "errors_at_selected_threshold.csv")
    (args.out_dir / "summary.json").write_text(json.dumps({"n": len(df), "threshold": threshold,
                   "selection_method": method, "baseline_0_5": m05, "selected": mstar}, indent=2),
                   encoding="utf-8")
    print(f"  {args.out_dir / 'summary.json'}")
    print("\nNOTE: Select thresholds on development data only; use a separate untouched test set for final reporting.")


if __name__ == "__main__":
    main()
