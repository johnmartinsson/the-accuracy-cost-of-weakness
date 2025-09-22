#!/usr/bin/env python3
"""
audioset_label_analysis.py (revised)
===================================
Computes how well weak (segment-level) AudioSet labels match strong (frame‑level)
labels **and** checks those empirical accuracies against our theoretical
annotation model.

Pipeline
--------
1. **Download** the three raw evaluation files if they aren’t present.
2. **Load** the strong labels, extract every occurrence of a chosen AudioSet
   class, and compute its *empirical* accuracy – the fraction of each 10‑second
   clip that is actually covered by that class, averaged across occurrences.
3. **Predict** theoretical accuracy across a grid of γ values using
   `theorems.P(d_e, d_q, γ)` (or a fallback proxy if the module isn’t
   installed).
4. **Plot & save**
   • *event_length_distribution_<label>.png* – histogram of event lengths.
   • *accuracy_curve_<label>.png* – theoretical curve **with** a dashed line at
     the empirical average; the script also prints the γ that best matches the
     strong‑vs‑weak accuracy.

Example
-------
    python audioset_label_analysis.py --label /m/0jbk --data-dir ./data \
                                      --output-dir ./plots

Command‑line flags let you analyse any label, tune γ range, or point to custom
folders. Run `-h/--help` for details.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# URLs to the canonical resources
# ---------------------------------------------------------------------------
EVAL_STRONG_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/strong/"
    "audioset_eval_strong.tsv"
)
EVAL_SEGMENTS_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
    "eval_segments.csv"
)
ONTOLOGY_URL = (
    "https://raw.githubusercontent.com/audioset/ontology/refs/heads/master/ontology.json"
)

# ---------------------------------------------------------------------------
# Optional theoretical model import
# ---------------------------------------------------------------------------
try:
    import theorems as thms  # noqa: F401

    def _P(d_e: float, d_q: float, gamma: float) -> float:  # type: ignore[misc]
        return thms.P(d_e, d_q, gamma)  # type: ignore[attr-defined]

except ModuleNotFoundError:

    def _P(d_e: float, d_q: float, gamma: float) -> float:  # noqa: D401
        # Simple placeholder – replaces real model if absent
        return min(1.0, (d_e / d_q) * gamma)

    print(
        "[WARN] Falling back to simplistic theoretical model. Install “theorems” "
        "package for accurate predictions.",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def download(url: str, dest: Path, force: bool = False) -> None:
    """Download *url* into *dest* unless it exists (or *force* is True)."""

    if dest.exists() and not force:
        print(f"[INFO] Using cached {dest.name}")
        return

    print(f"[INFO] Downloading {dest.name} …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                done += len(chunk)
                if total:
                    bar_len = int(50 * done / total)
                    sys.stdout.write("\r[{}{}]".format("#" * bar_len, "." * (50 - bar_len)))
                    sys.stdout.flush()
    print(" done.")


def get_label_name(ontology: Sequence[dict], label_id: str) -> str:
    for entry in ontology:
        if entry["id"] == label_id:
            return entry["name"]
    return label_id


# ---------------------------------------------------------------------------
# Theory helper
# ---------------------------------------------------------------------------

def theory_prediction(d_q: float, d_es: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """Vectorised mean‐accuracy prediction across γ values."""

    accs = np.empty_like(gammas)
    for i, γ in enumerate(gammas):
        accs[i] = np.mean([_P(d_e, d_q, γ) for d_e in d_es])
    return accs


# ---------------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------------

def analyse_label(
    df_strong: pd.DataFrame,
    ontology: Sequence[dict],
    label_id: str,
    segment_len: float,
    γ_start: float,
    γ_end: float,
    γ_steps: int,
    output_dir: Path,
) -> None:

    label_name = get_label_name(ontology, label_id)
    print(f"[INFO] Analysing {label_name} ({label_id}) …")

    subset = df_strong[df_strong["label"] == label_id]
    if subset.empty:
        print(f"[WARN] No occurrences of {label_id} in evaluation set.")
        return

    # Empirical data ----------------------------------------------------------------
    d_es = (subset["end_time_seconds"] - subset["start_time_seconds"]).values
    empirical_acc = float(np.mean(d_es / segment_len))

    # Figure 1 – event length distribution ------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_path = output_dir / f"event_length_distribution_{label_name.replace(' ', '_')}.png"

    plt.figure(figsize=(3.5, 2.5))
    plt.hist(d_es, bins=30)
    plt.title(f"Event Length Distribution • {label_name}")
    plt.xlabel("Event Length (s)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()

    # Theoretical curve -------------------------------------------------------------
    γs = np.linspace(γ_start, γ_end, γ_steps)
    theo_acc = theory_prediction(segment_len, d_es, γs)

    # γ that best matches empirical --------------------------------------------------
    best_idx = int(np.argmin(np.abs(theo_acc - empirical_acc)))
    best_γ = float(γs[best_idx])
    best_pred = float(theo_acc[best_idx])

    # Figure 2 – accuracy curve with overlay ----------------------------------------
    curve_path = output_dir / f"accuracy_curve_{label_name.replace(' ', '_')}.png"

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(γs, theo_acc, label="Theory")
    # Draw empirical line only over the γ range (x‑data, not axes coords)
    ax.hlines(empirical_acc, γ_start, γ_end, ls="--", label=f"Empirical ({empirical_acc:.3f})")
    ax.set_title(f"Accuracy vs γ • {label_name}")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300)
    plt.close()

    # Console summary ---------------------------------------------------------------
    print(f"[INFO] Empirical average accuracy : {empirical_acc:.3f}")
    print(f"[INFO] Best γ match              : {best_γ:.3f} (theory={best_pred:.3f})")
    print(f"[INFO] Saved → {hist_path.name}, {curve_path.name}\n")


# ---------------------------------------------------------------------------
# Argument parsing & program entry
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("AudioSet annotation accuracy checker")
    p.add_argument("--label", default="/m/0jbk", help="AudioSet ontology ID to study")
    p.add_argument("--data-dir", type=Path, default="./data", help="Folder for raw files")
    p.add_argument("--output-dir", type=Path, default="./plots", help="Where to write PNGs")
    p.add_argument("--segment-length", type=float, default=10.0, help="d_q in seconds")
    p.add_argument("--gamma-start", type=float, default=0.01, help="γ grid start (inclusive)")
    p.add_argument("--gamma-end", type=float, default=1.0, help="γ grid end (inclusive)")
    p.add_argument("--gamma-steps", type=int, default=100, help="Number of γ samples")
    p.add_argument("--force-download", action="store_true", help="Redownload resources")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve paths -----------------------------------------------------------
    data_dir = args.data_dir
    strong_path = data_dir / "audioset_eval_strong.tsv"
    segments_path = data_dir / "eval_segments.csv"  # downloaded but unused here
    ontology_path = data_dir / "ontology.json"

    # Downloads ---------------------------------------------------------------
    download(EVAL_STRONG_URL, strong_path, force=args.force_download)
    download(EVAL_SEGMENTS_URL, segments_path, force=args.force_download)
    download(ONTOLOGY_URL, ontology_path, force=args.force_download)

    # Load --------------------------------------------------------------------
    df_strong = pd.read_csv(strong_path, sep="\t")
    ontology = json.loads(ontology_path.read_text())

    # Analyse ---------------------------------------------------------------
    analyse_label(
        df_strong=df_strong,
        ontology=ontology,
        label_id=args.label,
        segment_len=args.segment_length,
        γ_start=args.gamma_start,
        γ_end=args.gamma_end,
        γ_steps=args.gamma_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")

