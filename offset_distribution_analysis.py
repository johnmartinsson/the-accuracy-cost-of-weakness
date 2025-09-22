#!/usr/bin/env python3
"""
offset_distribution_analyis.py
======================================
Download the **NIGENS** corpus (≈ 2 GB the first time), extract it, and plot
*offset* distributions for the **dog** and **baby** classes.

Offset definition
-----------------
```
offset = segment_end − event_start   (0 ≤ offset ≤ d_e + d_q)
```
where
* `d_e` – corpus‑wide **median event length**
* `d_q = f × d_e` – query window length for fraction *f*

For each fraction *f* the script tiles each WAV file with non‑overlapping
windows of length `d_q` and records `offset` whenever the window overlaps an
annotated event.

Outputs
-------
```
<root>/NIGENS/dog_offset_distribution.png
<root>/NIGENS/baby_offset_distribution.png
```
Each PNG overlays one histogram per fraction *f*.

Quick‑start
-----------
```bash
pip install numpy matplotlib requests

python offset_distribution_analyis.py \
       --root ~/datasets \
       --fractions 0.2 1 5
```
Add `--force-download` to redownload/extract even if cached.
"""
from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
import zipfile
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
import wave

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NIGENS_ZIP_URL = "https://zenodo.org/records/2535878/files/NIGENS.zip"
CLASSES = ("dog", "baby")  # subfolders to analyse

# -----------------------------------------------------------------------------
# Download & extraction helpers
# -----------------------------------------------------------------------------

def download(url: str, dest: Path, force: bool = False) -> None:
    """Download *url* to *dest* showing a simple progress bar."""

    if dest.exists() and not force:
        print(f"[INFO] Using cached {dest}")
        return

    print(f"[INFO] Downloading {url} → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    bar = int(50 * downloaded / total)
                    sys.stdout.write(
                        f"\r[{'#' * bar}{'.' * (50 - bar)}] "
                        f"{downloaded // (1 << 20)} MB / {total // (1 << 20)} MB"
                    )
                    sys.stdout.flush()
    print("\n[INFO] Download complete.")


def extract(zip_path: Path, dest_root: Path, force: bool = False) -> Path:
    """Extract *zip_path* under *dest_root* and return the resulting NIGENS path."""

    nigens_path = dest_root / "NIGENS"
    marker = nigens_path / "README.md"
    if marker.exists() and not force:
        print("[INFO] Archive already extracted – skipping.")
        return nigens_path

    print(f"[INFO] Extracting {zip_path.name} …")
    dest_root.mkdir(parents=True, exist_ok=True)

    # 1) system unzip
    if shutil.which("unzip"):
        if subprocess.call(["unzip", "-q", str(zip_path), "-d", str(dest_root)]) == 0:
            print("[INFO] Extraction finished via system unzip.")
            return nigens_path
        print("[WARN] System unzip failed.")

    # 2) 7‑Zip
    if shutil.which("7z"):
        if subprocess.call(["7z", "x", "-y", str(zip_path), f"-o{dest_root}"]) == 0:
            print("[INFO] Extraction finished via 7‑Zip.")
            return nigens_path
        print("[WARN] 7‑Zip extraction failed.")

    raise RuntimeError("Cannot extract NIGENS.zip – unsupported compression and no external extractor found.")

# -----------------------------------------------------------------------------
# WAV / annotation helpers
# -----------------------------------------------------------------------------

def wav_duration(path: Path) -> float:
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        return wf.getnframes() / float(wf.getframerate())


def load_events(txt: Path) -> List[Tuple[float, float]]:
    events: List[Tuple[float, float]] = []
    with txt.open() as fh:
        for line in fh:
            try:
                s, e = map(float, line.split())
            except ValueError:
                continue
            if e > s:
                events.append((s, e))
    return events


def event_offsets(event: Tuple[float, float], seg_edges: np.ndarray) -> List[float]:
    a_e, b_e = event
    offs: List[float] = []
    for a_q, b_q in zip(seg_edges[:-1], seg_edges[1:]):
        if a_q < b_e and b_q > a_e:  # overlap
            offs.append(b_q - a_e)
    return offs

# -----------------------------------------------------------------------------
# Analysis function
# -----------------------------------------------------------------------------

def analyse_class(cls_dir: Path, fractions: List[float], output_dir: str) -> None:
    wav_files = sorted(cls_dir.glob("*.wav"))
    if not wav_files:
        print(f"[WARN] No WAV files in {cls_dir}")
        return

    # Collect all event durations to compute corpus‑wide median
    durations: List[float] = []
    for wav in wav_files:
        txt = wav.with_suffix(".wav.txt")
        if txt.exists():
            durations.extend([e - s for s, e in load_events(txt)])
    if not durations:
        print(f"[WARN] No annotations for {cls_dir.name}")
        return

    d_e = float(np.median(durations))
    print(f"[INFO] {cls_dir.name}: median d_e = {d_e:.3f} s")

    offsets_by_f: Dict[float, List[float]] = {f: [] for f in fractions}

    # Iterate WAV files and accumulate offsets
    for wav in wav_files:
        txt = wav.with_suffix(".wav.txt")
        events_starts = [s for s, _ in load_events(txt)]
        if not events_starts:
            continue
        events = [(s, s + d_e) for s in events_starts]  # fixed‑length proxy
        T = wav_duration(wav)
        for f in fractions:
            d_q = f * d_e
            edges = np.arange(0, T + d_q, d_q)
            for evt in events:
                offsets_by_f[f].extend(event_offsets(evt, edges))

    if all(len(v) == 0 for v in offsets_by_f.values()):
        print(f"[WARN] No overlaps for {cls_dir.name}")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(4.5, 3))
    bins = np.linspace(0, d_e * (1 + max(fractions)), 50)
    for f, col in zip(fractions, cycle(plt.get_cmap("tab10").colors)):
        ax.hist(offsets_by_f[f], bins=bins, alpha=0.6, edgecolor="black", color=col,
                label=fr"$d_q = {f:.1f}\,d_e$ ({f * d_e:.2f}s)")
    ax.set_xlabel(r"Offset ($b_q - a_e$) [s]")
    ax.set_ylabel("Count")
    ax.set_title(f"Offset distribution • {cls_dir.name}")
    ax.legend(title=r"Query length $d_q$")
    fig.tight_layout()
    out_png = output_dir / f"{cls_dir.name}_offset_distribution.png"
    

    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved → {out_png}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser("Visualise event–segment offsets for NIGENS")
    parser.add_argument("--root", type=Path, default=Path("~/datasets").expanduser(),
                        help="Folder for NIGENS.zip and extraction")
    parser.add_argument("--fractions", "-f", type=float, nargs="+", required=True,
                        metavar="F", help="Positive fractions f (d_q = f × d_e)")
    parser.add_argument("--plot-dir", type=Path, default=Path("./plots"),
                        help="Where to write PNGs (default: <root>/NIGENS)")
    parser.add_argument("--force-download", action="store_true", help="Redownload / re‑extract")
    args = parser.parse_args(argv)

    if any(f <= 0 for f in args.fractions):
        parser.error("All fractions must be positive.")

    zip_path = args.root / "NIGENS.zip"

    # 1. Download (if needed)
    download(NIGENS_ZIP_URL, zip_path, force=args.force_download)

    # 2. Extract (if needed) – returns path to <root>/NIGENS
    nigens_dir = extract(zip_path, args.root, force=args.force_download)

    # 3. Analyse requested classes
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    fractions_sorted = sorted(set(args.fractions))
    for cls in CLASSES:
        cls_dir = nigens_dir / cls
        if not cls_dir.exists():
            print(f"[ERROR] Expected {cls_dir} but not found – skipping.")
            continue
        analyse_class(cls_dir, fractions_sorted, output_dir=args.plot_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(" [INFO] Interrupted.")