#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing for image-based polysomnography (PSG) analysis.

Pipeline
--------
    EDF  ->  channel selection (18 fixed channels, alias resolution)
         ->  resampling the full selected recording to 100 Hz
         ->  temporal cropping (recording seconds [900, 22500), up to 6 hours)
         ->  Savitzky-Golay smoothing (window 25, polyorder 3)
         ->  per-channel min-max normalisation to [0, 1]
         ->  30-second epoch segmentation (3,000 samples per epoch)
         ->  rendering to a 600 x 600 raster (18 stacked rows, white traces on
             black) and bicubic downsampling to 224 x 224 RGB

Usage
-----
    python psg_edf_to_images.py --input  /path/to/edf_root \
                                --output /path/to/image_root

    # reproduce a single recording only
    python psg_edf_to_images.py -i data/ -o images/ --limit 1

Outputs
-------
    <output>/<patient_id>_image/<patient_id>_T<start_second>.png
    <output>/preprocessing_manifest.csv

Requirements
------------
    mne, numpy, scipy, matplotlib, pillow
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless rendering; must precede pyplot import
import matplotlib.pyplot as plt
import mne
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter

DESIRED_CHANNELS: Tuple[str, ...] = (
    "F3", "F4", "C3", "C4", "O1", "O2",          # EEG   (6)
    "EOG Left", "EOG Right",                      # EOG   (2)
    "Chin", "Left Leg", "Right Leg",              # EMG   (3)
    "EKG",                                        # ECG   (1)
    "Nasal Pressure",                             # Airflow (1)
    "Abdomen",                                    # Respiratory effort (1)
    "SpO2",                                       # Oxygen saturation (1)
    "Snoring Sensor",                             # Snoring (1)
    "Gravity X", "Gravity Y",                     # Body position (2)
)

CHANNEL_ALIASES: Dict[str, List[str]] = {
    "EOG Left": ["EOG Left", "LOC"],
    "EOG Right": ["EOG Right", "ROC"],
    "Left Leg": ["Leg Left", "Left Leg", "Left Geg"],
    "Right Leg": ["Leg Right", "Right Leg"],
    "Chin": ["Chin", "Chin EMG"],
    "EKG": ["EKG", "ECG"],
    "SpO2": ["SpO2", "SaO2"],
    "Snoring Sensor": ["Snoring Sensor", "Snore", "Snoring"],
}

TARGET_SFREQ = 100.0        # Hz, after resampling
EPOCH_SECONDS = 30.0        # length of one epoch
START_SECOND = 900.0        # 15 min after recording onset
N_EPOCHS = 720              # 720 x 30 s = 6 h
SAVGOL_WINDOW = 25          # samples
SAVGOL_POLYORDER = 3

FIG_INCHES = 6.0            # 6 x 6 in
FIG_DPI = 100               # -> 600 x 600 px raster
LINE_WIDTH = 0.8            # pt, white polyline
OUTPUT_PIXELS = 224         # final image side length

LOGGER = logging.getLogger("psg2img")


# --------------------------------------------------------------------------- #
# Data containers                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class RecordingResult:
    patient_id: str
    edf_path: str
    status: str
    original_sfreq: float = float("nan")
    duration_s: float = float("nan")
    n_epochs_written: int = 0
    missing_channels: List[str] = field(default_factory=list)
    message: str = ""


# --------------------------------------------------------------------------- #
# Channel handling                                                             #
# --------------------------------------------------------------------------- #

def resolve_channel_map(available: Sequence[str]) -> Dict[str, Optional[str]]:
    lookup = {name.strip().lower(): name for name in available}
    mapping: Dict[str, Optional[str]] = {}
    for canonical in DESIRED_CHANNELS:
        aliases = CHANNEL_ALIASES.get(canonical, [canonical])
        if canonical not in aliases:
            aliases = [canonical, *aliases]
        resolved = None
        for alias in aliases:
            hit = lookup.get(alias.strip().lower())
            if hit is not None:
                resolved = hit
                break
        mapping[canonical] = resolved
    return mapping


# --------------------------------------------------------------------------- #
# Signal preparation                                                           #
# --------------------------------------------------------------------------- #

def minmax_normalise(signal: np.ndarray) -> np.ndarray:
    finite = np.isfinite(signal)
    if not finite.any():
        return np.full_like(signal, np.nan, dtype=np.float64)
    lo = float(np.min(signal[finite]))
    hi = float(np.max(signal[finite]))
    if hi == lo:
        out = np.zeros_like(signal, dtype=np.float64)
        out[~finite] = np.nan
        return out
    return (signal.astype(np.float64) - lo) / (hi - lo)


def smooth_channel(signal: np.ndarray, patient_id: str, channel: str) -> np.ndarray:
    if signal.size < SAVGOL_WINDOW:
        LOGGER.warning("[%s] %s: too short for Savitzky-Golay, left unfiltered",
                       patient_id, channel)
        return signal.astype(np.float64)
    work = np.nan_to_num(signal.astype(np.float64), nan=0.0,
                         posinf=0.0, neginf=0.0)
    try:
        filtered = savgol_filter(work, window_length=SAVGOL_WINDOW,
                                 polyorder=SAVGOL_POLYORDER)
    except Exception as exc:  # noqa: BLE001 - filtering must not abort a subject
        LOGGER.warning("[%s] %s: Savitzky-Golay failed (%s), left unfiltered",
                       patient_id, channel, exc)
        return work
    filtered[~np.isfinite(signal)] = np.nan  # keep gaps flagged as missing
    return filtered


def prepare_signals(
    raw: mne.io.BaseRaw,
    patient_id: str,
    start_second: float,
    n_epochs: int,
) -> Tuple[np.ndarray, Dict[str, Optional[np.ndarray]], List[str]]:
    channel_map = resolve_channel_map(raw.ch_names)
    present = [name for name in channel_map.values() if name is not None]
    if not present:
        raise ValueError("none of the 18 target channels are present")

    raw.pick(present)

    # Resample the full selected recording BEFORE extracting the analysis window.
    original_sfreq = float(raw.info["sfreq"])
    original_duration = raw.n_times / original_sfreq
    if not np.isclose(original_sfreq, 256.0):
        LOGGER.warning("[%s] EDF sampling rate is %.3f Hz, expected 256 Hz; "
                       "resampling the actual rate to 100 Hz",
                       patient_id, original_sfreq)
    raw.resample(TARGET_SFREQ, npad="auto", verbose=False)

    # Index the requested window on the resampled 100-Hz grid.
    # Floor the original duration to avoid extending short recordings by rounding.
    available_samples = min(raw.n_times,
                            int(np.floor(original_duration * TARGET_SFREQ + 1e-7)))
    start_idx = int(round(start_second * TARGET_SFREQ))
    requested_samples = n_epochs * int(round(EPOCH_SECONDS * TARGET_SFREQ))
    stop_idx = min(start_idx + requested_samples, available_samples)
    if start_idx >= available_samples or stop_idx <= start_idx:
        raise ValueError("recording has no samples after the requested start")
    raw.crop(tmin=start_idx / TARGET_SFREQ,
             tmax=(stop_idx - 1) / TARGET_SFREQ,
             include_tmax=True)
    data = raw.get_data()
    # Recording-relative timestamps: T900, T930, ... for default settings.
    times = (start_idx + np.arange(data.shape[1])) / TARGET_SFREQ
    by_name = {name: data[idx] for idx, name in enumerate(raw.ch_names)}

    signals: Dict[str, Optional[np.ndarray]] = {}
    missing: List[str] = []
    for canonical in DESIRED_CHANNELS:
        source = channel_map[canonical]
        if source is None:
            signals[canonical] = None
            missing.append(canonical)
            continue

        trace = by_name[source]
        trace = smooth_channel(trace, patient_id, canonical)
        trace = minmax_normalise(trace)

        if not np.isfinite(trace).any():
            # Present in the header but carrying no usable samples: treated as
            # missing so that the row is rendered blank rather than flat.
            signals[canonical] = None
            missing.append(canonical)
        else:
            signals[canonical] = trace

    return times, signals, missing


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #

def render_epoch(
    time: np.ndarray,
    signals: Dict[str, Optional[np.ndarray]],
    tight_bbox: bool,
) -> Image.Image:
    n_rows = len(DESIRED_CHANNELS)
    fig = plt.figure(figsize=(FIG_INCHES, FIG_INCHES),
                     facecolor="black", dpi=FIG_DPI)
    try:
        for idx, canonical in enumerate(DESIRED_CHANNELS):
            bottom = 1.0 - (idx + 1) / n_rows
            ax = fig.add_axes([0.0, bottom, 1.0, 1.0 / n_rows])
            ax.set_facecolor("black")
            ax.axis("off")
            ax.set_ylim(0, 1)

            trace = signals.get(canonical)
            if trace is None:
                continue
            ax.plot(time, np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0),
                    color="white", linewidth=LINE_WIDTH, antialiased=True)

        buffer = io.BytesIO()
        save_kwargs = {"facecolor": "black", "dpi": FIG_DPI}
        if tight_bbox:
            save_kwargs.update(bbox_inches="tight", pad_inches=0)
        fig.savefig(buffer, format="png", **save_kwargs)
    finally:
        plt.close(fig)

    buffer.seek(0)
    with Image.open(buffer) as raster:
        return raster.convert("RGB").resize(
            (OUTPUT_PIXELS, OUTPUT_PIXELS), Image.BICUBIC
        )


# --------------------------------------------------------------------------- #
# Per-recording driver                                                         #
# --------------------------------------------------------------------------- #

def process_recording(
    edf_path: Path,
    output_root: Path,
    patient_id: str,
    args: argparse.Namespace,
) -> RecordingResult:
    result = RecordingResult(patient_id=patient_id, edf_path=str(edf_path),
                             status="failed")
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as exc:  # noqa: BLE001
        result.message = f"read error: {exc}"
        LOGGER.error("[%s] cannot read %s: %s", patient_id, edf_path, exc)
        return result

    result.original_sfreq = float(raw.info["sfreq"])
    result.duration_s = float(raw.n_times) / result.original_sfreq

    try:
        time, signals, missing = prepare_signals(
            raw, patient_id, args.start_second, args.n_epochs
        )
    except Exception as exc:  # noqa: BLE001
        result.message = f"preprocessing error: {exc}"
        LOGGER.error("[%s] preprocessing failed: %s", patient_id, exc)
        return result
    finally:
        del raw

    result.missing_channels = missing
    if missing:
        LOGGER.info("[%s] %d channel(s) rendered blank: %s",
                    patient_id, len(missing), ", ".join(missing))

    samples_per_epoch = int(round(EPOCH_SECONDS * TARGET_SFREQ))
    n_available = time.size // samples_per_epoch
    if n_available < args.n_epochs:
        LOGGER.warning("[%s] only %d of %d epochs available",
                       patient_id, n_available, args.n_epochs)
    if n_available == 0:
        result.message = "no complete epoch within the analysis window"
        return result

    save_dir = output_root / f"{patient_id}_image"
    save_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for epoch_idx in range(min(n_available, args.n_epochs)):
        lo = epoch_idx * samples_per_epoch
        hi = lo + samples_per_epoch
        epoch_time = time[lo:hi]
        target = save_dir / f"{patient_id}_T{int(epoch_time[0])}.png"
        if target.exists() and not args.overwrite:
            written += 1
            continue

        epoch_signals = {
            name: (None if trace is None else trace[lo:hi])
            for name, trace in signals.items()
        }
        image = render_epoch(epoch_time, epoch_signals, tight_bbox=args.tight_bbox)
        image.save(target)
        written += 1

    result.status = "ok"
    result.n_epochs_written = written
    LOGGER.info("[%s] %d image(s) -> %s", patient_id, written, save_dir)
    return result


# --------------------------------------------------------------------------- #
# Discovery and CLI                                                            #
# --------------------------------------------------------------------------- #

def discover_edfs(root: Path, pattern: str, recursive: bool) -> List[Path]:
    globber = root.rglob if recursive else root.glob
    files = sorted({path for path in globber(pattern) if path.is_file()})
    return files


def derive_patient_id(edf_path: Path, root: Path, mode: str) -> str:
    parent = edf_path.parent.name
    stem = edf_path.stem
    if mode == "parent":
        return parent
    if mode == "stem":
        return stem
    if edf_path.parent == root or not parent:
        return stem
    siblings = [p for p in edf_path.parent.glob("*.edf") if p.is_file()]
    return parent if len(siblings) == 1 else f"{parent}_{stem}"


def write_manifest(path: Path, results: Sequence[RecordingResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "patient_id", "edf_path", "status", "original_sfreq_hz",
            "duration_s", "n_epochs_written", "n_missing_channels",
            "missing_channels", "message",
        ])
        for item in results:
            writer.writerow([
                item.patient_id, item.edf_path, item.status,
                f"{item.original_sfreq:.4f}", f"{item.duration_s:.2f}",
                item.n_epochs_written, len(item.missing_channels),
                ";".join(item.missing_channels), item.message,
            ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PSG recordings in EDF format into 224 x 224 "
                    "epoch images (Supplementary Algorithm 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, type=Path,
                        help="directory searched for EDF recordings")
    parser.add_argument("-o", "--output", required=True, type=Path,
                        help="directory receiving <patient_id>_image folders")
    parser.add_argument("--pattern", default="*.edf",
                        help="glob pattern for EDF files")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false",
                        help="do not descend into subdirectories")
    parser.add_argument("--id-mode", choices=("auto", "parent", "stem"),
                        default="auto",
                        help="how the subject identifier is derived")
    parser.add_argument("--start-second", type=float, default=START_SECOND,
                        help="onset of the analysis window, in seconds")
    parser.add_argument("--n-epochs", type=int, default=N_EPOCHS,
                        help="number of consecutive 30-second epochs to render")
    parser.add_argument("--preprocess-scope", "--minmax-scope",
                        dest="preprocess_scope",
                        choices=("window",), default="window",
                        help="compatibility option: only the cropped window "
                             "is supported")
    parser.add_argument("--no-tight-bbox", dest="tight_bbox",
                        action="store_false",
                        help="save the raw 600 x 600 canvas instead of the "
                             "tight bounding box")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-render images that already exist")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most this many recordings")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="manifest path (default: <output>/preprocessing_manifest.csv)")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.set_defaults(tight_bbox=False)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not np.isfinite(args.start_second) or args.start_second < 0:
        raise SystemExit("--start-second must be finite and non-negative")
    if not 1 <= args.n_epochs <= N_EPOCHS:
        raise SystemExit("--n-epochs must be between 1 and 720")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    mne.set_log_level("ERROR")

    if not args.input.is_dir():
        LOGGER.error("input directory not found: %s", args.input)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    edf_files = discover_edfs(args.input, args.pattern, args.recursive)
    if args.limit is not None:
        edf_files = edf_files[: args.limit]
    if not edf_files:
        LOGGER.error("no file matching %r under %s", args.pattern, args.input)
        return 1

    LOGGER.info("%d recording(s) queued | window %.0f-%.0f s | "
                "smoothing/min-max scope: %s",
                len(edf_files), args.start_second,
                args.start_second + args.n_epochs * EPOCH_SECONDS,
                args.preprocess_scope)

    results: List[RecordingResult] = []
    for order, edf_path in enumerate(edf_files, start=1):
        patient_id = derive_patient_id(edf_path, args.input, args.id_mode)
        LOGGER.info("(%d/%d) %s", order, len(edf_files), patient_id)
        results.append(process_recording(edf_path, args.output, patient_id, args))

    manifest_path = args.manifest or (args.output / "preprocessing_manifest.csv")
    write_manifest(manifest_path, results)

    n_ok = sum(1 for item in results if item.status == "ok")
    n_images = sum(item.n_epochs_written for item in results)
    LOGGER.info("done: %d/%d recording(s), %d image(s); manifest -> %s",
                n_ok, len(results), n_images, manifest_path)
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
