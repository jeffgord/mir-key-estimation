#!/usr/bin/env python3
"""Run Krumhansl-Schmuckler key estimation on every track in fma_keys_metadata.csv.

Resumable: predictions are appended to krumhansl_predictions.csv one row at a
time and flushed after each write, so re-running the script picks up where the
previous run left off (any track_id already present in the output is skipped).

Per-track progress is printed to stdout; redirect to a file if you want a log
(e.g. `python krumhansl_predict_keys.py | tee krumhansl_predictions.log`).

Usage:
    # Defaults: reads data/fma_keys_metadata.csv and audio under
    # data/NNN/NNNNNN.mp3, writes outputs into the cwd.
    python krumhansl_predict_keys.py

    # Point at a different data dir (must contain fma_keys_metadata.csv plus
    # the NNN/NNNNNN.mp3 tree) and a separate output dir:
    python krumhansl_predict_keys.py --data-dir /scratch/.../data \\
        --out-dir /scratch/out --workers 16
"""
import argparse
import csv
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import librosa

CSV_NAME = "krumhansl_predictions.csv"

ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def krumhansl_schmukler(file_path: str):
    y, sr = librosa.load(file_path)

    y_harm = librosa.effects.harmonic(y, margin=8)

    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_filtered = librosa.decompose.nn_filter(
        chroma_harm,
        aggregate=np.median,
        metric='cosine')
    chroma = np.minimum(chroma_harm, chroma_filtered)

    avg_chroma = np.mean(chroma, axis=1)

    best_correlation = -1.0
    best_key = None
    for i in range(12):
        rolled = np.roll(avg_chroma, -i)
        major_correlation = np.corrcoef(rolled, MAJOR_PROFILE)[0, 1]
        minor_correlation = np.corrcoef(rolled, MINOR_PROFILE)[0, 1]

        if major_correlation > best_correlation:
            best_correlation = major_correlation
            best_key = f"{ROOTS[i]} Major"
        if minor_correlation > best_correlation:
            best_correlation = minor_correlation
            best_key = f"{ROOTS[i]} Minor"

    return best_key, float(best_correlation)


def track_path(data_dir: Path, track_id: int) -> Path:
    full_id = str(track_id).zfill(6)
    return data_dir / full_id[:3] / f"{full_id}.mp3"


def predict_one(data_dir_str: str, track_id: int):
    data_dir = Path(data_dir_str)
    file_path = track_path(data_dir, track_id)
    try:
        predicted_key, correlation = krumhansl_schmukler(str(file_path))
        return track_id, predicted_key, correlation, ""
    except Exception as e:
        return track_id, "", "", repr(e)


def load_done_ids(output_path: Path) -> set[int]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    done = set()
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["track_id"]))
            except (KeyError, ValueError):
                continue
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing fma_keys_metadata.csv "
                             "and the NNN/NNNNNN.mp3 tree")
    parser.add_argument("--out-dir", default=".",
                        help=f"Directory for {CSV_NAME}")
    parser.add_argument("--workers", type=int, default=8,
                        help="ProcessPoolExecutor max_workers")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    metadata_path = data_dir / "fma_keys_metadata.csv"
    output_path = out_dir / CSV_NAME

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)
    all_ids = df["track_id"].astype(int).tolist()
    total = len(all_ids)

    done_ids = load_done_ids(output_path)
    pending = [tid for tid in all_ids if tid not in done_ids]

    summary = (f"Total: {total}  Done: {len(done_ids)}  "
               f"Pending: {len(pending)}")
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"{ts} START — {summary}", flush=True)

    if not pending:
        return

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with open(output_path, "a", newline="", buffering=1) as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["track_id", "predicted_key", "correlation", "error"])
            f.flush()

        completed = len(done_ids)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(predict_one, str(data_dir), tid): tid
                for tid in pending
            }
            for future in as_completed(futures):
                track_id, predicted_key, correlation, error = future.result()
                writer.writerow([track_id, predicted_key, correlation, error])
                f.flush()

                completed += 1
                ts = datetime.now().isoformat(timespec="seconds")
                if error:
                    result = f"ERROR {error}"
                else:
                    result = f"{predicted_key} (r={correlation:.3f})"
                print(f"{ts} [{completed}/{total}] "
                      f"{str(track_id).zfill(6)} -> {result}", flush=True)


if __name__ == "__main__":
    main()
