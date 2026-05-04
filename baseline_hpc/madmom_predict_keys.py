#!/usr/bin/env python3
"""Run madmom CNN key recognition on every track in fma_keys_metadata.csv.

Resumable: predictions are appended to madmom_predictions.csv one row at a time
and flushed after each write, so re-running the script picks up where the
previous run left off (any track_id already present in the output is skipped).

Per-track progress is printed to stdout; redirect to a file if you want a log
(e.g. `python madmom_predict_keys.py | tee madmom_predictions.log`).

Usage:
    # Defaults: reads data/fma_keys_metadata.csv and audio under
    # data/NNN/NNNNNN.mp3, writes outputs into the cwd.
    python madmom_predict_keys.py

    # Point at a different data dir (must contain fma_keys_metadata.csv plus
    # the NNN/NNNNNN.mp3 tree) and a separate output dir:
    python madmom_predict_keys.py --data-dir /scratch/.../data \\
        --out-dir /scratch/out --workers 16
"""
import argparse
import csv
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*dtype.*align.*")

import pandas as pd

import madmom

CSV_NAME = "madmom_predictions.csv"


def track_path(data_dir: Path, track_id: int) -> Path:
    full_id = str(track_id).zfill(6)
    return data_dir / full_id[:3] / f"{full_id}.mp3"


def predict_one(recognizer, data_dir: Path, track_id: int):
    file_path = track_path(data_dir, track_id)
    try:
        predictions = recognizer(str(file_path))
        predicted_key = madmom.features.key.key_prediction_to_label(predictions)
        return track_id, predicted_key, ""
    except Exception as e:
        return track_id, "", repr(e)


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
                        help="ThreadPoolExecutor max_workers")
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
    recognizer = madmom.features.key.CNNKeyRecognitionProcessor()

    with open(output_path, "a", newline="", buffering=1) as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["track_id", "predicted_key", "error"])
            f.flush()

        completed = len(done_ids)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(predict_one, recognizer, data_dir, tid): tid
                for tid in pending
            }
            for future in as_completed(futures):
                track_id, predicted_key, error = future.result()
                writer.writerow([track_id, predicted_key, error])
                f.flush()

                completed += 1
                ts = datetime.now().isoformat(timespec="seconds")
                result = f"ERROR {error}" if error else predicted_key
                print(f"{ts} [{completed}/{total}] "
                      f"{str(track_id).zfill(6)} -> {result}", flush=True)


if __name__ == "__main__":
    main()
