# python extract_features.py --data-dir data --output-dir features --n-jobs 4 --max-folder 019

import argparse
import os
import warnings
import numpy as np
import librosa
import librosa.util.utils
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=FutureWarning, message=".*__audioread_load.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*PySoundFile failed.*")


# numba's @guvectorize + @stencil segfaults on Python 3.13 / Apple Silicon.
# Replace the inner kernel with a pure numpy equivalent; the public `localmax`
# wrapper is regular Python and looks up `_localmax` at call time, so this works.
def _localmax_numpy(x, y):
    y[...] = False
    if x.shape[-1] > 2:
        y[..., 1:-1] = (x[..., 1:-1] > x[..., :-2]) & (x[..., 1:-1] >= x[..., 2:])

librosa.util.utils._localmax = _localmax_numpy


def _init_worker():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*__audioread_load.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*PySoundFile failed.*")

    # Loky workers are spawned fresh and don't inherit the main process patch,
    # so reapply it here before any tasks run.
    import librosa.util.utils as _luu

    def _lm(x, y):
        y[...] = False
        if x.shape[-1] > 2:
            y[..., 1:-1] = (x[..., 1:-1] > x[..., :-2]) & (x[..., 1:-1] >= x[..., 2:])

    _luu._localmax = _lm


DURATION = 180  # seconds (3 minutes)
SR = 22050
MAX_SAMPLES = DURATION * SR


def extract_chroma(file_path):
    y, sr = librosa.load(file_path, sr=SR)

    if len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    else:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)))

    y_harm = librosa.effects.harmonic(y, margin=8)

    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_filtered = librosa.decompose.nn_filter(
        chroma_harm,
        aggregate=np.median,
        metric='cosine'
    )
    chroma = np.minimum(chroma_harm, chroma_filtered)

    return chroma  # shape: (12, T)


def process_track(row, data_dir, output_dir):
    track_id = row['track_id']
    full_id = str(track_id).zfill(6)
    folder = full_id[:3]
    file_path = os.path.join(data_dir, folder, f"{full_id}.mp3")
    out_path = os.path.join(output_dir, f"{full_id}.npy")

    if not os.path.exists(file_path):
        return {'track_id': track_id, 'status': 'missing', 'out_path': None}

    if os.path.exists(out_path):
        return {'track_id': track_id, 'status': 'ok', 'out_path': out_path}

    try:
        chroma = extract_chroma(file_path)
        np.save(out_path, chroma)
        return {'track_id': track_id, 'status': 'ok', 'out_path': out_path}
    except Exception as e:
        return {'track_id': track_id, 'status': f'error: {e}', 'out_path': None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Root folder containing audio subfolders and metadata CSV")
    parser.add_argument("--output-dir", default="features", help="Folder to save .npy feature files and manifest")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--max-folder", default=None,
                        help="Only process tracks whose folder prefix is <= this value (e.g. '019')")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metadata_path = os.path.join(args.data_dir, "fma_keys_metadata.csv")
    df = pd.read_csv(metadata_path)

    if args.max_folder:
        df = df[df['track_id'].apply(lambda tid: str(tid).zfill(6)[:3] <= args.max_folder)]
        df = df.reset_index(drop=True)

    total = len(df)
    results = []
    gen = Parallel(n_jobs=args.n_jobs, return_as="generator", initializer=_init_worker)(
        delayed(process_track)(row, args.data_dir, args.output_dir)
        for _, row in df.iterrows()
    )
    for i, result in enumerate(gen):
        results.append(result)
        print(f"[{i + 1}/{total}] {result['track_id']} — {result['status']}", flush=True)

    results_df = pd.DataFrame(results)
    manifest_path = os.path.join(args.output_dir, "features_manifest.csv")
    results_df.to_csv(manifest_path, index=False)

    ok = (results_df['status'] == 'ok').sum()
    print(f"\nDone. {ok}/{len(df)} tracks extracted successfully.")
    print(f"Features saved to '{args.output_dir}/', manifest at '{manifest_path}'.")

    sample = results_df[results_df['status'] == 'ok'].iloc[0]
    sample_chroma = np.load(sample['out_path'])
    print(f"Feature shape per track: {sample_chroma.shape}  (12 chroma bins × {sample_chroma.shape[1]} time frames)")