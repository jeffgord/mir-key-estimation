import pandas as pd
import numpy as np
import os

SEED = 42
N_FOLDS = 10
DATA_PATH = "data/fmakv2.csv"
OUT_DIR = "out/folds"

df = pd.read_csv(DATA_PATH)
track_ids = df["track_id"].values.copy()

rng = np.random.default_rng(SEED)
rng.shuffle(track_ids)

os.makedirs(OUT_DIR, exist_ok=True)

folds = np.array_split(track_ids, N_FOLDS)

for i, test_ids in enumerate(folds):
    val_idx = (i + 1) % N_FOLDS
    train_ids = np.concatenate([folds[j] for j in range(N_FOLDS) if j != i and j != val_idx])
    fold_dir = os.path.join(OUT_DIR, f"fold_{i:02d}")
    os.makedirs(fold_dir, exist_ok=True)

    train_df = df[df["track_id"].isin(train_ids)]
    val_df   = df[df["track_id"].isin(folds[val_idx])]
    test_df  = df[df["track_id"].isin(test_ids)]

    train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

    print(f"Fold {i:02d}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

print(f"\nSaved folds to {OUT_DIR}/")
