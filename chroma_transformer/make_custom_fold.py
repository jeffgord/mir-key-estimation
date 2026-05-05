import pandas as pd
import numpy as np
import os

SEED = 42
VAL_FRAC = 0.1
DATA_PATH = "data/fmakv2.csv"
OUT_DIR = "out/folds/fold_10"

df = pd.read_csv(DATA_PATH)

folder = df["track_id"] // 1000
test_df = df[folder <= 19]
rest_df = df[folder > 19]

rng = np.random.default_rng(SEED)
rest_ids = rest_df["track_id"].values.copy()
rng.shuffle(rest_ids)

n_val = int(len(rest_ids) * VAL_FRAC)
val_ids = rest_ids[:n_val]
train_ids = rest_ids[n_val:]

train_df = rest_df[rest_df["track_id"].isin(train_ids)]
val_df   = rest_df[rest_df["track_id"].isin(val_ids)]

os.makedirs(OUT_DIR, exist_ok=True)
train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

print(f"fold_10: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
print(f"Saved to {OUT_DIR}/")
