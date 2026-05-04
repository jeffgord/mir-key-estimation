# python chroma_transformer.py <fold_number>
import argparse
import copy
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from pathlib import Path

class KeyLabelConverter:
    def key_to_label(key_mode):
        key, mode = key_mode.split(' ')
        key_to_int = {
            'C': 0,
            'C#': 1,
            'Db': 1,
            'D': 2,
            'D#': 3,
            'Eb': 3,
            'E': 4,
            'F': 5,
            'F#': 6,
            'Gb': 6,
            'G': 7,
            'G#': 8,
            'Ab': 8,
            'A': 9,
            'A#': 10,
            'Bb': 10,
            'B': 11
        }
        mode_to_int = {
            'minor': 0,
            'Major': 1
        }
        return torch.tensor(key_to_int[key] * 2 + mode_to_int[mode], dtype=torch.long)
    
    def label_to_key(label):
        int_to_key = {
            0: 'C',
            1: 'C#',
            2: 'D',
            3: 'D#',
            4: 'E',
            5: 'F',
            6: 'F#',
            7: 'G',
            8: 'G#',
            9: 'A',
            10: 'A#',
            11: 'B'
        }
        int_to_mode = {
            0: 'minor',
            1: 'Major'
        }
        key_int = label // 2
        mode_int = label % 2
        return f"{int_to_key[key_int]} {int_to_mode[mode_int]}"
    
    def get_num_classes():
        return 24

class ChromaDataset(Dataset):
    def __init__(self, metadata_path, features_path, augment=False):
        self.features_path = features_path
        self.augment = augment

        df = pd.read_csv(metadata_path)

        self.track_ids = df['track_id'].tolist()
        self.labels = df['key_and_mode'].apply(KeyLabelConverter.key_to_label).tolist()
        
        # Pre-load all features into memory as torch tensors
        self.features = {}
        for track_id in self.track_ids:
            track_id_str = str(track_id).zfill(6)
            feature_path = os.path.join(self.features_path, f"{track_id_str}.npy")
            if os.path.exists(feature_path):
                self.features[track_id] = torch.tensor(np.load(feature_path), dtype=torch.float32)
        print(f"Loaded {len(self.features)} feature files into memory")

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        chroma = self.features[track_id].clone()  # shape: (12, T)

        # Normalize: z-score normalization per chroma bin across time
        chroma = chroma - chroma.mean(dim=1, keepdim=True)
        chroma = chroma / (chroma.std(dim=1, keepdim=True) + 1e-8)

        label = self.labels[idx]

        if self.augment:
            k = np.random.randint(0, 12)
            chroma = torch.roll(chroma, k, dims=0)
            label = (label + 2 * k) % 24

        return chroma, label

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ChromaTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, chunk_size=500):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size

        self.input_projection = nn.Linear(12, d_model)
        self.pos_encoding = SinusoidalPE(d_model, max_len=chunk_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learned attention pooling over chunks
        self.chunk_attn = nn.Linear(d_model, 1)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, KeyLabelConverter.get_num_classes())
        )

    def forward(self, x):
        B, C, T = x.shape
        cs = self.chunk_size

        if T % cs != 0:
            x = torch.nn.functional.pad(x, (0, cs - T % cs))
            T = x.shape[2]

        num_chunks = T // cs

        # (B, 12, num_chunks, cs) -> (B*num_chunks, cs, 12)
        x = x.reshape(B, C, num_chunks, cs).permute(0, 2, 3, 1).reshape(B * num_chunks, cs, C)

        # Project, add PE, encode, mean pool within chunk -> (B*num_chunks, d_model)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)

        # Attention pooling across chunks -> (B, d_model)
        x = x.reshape(B, num_chunks, self.d_model)
        attn_weights = torch.softmax(self.chunk_attn(x), dim=1)  # (B, num_chunks, 1)
        x = (x * attn_weights).sum(dim=1)

        return self.output_head(x)

def collate_batch(batch):
    """Collate batch with variable-length chroma sequences."""
    chromas, labels = zip(*batch)
    
    # Find max length
    max_len = max(c.shape[1] for c in chromas)
    
    # Pad all to max_len
    padded_chromas = []
    for c in chromas:
        if c.shape[1] < max_len:
            padding = torch.zeros(12, max_len - c.shape[1])
            c = torch.cat([c, padding], dim=1)
        padded_chromas.append(c)
    
    chromas_tensor = torch.stack(padded_chromas)
    labels_tensor = torch.stack(labels)
    
    return chromas_tensor, labels_tensor

def train(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=200, patience=25, verbose=True):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda') if 'cuda' in str(device) else None

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(f"  Epoch {epoch+1}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}, best was epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    return best_epoch, best_val_acc

parser = argparse.ArgumentParser()
parser.add_argument("fold", type=int, help="Fold number")
args = parser.parse_args()

fold_dir     = f"out/folds/fold_{args.fold:02d}"
features_dir = "features"
train_csv    = os.path.join(fold_dir, "train.csv")
val_csv      = os.path.join(fold_dir, "val.csv")
test_csv     = os.path.join(fold_dir, "test.csv")

train_set    = ChromaDataset(train_csv, features_dir, augment=True)
val_set      = ChromaDataset(val_csv,   features_dir, augment=False)
dataset_test = ChromaDataset(test_csv,  features_dir, augment=False)

# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Metal Performance Shaders (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

print("Load in data")
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=collate_batch, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
val_loader   = DataLoader(val_set,   batch_size=128, shuffle=False, collate_fn=collate_batch, num_workers=4, pin_memory=True)
test_loader  = DataLoader(dataset_test, batch_size=128, shuffle=False, collate_fn=collate_batch, num_workers=4, pin_memory=True)

model = ChromaTransformer(d_model=64, nhead=4, num_layers=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: ChromaTransformer ({total_params:,} params)")
print(f"  - Input: 12-bin chroma, full song length, split into {model.chunk_size}-frame chunks")
print(f"  - Local transformer: d_model={model.d_model}, {model.transformer_encoder.num_layers} layers, 4 heads — attends within each chunk (+ sinusoidal PE)")
print(f"  - Chunk aggregation: learned attention pooling (weighted sum over chunks)")
print(f"  - Output head: 64-dim MLP -> 24 classes (12 keys x major/minor)")
print(f"  - Augmentation: random pitch shift (chroma roll) on train split")
print("Training...")

epochs_trained, final_acc = train(
    model, criterion, optimizer, train_loader, val_loader, device,
    num_epochs=200, verbose=True
)

print(f"\n" + "="*60)
print(f"Training complete!")
print(f"Converged in {epochs_trained} epochs with {final_acc:.2f}% val accuracy")
print(f"="*60)

# Evaluate on held-out test set
model.eval()
all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(y_batch.numpy())

all_probs  = np.concatenate(all_probs,  axis=0)
all_preds  = np.concatenate(all_preds,  axis=0)
all_labels = np.concatenate(all_labels, axis=0)

test_acc = 100. * (all_preds == all_labels).mean()
print(f"Test accuracy: {test_acc:.2f}%")

out_dir  = "out/predictions"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"fold_{args.fold:02d}_predictions.npz")
np.savez(out_path, y_true=all_labels, y_pred=all_preds, probs=all_probs)
print(f"Saved test predictions to {out_path}")

weights_dir = "out/weights"
os.makedirs(weights_dir, exist_ok=True)
weights_path = os.path.join(weights_dir, f"fold_{args.fold:02d}.pt")
torch.save(model.state_dict(), weights_path)
print(f"Saved model weights to {weights_path}")
