import mir_eval
import mirdata
import os
import urllib.request
import librosa
import numpy as np
import madmom
from chroma_transformer import ChromaTransformer, KeyLabelConverter, extract_features
import torch
from music21 import key, scale
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def download_metadata(data_home):
    """
    Download FMAKv2 from Zenodo.

    This is an updated version of the metadata file with corrected annotations. 
    """
    os.makedirs(data_home, exist_ok=True)
    file_path = os.path.join(data_home, 'fma_keys_metadata.csv')
    
    if os.path.exists(file_path):
        print(f"V2 annotations already exist at {file_path}. Skipping download.")
        return file_path
    
    url = 'https://zenodo.org/records/12759100/files/fmakv2.csv'
    print(f"Downloading FMAKv2 from {url}...")
    urllib.request.urlretrieve(url, file_path)
    print(f"Downloaded to {file_path}")
    return file_path


def load_data(data_home='subset/', subset=True):
    """
    Load the FMAKv2 dataset using mirdata, ensuring that the corrected metadata is used.
    """
    dataset = mirdata.initialize('fma_keys', data_home=data_home)

    if subset:
        dataset.download(partial_download=['tracks-000-019'])
    else:
        dataset.download()
        metadata_path = os.path.join(data_home, 'fma_keys_metadata.csv')

        # delete the original metadata file with the bad annotations
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

    download_metadata(data_home=data_home) # download the correct annotations separately
    return dataset

ROOTS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

def normalize_key(key_str):
    """
    Convert a key string to a canonical form.
    E.g., "Gb major" -> "F# major"
    Enharmonically equivalent keys map to the same representation.
    """
    try:
        raw_tonic, raw_mode = key_str.split()
        k = key.Key(raw_tonic, raw_mode.lower())
        # Get the pitch and convert to sharp representation (canonical form)
        tonic = k.tonic
        pc = tonic.pitchClass  # 0-11 representing pitch class

        canonical_tonic = ROOTS[pc]
        mode = k.mode.lower()
        
        return f"{canonical_tonic} {mode}"
    except Exception as e:
        print(e)


MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def krumhansl_schmuckler_predict(track):
    # load audio
    y, sr = track.audio

    # extract just the harmonic component
    y_harm = librosa.effects.harmonic(y, margin=8)

    # compute chroma features
    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_filtered = librosa.decompose.nn_filter(
        chroma_harm,
        aggregate=np.median,
        metric='cosine')
    chroma = np.minimum(
        chroma_harm,
        chroma_filtered
    )

    # get average chroma
    avg_chroma = np.mean(chroma, axis=1)

    best_correlation = -1
    best_key = None

    for i in range(12):
        major_correlation = np.corrcoef(np.roll(avg_chroma, -i), MAJOR_PROFILE)[0, 1]
        minor_correlation = np.corrcoef(np.roll(avg_chroma, -i), MINOR_PROFILE)[0, 1]
        
        if major_correlation > best_correlation:
            best_correlation = major_correlation
            best_key = f"{ROOTS[i]} major"
        if minor_correlation > best_correlation:
            best_correlation = minor_correlation
            best_key = f"{ROOTS[i]} minor"

    return best_key

key_recognizer = madmom.features.key.CNNKeyRecognitionProcessor()

def madmom_key_predict(track):
    predictions = key_recognizer(track.audio_path)
    predicted_key = madmom.features.key.key_prediction_to_label(predictions)
    return predicted_key

chroma_transformer_demo = ChromaTransformer(d_model=64, nhead=4, num_layers=2)
demo_weights_path = 'chroma_transformer/weights/fold_10.pt'
chroma_transformer_demo.load_state_dict(torch.load(demo_weights_path, map_location='cpu'))
chroma_transformer_demo.eval()

def chroma_transformer_predict(track):
    chroma = extract_features.extract_chroma(track.audio_path)
    features = torch.tensor(chroma, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = chroma_transformer_demo(features)
    predicted_label = prediction.argmax(dim=1).item()
    predicted_key = KeyLabelConverter.label_to_key(predicted_label)
    return predicted_key

def get_avg_weighted_score(y_true, y_pred):
    cumulative_score = 0.0

    for true, pred in zip(y_true, y_pred):
        weighted_score = mir_eval.key.weighted_score(true, pred)
        cumulative_score += weighted_score

    avg_score = cumulative_score / len(y_true)
    return avg_score

def get_metrics(predicted_keys, true_keys):
    return {
        'Accuracy': accuracy_score(true_keys, predicted_keys),
        'Precision (Macro)': precision_score(true_keys, predicted_keys, average='macro'),
        'Recall (Macro)': recall_score(true_keys, predicted_keys, average='macro'),
        'F1 Score (Macro)': f1_score(true_keys, predicted_keys, average='macro'),
        'Average MIREX Score': get_avg_weighted_score(true_keys, predicted_keys),
    }

def print_metrics(metrics, name):
    print(f"=== {name} ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

def _build_key_distance_matrix() -> dict[tuple[str, str], int]:
    all_keys = [f"{r} {m}" for r in ROOTS for m in ('major', 'minor')]

    def _pitch_classes(k: key.Key) -> frozenset[int]:
        s = scale.MinorScale(k.tonic) if k.mode == 'minor' else scale.MajorScale(k.tonic)
        return frozenset(p.pitchClass for p in s.getPitches()[:-1])

    pcs = {k: _pitch_classes(key.Key(*k.split())) for k in all_keys}
    return {(k1, k2): 7 - len(pcs[k1] & pcs[k2]) for k1 in all_keys for k2 in all_keys}

_KEY_DISTANCE = _build_key_distance_matrix()

def get_key_distance(key1: str, key2: str) -> int:
    """Number of notes that differ between two keys. Uses natural minor for minor keys."""
    return _KEY_DISTANCE[(normalize_key(key1), normalize_key(key2))]

def is_relative_key(key1: str, key2: str) -> bool:
    """Check if two keys are relative (same key signature)."""
    return get_key_distance(key1, key2) == 0 and normalize_key(key1) != normalize_key(key2)

def is_parallel_key(key1: str, key2: str) -> bool:
    """Check if two keys are parallel (same tonic)."""
    k1_norm = normalize_key(key1)
    k2_norm = normalize_key(key2)
    return k1_norm.split()[0] == k2_norm.split()[0] and k1_norm.split()[1] != k2_norm.split()[1]

def get_relative_key_error_proportion(predicted_keys, true_keys):
    relative_errors = [is_relative_key(pred, true) for pred, true in zip(predicted_keys, true_keys)]
    return sum(relative_errors) / len(relative_errors)

def get_parallel_key_error_proportion(predicted_keys, true_keys):
    parallel_errors = [is_parallel_key(pred, true) for pred, true in zip(predicted_keys, true_keys)]
    return sum(parallel_errors) / len(parallel_errors)