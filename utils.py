import mirdata
import os
import urllib.request
import librosa
import numpy as np
import madmom

def download_metadata(data_home):
    """
    Download FMAKv2 metadata from Zenodo.

    This is an updated version of the annotations file with corrections. 
    """
    os.makedirs(data_home, exist_ok=True)
    file_path = os.path.join(data_home, 'fma_keys_metadata.csv')
    
    if os.path.exists(file_path):
        print(f"File already exists at {file_path}")
        return file_path
    
    url = 'https://zenodo.org/records/12759100/files/fmakv2.csv'
    print(f"Downloading FMAKv2 from {url}...")
    urllib.request.urlretrieve(url, file_path)
    print(f"Downloaded to {file_path}")
    return file_path


def load_data(data_home='subset/', subset=True):
    dataset = mirdata.initialize('fma_keys', data_home=data_home)

    if subset:
        dataset.download(partial_download=['tracks-000-019'])
    else:
        dataset.download()

    download_metadata(data_home=data_home) # download metadata separately
    return dataset

ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
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
