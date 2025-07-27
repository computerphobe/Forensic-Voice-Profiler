import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import os

def extract_vocal_features(file_path):
    """
    Extracts a set of vocal features from an audio file.
    This version includes robust error handling for jitter and shimmer calculation.
    """
    try:
        # Load audio using both libraries
        y, sr = librosa.load(file_path, sr=None)
        sound = parselmouth.Sound(file_path)

        # --- Pitch (F0) ---
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        if np.isnan(mean_pitch):
            mean_pitch = 0.0

        # --- Jitter and Shimmer ---
        jitter_local = 0.0
        shimmer_local = 0.0
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_local = call(point_process, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except parselmouth.PraatError:
            pass

        # --- MFCCs ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # --- Speech Rate ---
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        speech_rate = len(onset_frames) / duration if duration > 0 else 0

        # --- Create a dictionary for this file ---
        features = {
            'filename': os.path.basename(file_path),
            'mean_pitch_hz': mean_pitch,
            'jitter_local': jitter_local,
            'shimmer_local': shimmer_local,
            'speech_rate_onsets_per_sec': speech_rate,
        }
        for i, mfcc_val in enumerate(mfccs_mean):
            features[f'mfcc_{i+1}_mean'] = mfcc_val

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# --- Main Execution ---
# !! IMPORTANT !!
# !! Change this path to the top-level directory (e.g., 'dataset')
AUDIO_DIRECTORY = './dataset'

# List to hold all the feature dictionaries
all_features = []

print(f"Starting recursive feature extraction from root directory: {AUDIO_DIRECTORY}")

# Use os.walk to go through all subdirectories
if os.path.exists(AUDIO_DIRECTORY):
    for root, dirs, files in os.walk(AUDIO_DIRECTORY):
        for filename in files:
            if filename.endswith('.wav'):
                # Construct the full path to the audio file
                file_path = os.path.join(root, filename)
                print(f"Processing: {file_path}...")

                # Extract features and add to our list
                features = extract_vocal_features(file_path)
                if features:
                    all_features.append(features)
else:
    print(f"Error: Directory not found at '{AUDIO_DIRECTORY}'")

# Convert the list of dictionaries to a pandas DataFrame
if all_features:
    df_features = pd.DataFrame(all_features)
    output_csv_path = 'vocal_features.csv'
    df_features.to_csv(output_csv_path, index=False)
    print("\n-------------------------------------------")
    print(f"✅ Feature extraction complete!")
    print(f"Data for {len(all_features)} files saved to: {output_csv_path}")
    print("-------------------------------------------")
else:
    print("\nNo features were extracted. Please check the audio directory and files.")