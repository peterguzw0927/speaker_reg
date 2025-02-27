import numpy as np
import librosa
import os
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def compute_log_likelihood(gmm, file_path,scaler):
    X_test = preprocess_audio(file_path)
    X_test = scaler.transform(X_test)
    return gmm.score(X_test)  # Returns average log-likelihood

def preprocess_audio(enrollment_file_path):
    """Processes the audio, extracts the top 70% highest energy frames, and computes MFCCs.

    Args:
        enrollment_file_path (str): Path of the enrollment file.

    Returns:
        np.ndarray: MFCC array.
    """
    # Load audio and take first 15 seconds
    audio, sr = librosa.load(enrollment_file_path, sr=None)
    max_samples = sr * 15
    audio = audio[:max_samples]
    audio = librosa.effects.preemphasis(audio,coef=0.99)
    # Define frame parameters
    frame_length = 320
    hop_length = 160

    # Compute sum of squares energy for each frame
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    std_per_frame = np.std(frames, axis=0, keepdims=True) + 1e-8
    frames = (frames - np.mean(frames, axis=0, keepdims=True))/std_per_frame

    frame_energy = np.sum(frames**2, axis=0)  # Sum of squares energy per frame
    frame_energy_normalized = frame_energy

    # Compute energy threshold (30th percentile)
    energy_threshold = np.percentile(frame_energy_normalized, 30)

    # Get indices of high-energy frames
    high_energy_frame_indices = frame_energy_normalized > energy_threshold

    # Ensure `frame_samples` matches `frame_energy` length
    frame_samples = np.arange(0, len(frame_energy) * hop_length, hop_length)[:len(frame_energy)]

    # Extract high-energy audio segments
    high_energy_audio = []
    for i in range(len(frame_energy)):  # Iterate over valid frame indices
        if high_energy_frame_indices[i]:
            start = frame_samples[i]
            end = start + frame_length  # Ensure full frame extraction
            high_energy_audio.extend(audio[start:end])

    # Convert to numpy array
    high_energy_audio = np.array(high_energy_audio)

    # Compute MFCC features
    mfcc = librosa.feature.mfcc(y=high_energy_audio, sr=sr, n_fft=512, hop_length=hop_length, 
                                win_length=frame_length, n_mels=128, n_mfcc=20)

    return mfcc.T 

def compute_frame_log_likelihood_(gmm, file_path, sr=16000):
    """_summary_

    Args:
        gmm (_type_): _description_
        file_path (_type_): _description_
        scaler (_type_): _description_
        sr (int, optional): _description_. Defaults to 16000.

    Returns:
        _type_: _description_
    """
    audio, _ = librosa.load(file_path, sr=sr)
    max_samples = sr * 15  # Take only the first 15 seconds
    audio = audio[:max_samples]
    audio = librosa.effects.preemphasis(audio,coef=0.95)

    frame_length = sr * 3  # 3 seconds per frame
    hop_length = sr // 2  # 0.5-second hop
    
    frame_length_mfcc = 320
    hop_length_mfcc = 160

    log_likelihoods = []

    for start in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[start: start + frame_length]
        frames = librosa.util.frame(frame, frame_length=frame_length_mfcc, hop_length=hop_length_mfcc)
        std_per_frame = np.std(frames, axis=0, keepdims=True) + 1e-8
        frames = (frames - np.mean(frames, axis=0, keepdims=True))/std_per_frame

        frame_energy = np.sum(frames**2, axis=0) 

        # Normalize 
        frame_energy_normalized = frame_energy
        energy_threshold = np.percentile(frame_energy_normalized, 30)

        # Get high-energy frame indices
        high_energy_frame_indices = frame_energy_normalized > energy_threshold
        
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_fft=512, win_length=frame_length_mfcc, hop_length=hop_length_mfcc, 
                                n_mfcc=20, n_mels=128).T

        # Ensure `mfcc` matches `frame_energy`
        num_frames = min(mfcc.shape[0], len(frame_energy))
        mfcc = mfcc[:num_frames]  
        high_energy_frame_indices = high_energy_frame_indices[:num_frames]  

        # Extract high-energy MFCC frames
        high_energy_mfcc = mfcc[high_energy_frame_indices]
        log_likelihoods.append(gmm.score(high_energy_mfcc))

    return np.array(log_likelihoods)  # Return an array of log-likelihoods

def train_gmm_for_all_files(database_path, exclude_files, n_components=20):
    """
    Train a GMM for each file in the database excluding the specified files.
    """

    flac_files = [f for f in os.listdir(database_path) if f.endswith("1.flac") and f not in exclude_files]
    ubm_features = []
    for file in flac_files:
        file_path = os.path.join(database_path, file)
        print(f"Training GMM for {file_path}")
        
        # Preprocess audio to extract features (assuming preprocess_audio function exists)
        X = preprocess_audio(file_path)
        ubm_features.append(X)
    ubm_features = np.vstack(ubm_features)
    
    gmm_ubm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm_ubm.fit(ubm_features)
        
    return gmm_ubm


# def compute_frame_log_likelihood_15s(gmm, file_path,sr=16000):
#     # Load audio
#     audio, _ = librosa.load(file_path, sr=sr)
#     max_samples = sr * 15  # Take only the first 15 seconds
#     audio = audio[:max_samples]
#     audio = librosa.effects.preemphasis(audio,coef=0.99)

#     frame_length = 320
#     hop_length = 160

#     # Compute frame energy
#     frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
#     frame_energy = np.sum(frames**2, axis=0) 

#     # Normalize 
#     frame_energy_normalized = frame_energy
#     energy_scaler = StandardScaler()
#     frame_energy_normalized = energy_scaler.fit_transform(frame_energy.reshape(-1, 1)).flatten()
#     energy_threshold = np.percentile(frame_energy_normalized, 30)

#     # Get high-energy frame indices
#     high_energy_frame_indices = frame_energy_normalized > energy_threshold

#     # Compute MFCC features
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_fft=512, win_length=frame_length, hop_length=hop_length, 
#                                 n_mfcc=20, n_mels=128).T

#     # Ensure `mfcc` matches `frame_energy`
#     num_frames = min(mfcc.shape[0], len(frame_energy))
#     mfcc = mfcc[:num_frames]  
#     high_energy_frame_indices = high_energy_frame_indices[:num_frames]  

#     # Extract high-energy MFCC frames
#     high_energy_mfcc = mfcc[high_energy_frame_indices]

#     # Compute log likelihood using GMM
#     log_likelihoods = gmm.score(high_energy_mfcc)

#     return np.array(log_likelihoods)
