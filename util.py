import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


def load_audio(file_path, sr=16000):
    y, sr = librosa.load(file_path, sr=sr)  
    return y, sr

def compute_log_likelihood(gmm, file_path,scaler):
    X_test = preprocess_audio(file_path)
    X_test = scaler.transform(X_test)
    return gmm.score(X_test)  # Returns average log-likelihood

def preprocess_audio(enrollment_file_path,energy_percentage=0.7):
    """process the audio, only take 15 seconds. do VAD, and then do mfcc

    Args:
        enrollment_file_path (_type_): path of enrollment file

    Returns:
        _type_: mfcc arrary
    """
    audio, sr = librosa.load(enrollment_file_path, sr=None)
    hop_length = 320
    frame_length=1024
    frame_energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    sorted_indices = np.argsort(frame_energy)[::-1]  
    total_energy = np.sum(frame_energy)
    cumulative_energy = np.cumsum(frame_energy[sorted_indices])
    threshold_index = np.searchsorted(cumulative_energy, energy_percentage * total_energy)

    # Keep only the top frames that account for 70% of total energy
    selected_indices = sorted_indices[:threshold_index + 1]
    frame_samples = np.arange(0, len(audio), hop_length)
    high_energy_audio = []

    for i in selected_indices:
        start = frame_samples[i]
        end = min(start + frame_length, len(audio))  # Ensure within bounds
        high_energy_audio.extend(audio[start:end])

    # Convert to NumPy array
    high_energy_audio = np.array(high_energy_audio)
    
    mfcc = librosa.feature.mfcc(y=high_energy_audio, sr=sr, n_fft=1024, hop_length=hop_length, win_length=640, n_mfcc=30)
    
    return mfcc.T  


def test_audio_gmm(test_file_path,gmm,scaler,sr=16000,window_size=3,hop_size=1.5,energy_percentage=0.7):
    """
    Processes a test audio file in sliding windows, extracts MFCCs, 
    and evaluates speaker match using a trained GMM.

    - Uses high-energy frames (70% of total energy).
    - Gets probability from GMM.
    - Counts votes where max probability > 0.8.
    - Returns the ratio of votes per window.

    Args:
        test_file_path (str): Path to test audio file.
        gmm (GaussianMixture): Trained Gaussian Mixture Model.
        scaler (StandardScaler): StandardScaler for normalization.
        sr (int, optional): Sampling rate. Defaults to 16000.
        window_size (float, optional): Window size in seconds. Defaults to 3.
        hop_size (float, optional): Hop size in seconds. Defaults to 1.5.
        energy_percentage (float, optional): Percentage of energy frames to keep. Defaults to 0.7.

    Returns:
        list: List of vote ratios for each window.
    """
    audio, sr = librosa.load(test_file_path, sr=sr)
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    window_results = []
    frame_length=1024
    hop_length = 320
    for start in range(0, len(audio)-window_samples + 1, hop_samples):  # Ensure full coverage
        end = start + window_samples
        if end > len(audio):  # Handle last segment
            segment = audio[start:]
        else:
            segment = audio[start:end]
        # print(f"window start {start/16000}, window end {end/16000}")
        segment = audio[start:end]
        frame_energy = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
        sorted_indices = np.argsort(frame_energy)[::-1]  
        total_energy = np.sum(frame_energy)
        cumulative_energy = np.cumsum(frame_energy[sorted_indices])
        threshold_index = np.searchsorted(cumulative_energy, energy_percentage * total_energy)

        # Keep only the top frames that account for 70% of total energy
        selected_indices = sorted_indices[:threshold_index + 1]
        frame_samples = np.arange(0, len(audio), hop_length)
        high_energy_audio = []

        for i in selected_indices:
            start = frame_samples[i]
            end = min(start + frame_length, len(audio))  # Ensure within bounds
            high_energy_audio.extend(audio[start:end])

        # Convert to NumPy array
        high_energy_audio = np.array(high_energy_audio)
        mfcc = librosa.feature.mfcc(y=high_energy_audio, sr=sr, n_fft=1024, hop_length=hop_length, win_length=640, n_mfcc=30)
        mfcc_scaled = scaler.transform(mfcc.T)
        probabilities = gmm.predict_proba(mfcc_scaled)
        print(probabilities[0])
        # print(np.max(probabilities,1))
        max_probs = np.max(probabilities,1)

        votes = np.sum(max_probs > 0.9)
        total_frames = len(max_probs)

        vote_ratio = votes / total_frames if total_frames > 0 else 0
        window_results.append(vote_ratio)
        
    return  window_results
        

def compute_frame_log_likelihood_(gmm, file_path, scaler, sr=16000):
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
    audio = audio[sr:max_samples+sr]

    frame_length = sr * 3  # 3 seconds per frame
    hop_length = sr // 2  # 0.5-second hop

    log_likelihoods = []

    for start in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[start: start + frame_length]
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=20).T
        mfcc_scaled = scaler.transform(mfcc)
        log_likelihoods.append(gmm.score(mfcc_scaled))

    return np.array(log_likelihoods)  # Return an array of log-likelihoods

def runningMeanFast(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')

def runningMedianManual(x, N):
    return np.array([np.median(x[i:i+N]) for i in range(len(x) - N + 1)])

def is_enrolled_speaker(likelihoods_enroll, likelihoods_test, threshold_factor=6):
    mean_enroll = np.mean(likelihoods_enroll)
    std_enroll = np.std(likelihoods_enroll)
    
    # Decision threshold
    threshold = mean_enroll - threshold_factor * std_enroll
    
    # Compare test likelihood with threshold
    mean_test = np.mean(likelihoods_test)
    
    if mean_test >= threshold:
        return "Speaker matches enrollment."
    else:
        return "Speaker is different."