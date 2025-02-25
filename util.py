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

def preprocess_audio(enrollment_file_path):
    """process the audio, only take 15 seconds. get 70% of the largest energy, and then do mfcc

    Args:
        enrollment_file_path (_type_): path of enrollment file

    Returns:
        _type_: mfcc arrary
    """
    audio, sr = librosa.load(enrollment_file_path, sr=None)
    max_samples = sr * 15
    audio = audio[sr:max_samples+sr]
    frame_length = 320
    hop_length = 160
    frame_energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy_threshold = np.percentile(frame_energy, 30)  # Keep top 80% energy
    high_energy_frame_indices = frame_energy > energy_threshold
    frame_samples = np.arange(0, len(audio), hop_length)
    
    high_energy_audio = []
    
    for i in range(len(frame_samples)):
        if high_energy_frame_indices[i]:
            start = frame_samples[i]
            end = start + frame_length  # Frame length
            high_energy_audio.extend(audio[start:end])
    
    high_energy_audio = np.array(high_energy_audio)
    mfcc = librosa.feature.mfcc(y=high_energy_audio, sr=sr, n_fft=512, hop_length=hop_length, win_length=frame_length, n_mels=128,n_mfcc=20)
    
    return mfcc.T  

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
        frame_energy = librosa.feature.rms(y=frame, frame_length=320, hop_length=160)[0]
        energy_threshold = np.percentile(frame_energy, 30)  # Keep top 70% energy
        high_energy_frame_indices = frame_energy > energy_threshold
        frame_samples = np.arange(0, len(audio), hop_length)
        
        high_energy_audio = []
        
        for i in range(len(frame_samples)):
            if high_energy_frame_indices[i]:
                start = frame_samples[i]
                end = start + frame_length  # Frame length
                high_energy_audio.extend(audio[start:end])
        
        high_energy_audio = np.array(high_energy_audio)
        mfcc = librosa.feature.mfcc(y=high_energy_audio, sr=sr,n_fft=512,win_length=320,hop_length=160 ,n_mels=128, n_mfcc=20).T
        mfcc_scaled = scaler.transform(mfcc)
        log_likelihoods.append(gmm.score(mfcc_scaled))

    return np.array(log_likelihoods)  # Return an array of log-likelihoods

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