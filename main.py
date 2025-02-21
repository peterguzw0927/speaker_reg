from util import *
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    # Define the directory containing FLAC files
    # audio_directory = "/Users/guzhaowen/Downloads/Yobe/different_person"

    # Load all FLAC files in the directory
    # flac_files = [os.path.join(audio_directory, f) for f in os.listdir(audio_directory) if f.endswith('.flac')]

    different_person_path = "/Users/guzhaowen/Downloads/Yobe/different_person/C18A-ORD_2025-01-14-20-16-57--266720_processed.flac"
    # different_person_path = "/Users/guzhaowen/Downloads/Yobe/different_person/C18A-ORD_2025-01-16-19-22-04--403354_processed.flac"
    same_person_path = "/Users/guzhaowen/Downloads/Yobe/ATTD_1_male_1.flac"  # Replace with your FLAC file path
    enrollment_file_path = "/Users/guzhaowen/Downloads/Yobe/ATTD_1_male_2.flac"
    X_enroll = preprocess_audio(enrollment_file_path)#take the slience out
    scaler = StandardScaler()
    X_enroll_scaled = scaler.fit_transform(X_enroll)

    gmm = GaussianMixture(n_components=20, covariance_type='full', random_state=42)
    gmm.fit(X_enroll_scaled)
    labels = gmm.predict(X_enroll_scaled)#take 3 seconds of audio, hop_length=0.5 and calculate loglikelihood
    
    likelihoods_enroll = compute_frame_log_likelihood_(gmm,enrollment_file_path,scaler)
    likelihoods_same = compute_frame_log_likelihood_(gmm,same_person_path,scaler)
    likelihoods_diff = compute_frame_log_likelihood_(gmm,different_person_path,scaler)
    
    threshold_factor = 6
    
    decision_same = is_enrolled_speaker(likelihoods_enroll, likelihoods_same,threshold_factor=threshold_factor)
    decision_diff = is_enrolled_speaker(likelihoods_enroll, likelihoods_diff,threshold_factor=threshold_factor)

    print("Decision for same person test file:", decision_same)
    print("Decision for different person test file:", decision_diff)
    plt.figure(figsize=(8, 6))
    plt.hist(likelihoods_enroll, bins=20, alpha=0.5, label="Enrollment")
    plt.hist(likelihoods_same, bins=20, alpha=0.5, label="Same Person")
    plt.hist(likelihoods_diff, bins=20, alpha=0.5, label="Different Person", color='r')
    plt.axvline(np.mean(likelihoods_enroll), color='b', linestyle='dashed', label="Mean Enroll")
    plt.axvline(np.mean(likelihoods_enroll) - threshold_factor*np.std(likelihoods_enroll), color='k', linestyle='dashed', label="Threshold")
    plt.xlabel("Log-Likelihood")
    plt.ylabel("Frequency")
    plt.title("Likelihood Distributions")
    plt.legend()
    plt.show()
