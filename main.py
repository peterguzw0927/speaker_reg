from util import *
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    # different_person_path = "/Users/guzhaowen/Downloads/Yobe/different_person/C18A-ORD_2025-01-14-20-16-57--266720_processed.flac"
    different_person_path = "/Users/guzhaowen/Downloads/Yobe/different_person/ATTD_2_Female_1_15s.wav"
    same_person_path = "/Users/guzhaowen/Downloads/Yobe/ATTD_1_male_1_15s.wav" 
    enrollment_file_path = "/Users/guzhaowen/Downloads/Yobe/ATTD_1_male_2_15s.wav"
    
    
    X_enroll = preprocess_audio(enrollment_file_path)#take the slience out
    scaler = StandardScaler()
    X_enroll_scaled = scaler.fit_transform(X_enroll)

    gmm = GaussianMixture(n_components=20, covariance_type='full', random_state=42)
    gmm.fit(X_enroll_scaled)

    res=test_audio_gmm(different_person_path,gmm,scaler)
    print(res)