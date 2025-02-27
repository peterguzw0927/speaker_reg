from util import *
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import math
import pickle
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot likelihoods and histograms")
    parser.add_argument('--plot', choices=['histogram', 'time'], default='histogram', 
                        help="Select the type of plot to generate: 'histogram' or 'time'.")
    parser.add_argument('--save', action='store_true', help="Save the plot as a file instead of showing.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    database_path = "/Users/guzhaowen/Downloads/Yobe/database/"
    enrollment_file_path = os.path.join(database_path, "Male_ATTD4_1.flac")
    same_person_path = os.path.join(database_path,"Male_ATTD4_2.flac")
    
    # Preprocess enrollment audio
    X_enroll = preprocess_audio(enrollment_file_path)
    
    # Train GMM model
    gmm = GaussianMixture(n_components=20, covariance_type='full', random_state=42)
    gmm.fit(X_enroll)
    exclude_files = []
    gmm2 = train_gmm_for_all_files(database_path, exclude_files)

    # Compute likelihoods for enrollment file
    likelihoods_same = compute_frame_log_likelihood_(gmm, same_person_path)
    mean_enroll = np.mean(likelihoods_same)
    std_enroll = np.std(likelihoods_same)

    print(f"Enrollment File: {enrollment_file_path}")
    print(f"  Mean Log-Likelihood: {mean_enroll:.4f}")
    print(f"  Std Dev Log-Likelihood: {std_enroll:.4f}")
    print("-" * 50)

    if args.plot == 'time':
        flac_files = [f for f in os.listdir(database_path) if f.endswith("2.flac")]
        num_files = len(flac_files)

        # Determine grid size (rows x cols)
        cols = math.ceil(math.sqrt(num_files))  # Try to make it square-like
        rows = math.ceil(num_files / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))  # Adjust figure size dynamically
        axes = np.array(axes).reshape(rows, cols)  # Ensure axes is a 2D array
        likelihood_dict = {}
        
        for i, flac_file in enumerate(flac_files):
            file_path = os.path.join(database_path, flac_file)
            likelihoods_diff = compute_frame_log_likelihood_(gmm, file_path)
            likelihoods_all = compute_frame_log_likelihood_(gmm2,file_path)            
            max_length = max(len(likelihoods_same), len(likelihoods_diff))
            likelihoods_bool = [likelihoods_all - likelihoods_diff <= 7]
            likelihood_dict[flac_file] = likelihoods_bool

            likelihoods_same_cat = np.pad(likelihoods_same, (0, max_length - len(likelihoods_same)), constant_values=np.nan)
            likelihoods_diff = np.pad(likelihoods_diff, (0, max_length - len(likelihoods_diff)), constant_values=np.nan)
            likelihoods_all = np.pad(likelihoods_all, (0, max_length - len(likelihoods_all)), constant_values=np.nan)
            
            # Get subplot position
            row, col = divmod(i, cols)
            ax = axes[row, col]  # Access subplot
            time = np.arange(len(likelihoods_diff))
            ax.plot(time,likelihoods_diff,marker='o',label=f"Test ({flac_file[:-5]})")
            # ax.plot(time,likelihoods_same_cat,marker='o',label=f"Enrollment person different audio")
            ax.plot(time,likelihoods_all,marker='o',label=f"Background",color='k')
            ax.set_xlabel('Time')
            ax.set_ylabel('Log-Likelihood Difference')
            ax.set_title(f'{flac_file}\nLikelihood Difference over Time')
            ax.set_ylim(-120,-70)
            ax.set_xlabel("Time")
            ax.set_ylabel("Log Likelihood")
            ax.legend(loc='lower center',fontsize='x-small')

        # Convert dictionary to a DataFrame
        # df = pd.DataFrame.from_dict(likelihood_dict, orient='index')

        # # Save to CSV
        # csv_path = "Decision.csv"
        # df.to_csv(csv_path, index_label="File Name")

        # print(f"Saved dictionary to {csv_path}")


        # Remove empty subplots if any
        for i in range(num_files, rows * cols):
            fig.delaxes(axes.flatten()[i])

        if args.save:
            plt.tight_layout()
            plt.show()
            plt.savefig("likelihood_time.png")
        else:
            plt.tight_layout()
            plt.show()
            
    else:
        
        y, sr = librosa.load('/Users/guzhaowen/Downloads/Yobe/database/Female_ATTD6_1.flac')
        y_filt = librosa.effects.preemphasis(y)
        sf.write('/Users/guzhaowen/Downloads/Yobe/database/Female_ATTD6_1_preemp.flac',y_filt,sr)
        
        likelihoods_diff_2=compute_frame_log_likelihood_(gmm,database_path+'Female_ATTD6_2.flac')
        likelihoods_diff_1=compute_frame_log_likelihood_(gmm,database_path+'Female_ATTD6_1.flac')
        max_length = max(len(likelihoods_same), len(likelihoods_diff_1))
            
        likelihoods_same_cat = np.pad(likelihoods_same, (0, max_length - len(likelihoods_same)), constant_values=np.nan)
        likelihoods_diff_1 = np.pad(likelihoods_diff_1, (0, max_length - len(likelihoods_diff_1)), constant_values=np.nan)
        time = np.arange(len(likelihoods_diff_1))
        plt.plot(time,likelihoods_diff_1,color='k',marker='o')
        plt.plot(time,likelihoods_diff_2,color='b',marker='o')
        plt.plot(time,likelihoods_same_cat,color='r',marker='o')
        plt.title('Female_ATTD6\nLikelihood difference over time')
        # plt.ylim(-105,-86)
        plt.show()


if __name__ == '__main__':
    main()