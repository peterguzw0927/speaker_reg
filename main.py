from util import *
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import math

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot likelihoods and histograms")
    parser.add_argument('--plot', choices=['histogram', 'time'], default='histogram', 
                        help="Select the type of plot to generate: 'histogram' or 'time'.")
    parser.add_argument('--save', action='store_true', help="Save the plot as a file instead of showing.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    database_path = "/Users/guzhaowen/Downloads/Yobe/database/"
    enrollment_file_path = os.path.join(database_path, "Female_ATTD1_1.flac")
    same_person_path = os.path.join(database_path,"Female_ATTD1_2.flac")
    
    # Preprocess enrollment audio
    X_enroll = preprocess_audio(enrollment_file_path)
    scaler = StandardScaler()
    X_enroll_scaled = scaler.fit_transform(X_enroll)
    
    # Train GMM model
    gmm = GaussianMixture(n_components=200, covariance_type='full', random_state=42)
    gmm.fit(X_enroll_scaled)

    # Compute likelihoods for enrollment file
    likelihoods_same = compute_frame_log_likelihood_(gmm, same_person_path, scaler)
    mean_enroll = np.mean(likelihoods_same)
    std_enroll = np.std(likelihoods_same)

    print(f"Enrollment File: {enrollment_file_path}")
    print(f"  Mean Log-Likelihood: {mean_enroll:.4f}")
    print(f"  Std Dev Log-Likelihood: {std_enroll:.4f}")
    print("-" * 50)

    # Find all files ending with "2.flac"
    flac_files = [f for f in os.listdir(database_path) if f.endswith("2.flac")]
    num_files = len(flac_files)

    # Determine grid size (rows x cols)
    cols = math.ceil(math.sqrt(num_files))  # Try to make it square-like
    rows = math.ceil(num_files / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))  # Adjust figure size dynamically
    axes = np.array(axes).reshape(rows, cols)  # Ensure axes is a 2D array

    # if args.plot == 'histogram':
        # for i, flac_file in enumerate(flac_files):
        #     file_path = os.path.join(database_path, flac_file)
        #     likelihoods_diff = compute_frame_log_likelihood_(gmm, file_path, scaler)

        #     mean_diff = np.mean(likelihoods_diff)
        #     std_diff = np.std(likelihoods_diff)

        #     print(f"Test File: {flac_file}")
        #     print(f"  Mean Log-Likelihood: {mean_diff:.4f}")
        #     print(f"  Std Dev Log-Likelihood: {std_diff:.4f}")
        #     print("-" * 50)

        #     # Get subplot position
        #     row, col = divmod(i, cols)
        #     ax = axes[row, col]  # Access subplot

        #     # Plot likelihood distributions
        #     ax.hist(likelihoods_same, bins=25, alpha=0.5, label="Enrollment person different audio")
        #     ax.hist(likelihoods_diff, bins=25, alpha=0.5, label=f"Test ({flac_file[:-5]})", color='r')
        #     ax.axvline(mean_enroll, color='b', linestyle='dashed')
        #     ax.axvline(mean_diff, color='r', linestyle='dashed')
        #     ax.set_xlabel("Log-Likelihood")
        #     ax.set_ylabel("Frequency")
        #     ax.set_xlim(-50,-30)
        #     ax.set_title(f"{flac_file}\nMean: {mean_diff:.2f}, Std: {std_diff:.2f}")
        #     ax.legend(fontsize='small',loc='upper left')

        # # Remove empty subplots if any
        # for i in range(num_files, rows * cols):
        #     fig.delaxes(axes.flatten()[i])

        # if args.save:
        #     plt.savefig("likelihood_histograms.png")
        # else:
        #     plt.tight_layout()
        #     plt.show()


    if args.plot == 'time':
        for i, flac_file in enumerate(flac_files):
            file_path = os.path.join(database_path, flac_file)
            likelihoods_diff = compute_frame_log_likelihood_(gmm, file_path, scaler)
            max_length = max(len(likelihoods_same), len(likelihoods_diff))
            
            likelihoods_same_cat = np.pad(likelihoods_same, (0, max_length - len(likelihoods_same)), constant_values=np.nan)
            likelihoods_diff = np.pad(likelihoods_diff, (0, max_length - len(likelihoods_diff)), constant_values=np.nan)

            # Get subplot position
            row, col = divmod(i, cols)
            ax = axes[row, col]  # Access subplot
            time = np.arange(len(likelihoods_diff))
            ax.plot(time,likelihoods_diff,marker='o',label=f"Test ({flac_file[:-5]})")
            ax.plot(time,likelihoods_same_cat,marker='o',label=f"Enrollment person different audio")
            ax.set_xlabel('Time')
            ax.set_ylabel('Log-Likelihood Difference')
            ax.set_title(f'{flac_file}\nLikelihood Difference over Time')
            ax.set_ylim(-160,-115) 
            ax.set_xlabel("Time")
            ax.set_ylabel("Log Likelihood")
            ax.legend(fontsize='small',loc='lower right')

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


if __name__ == '__main__':
    main()