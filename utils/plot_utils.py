import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

def generate_plots(log_dir_base, window_size=10, confidence_level=0.9):
    all_timesteps = None
    all_mean_rewards = defaultdict(dict)
    all_mean_ep_lengths = defaultdict(dict)

    assignment_methods = os.listdir(log_dir_base)

    for method in assignment_methods:
        method_log_dir_base = os.path.join(log_dir_base, method)
        iteration_dirs = os.listdir(method_log_dir_base)

        for iteration_dir in iteration_dirs:
            npz_path = os.path.join(method_log_dir_base, iteration_dir, "evaluations.npz")
            eval_data = np.load(npz_path)

            # Extract the relevant data
            timesteps = eval_data['timesteps']
            rewards = eval_data['results'].mean(axis=1)  # Mean over n_eval_episodes
            ep_lengths = eval_data['ep_lengths'].mean(axis=1)  # Mean over n_eval_episodes

            if all_timesteps is None:
                all_timesteps = timesteps

            # Store the data in the dictionaries
            all_mean_rewards[method][iteration_dir] = rewards
            all_mean_ep_lengths[method][iteration_dir] = ep_lengths

    # Plot results using Matplotlib
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for method in assignment_methods:
        df_mean_rewards = pd.DataFrame(all_mean_rewards[method]).transpose()
        mean_rewards = df_mean_rewards.mean(axis=0).rolling(window=window_size).mean()
        sem_rewards = df_mean_rewards.sem(axis=0).rolling(window=window_size).mean()
        t_value = stats.t.ppf((1 + confidence_level) / 2, df_mean_rewards.count(axis=0) - 1)
        ci_rewards = sem_rewards * t_value
        plt.plot(all_timesteps, mean_rewards, label=f"{method}")
        plt.fill_between(all_timesteps, mean_rewards - ci_rewards, mean_rewards + ci_rewards, alpha=0.2)
    plt.xlabel("Total Timesteps")
    plt.ylabel("Mean Reward")
    plt.title(f"Mean Reward over Total Timesteps (Smoothed with {int(confidence_level*100)}% CI)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    for method in assignment_methods:
        df_mean_ep_lengths = pd.DataFrame(all_mean_ep_lengths[method]).transpose()
        mean_ep_lengths = df_mean_ep_lengths.mean(axis=0).rolling(window=window_size).mean()
        sem_ep_lengths = df_mean_ep_lengths.sem(axis=0).rolling(window=window_size).mean()
        t_value = stats.t.ppf((1 + confidence_level) / 2, df_mean_ep_lengths.count(axis=0) - 1)
        ci_ep_lengths = sem_ep_lengths * t_value
        plt.plot(all_timesteps, mean_ep_lengths, label=f"{method}")
        plt.fill_between(all_timesteps, mean_ep_lengths - ci_ep_lengths, mean_ep_lengths + ci_ep_lengths, alpha=0.2)
    plt.xlabel("Total Timesteps")
    plt.ylabel("Mean Episode Length")
    plt.title(f"Mean Episode Length over Total Timesteps (Smoothed with {int(confidence_level*100)}% CI)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def generate_plots_legacy(log_dir_base, window_size=10):
    all_timesteps = None
    all_mean_rewards = defaultdict(dict)
    all_mean_ep_lengths = defaultdict(dict)

    assignment_methods = os.listdir(log_dir_base)

    for method in assignment_methods:
        method_log_dir_base = os.path.join(log_dir_base, method)
        iteration_dirs = os.listdir(method_log_dir_base)

        for iteration_dir in iteration_dirs:
            npz_path = os.path.join(method_log_dir_base, iteration_dir, "evaluations.npz")
            eval_data = np.load(npz_path)

            # Extract the relevant data
            timesteps = eval_data['timesteps']
            rewards = eval_data['results'].mean(axis=1)  # Mean over n_eval_episodes
            ep_lengths = eval_data['ep_lengths'].mean(axis=1)  # Mean over n_eval_episodes

            if all_timesteps is None:
                all_timesteps = timesteps

            # Store the data in the dictionaries
            all_mean_rewards[method][iteration_dir] = rewards
            all_mean_ep_lengths[method][iteration_dir] = ep_lengths

    # Plot results using Matplotlib
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for method in assignment_methods:
        df_mean_rewards = pd.DataFrame(all_mean_rewards[method]).transpose()
        mean_rewards = df_mean_rewards.mean(axis=0).rolling(window=window_size).mean()
        std_rewards = df_mean_rewards.std(axis=0).rolling(window=window_size).mean()

        # print("PLS", df_mean_rewards.mean(axis=0))

        plt.plot(all_timesteps, mean_rewards, label=f"{method}")
        plt.fill_between(all_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    plt.xlabel("Total Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward over Total Timesteps (Smoothed)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    for method in assignment_methods:
        df_mean_ep_lengths = pd.DataFrame(all_mean_ep_lengths[method]).transpose()
        mean_ep_lengths = df_mean_ep_lengths.mean(axis=0).rolling(window=window_size).mean()
        std_ep_lengths = df_mean_ep_lengths.std(axis=0).rolling(window=window_size).mean()
        plt.plot(all_timesteps, mean_ep_lengths, label=f"{method}")
        plt.fill_between(all_timesteps, mean_ep_lengths - std_ep_lengths, mean_ep_lengths + std_ep_lengths, alpha=0.2)
    plt.xlabel("Total Timesteps")
    plt.ylabel("Mean Episode Length")
    plt.title("Mean Episode Length over Total Timesteps (Smoothed)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()