import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
import re

def generate_discount_plot(log_dir_base, window_size=10, confidence_level=0.9, gamma=0.99):
    all_timesteps = None
    all_discounted_rewards = defaultdict(dict)
    def extract_value_after_equals(name):
        """
        Extracts the value after '=' in the directory name. It handles cases where the name
        contains non-numeric characters by stripping them before parsing the float.
        If no '=' is found, returns a high value to sort it last.
        """
        if '=' in name:
            # Extract the part after '='
            value_str = name.split('=')[1]
            # Use regex to extract only the numeric portion (handling cases like "$beta = 0.5")
            value_str = re.sub(r'[^\d.]+', '', value_str)
            try:
                return float(value_str)
            except ValueError:
                return float('inf')  # In case the string cannot be converted, place it last
        return float('inf')

    assignment_methods = os.listdir(log_dir_base)

    assignment_methods.sort(key=extract_value_after_equals)
    # First pass to collect all timesteps lengths
    timesteps_lengths = []

    for method in assignment_methods:
        method_log_dir_base = os.path.join(log_dir_base, method)
        if not os.path.isdir(method_log_dir_base):
            print(f"Skipping {method_log_dir_base}, not a directory.")
            continue
        iteration_dirs = os.listdir(method_log_dir_base)

        for iteration_dir in iteration_dirs:
            npz_path = os.path.join(method_log_dir_base, iteration_dir, "evaluations.npz")
            if not os.path.isfile(npz_path):
                print(f"File not found: {npz_path}. Skipping.")
                continue
            eval_data = np.load(npz_path)

            timesteps = eval_data['timesteps']
            timesteps_lengths.append(len(timesteps))

    if not timesteps_lengths:
        print("No data found in the specified log directory.")
        return

    # Determine the minimum number of timesteps across all files
    min_length = min(timesteps_lengths)

    # Now, process the data again with the min_length
    for method in assignment_methods:
        method_log_dir_base = os.path.join(log_dir_base, method)
        if not os.path.isdir(method_log_dir_base):
            continue
        iteration_dirs = os.listdir(method_log_dir_base)

        for iteration_dir in iteration_dirs:
            npz_path = os.path.join(method_log_dir_base, iteration_dir, "evaluations.npz")
            if not os.path.isfile(npz_path):
                continue
            eval_data = np.load(npz_path)

            # Extract the relevant data and truncate to min_length
            timesteps = eval_data['timesteps'][:min_length]
            rewards = eval_data['results'].mean(axis=1)[:min_length]  # Mean over n_eval_episodes
            mean_ep_lengths = eval_data['ep_lengths'].mean(axis=1)[:min_length]

            if all_timesteps is None:
                all_timesteps = timesteps

            # Calculate discounted rewards
            discounted_rewards = rewards * gamma ** mean_ep_lengths

            # Store the data in the dictionary
            all_discounted_rewards[method][iteration_dir] = discounted_rewards

    # Plot results using Matplotlib
    plt.figure(figsize=(9, 4.5))
    # 8.5, 5 for main

    for method in assignment_methods:
        if method not in all_discounted_rewards:
            continue
        df_discounted_rewards = pd.DataFrame(all_discounted_rewards[method]).transpose()
        if df_discounted_rewards.empty:
            continue
        # Compute mean and confidence intervals
        mean_discounted_rewards = df_discounted_rewards.mean(axis=0).rolling(window=window_size).mean()
        sem_discounted_rewards = df_discounted_rewards.sem(axis=0).rolling(window=window_size).mean()
        degrees_of_freedom = df_discounted_rewards.count(axis=0) - 1
        degrees_of_freedom = degrees_of_freedom.clip(lower=1)
        t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        ci_discounted_rewards = sem_discounted_rewards * t_value

        all_timesteps_with_0 = np.insert(all_timesteps, 0, 0)
        mean_discounted_rewards_with_0 = np.insert(mean_discounted_rewards.fillna(0).values, 0, 0)  # Fill NaN values with 0 for plotting

        # Plotting the smoothed data with the (0, 0) point added
        plt.plot(all_timesteps_with_0, mean_discounted_rewards_with_0, label=f"{method}", linewidth=2)  # Bold line
        plt.fill_between(all_timesteps_with_0, 
                         mean_discounted_rewards_with_0 - np.insert(ci_discounted_rewards.fillna(0).values, 0, 0),
                         mean_discounted_rewards_with_0 + np.insert(ci_discounted_rewards.fillna(0).values, 0, 0), 
                         alpha=0.2)


        # plt.plot(all_timesteps, mean_discounted_rewards, label=f"{method}", linewidth=2)  # Bold line
        # plt.fill_between(all_timesteps, mean_discounted_rewards - ci_discounted_rewards, mean_discounted_rewards + ci_discounted_rewards, alpha=0.2)

    plt.xlabel("Total Timesteps (x1e6)", fontsize=15)
    plt.ylabel("Discounted Reward", fontsize=15)
    # plt.title(f"Mean Discounted Reward over Total Timesteps (Gamma={gamma}, Smoothed with {int(confidence_level*100)}% CI)", fontsize=15)
    plt.title(f"Repairs Task", fontsize=18)
    plt.legend(fontsize=14, loc="upper left")
    plt.grid()
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/graph.png')
    plt.show()


def generate_plots(log_dir_base, window_size=10, confidence_level=0.9):
    all_timesteps = None
    all_mean_rewards = defaultdict(dict)
    all_mean_ep_lengths = defaultdict(dict)

    assignment_methods = os.listdir(log_dir_base)

    # First pass to collect all timesteps lengths
    timesteps_lengths = []

    for method in assignment_methods:
        method_log_dir_base = os.path.join(log_dir_base, method)
        if not os.path.isdir(method_log_dir_base):
            print(f"Skipping {method_log_dir_base}, not a directory.")
            continue
        iteration_dirs = os.listdir(method_log_dir_base)

        for iteration_dir in iteration_dirs:
            npz_path = os.path.join(method_log_dir_base, iteration_dir, "evaluations.npz")
            if not os.path.isfile(npz_path):
                print(f"File not found: {npz_path}. Skipping.")
                continue
            eval_data = np.load(npz_path)

            # Extract the relevant data
            timesteps = eval_data['timesteps']
            timesteps_lengths.append(len(timesteps))

    if not timesteps_lengths:
        print("No data found in the specified log directory.")
        return

    # Determine the minimum number of timesteps across all files
    min_length = min(timesteps_lengths)

    # Now, process the data again with the min_length
    for method in assignment_methods:
        method_log_dir_base = os.path.join(log_dir_base, method)
        if not os.path.isdir(method_log_dir_base):
            continue
        iteration_dirs = os.listdir(method_log_dir_base)

        for iteration_dir in iteration_dirs:
            npz_path = os.path.join(method_log_dir_base, iteration_dir, "evaluations.npz")
            if not os.path.isfile(npz_path):
                continue
            eval_data = np.load(npz_path)

            # Extract the relevant data and truncate to min_length
            timesteps = eval_data['timesteps'][:min_length]
            rewards = eval_data['results'].mean(axis=1)[:min_length]  # Mean over n_eval_episodes
            ep_lengths = eval_data['ep_lengths'].mean(axis=1)[:min_length]  # Mean over n_eval_episodes

            if all_timesteps is None:
                all_timesteps = timesteps

            # Store the truncated data in the dictionaries
            all_mean_rewards[method][iteration_dir] = rewards
            all_mean_ep_lengths[method][iteration_dir] = ep_lengths

    # Plot results using Matplotlib
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for method in assignment_methods:
        if method not in all_mean_rewards:
            continue
        df_mean_rewards = pd.DataFrame(all_mean_rewards[method]).transpose()
        if df_mean_rewards.empty:
            continue
        mean_rewards = df_mean_rewards.mean(axis=0).rolling(window=window_size).mean()

        sem_rewards = df_mean_rewards.sem(axis=0).rolling(window=window_size).mean()
        degrees_of_freedom = df_mean_rewards.count(axis=0) - 1
        # Avoid issues where degrees_of_freedom <=0
        degrees_of_freedom = degrees_of_freedom.clip(lower=1)
        t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        ci_rewards = sem_rewards * t_value
        plt.plot(all_timesteps, mean_rewards, label=f"{method}", linewidth=2)  # Bold line
        plt.fill_between(all_timesteps, mean_rewards - ci_rewards, mean_rewards + ci_rewards, alpha=0.2)
    plt.xlabel("Total Timesteps (x1e6)")
    plt.ylabel("Mean Reward")
    plt.title(f"Mean Reward over Total Timesteps (Smoothed with {int(confidence_level*100)}% CI)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    for method in assignment_methods:
        if method not in all_mean_ep_lengths:
            continue
        df_mean_ep_lengths = pd.DataFrame(all_mean_ep_lengths[method]).transpose()
        if df_mean_ep_lengths.empty:
            continue
        mean_ep_lengths = df_mean_ep_lengths.mean(axis=0).rolling(window=window_size).mean()
        sem_ep_lengths = df_mean_ep_lengths.sem(axis=0).rolling(window=window_size).mean()
        degrees_of_freedom = df_mean_ep_lengths.count(axis=0) - 1
        degrees_of_freedom = degrees_of_freedom.clip(lower=1)
        t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        ci_ep_lengths = sem_ep_lengths * t_value
        plt.plot(all_timesteps, mean_ep_lengths, label=f"{method}", linewidth=2)  # Bold line
        plt.fill_between(all_timesteps, mean_ep_lengths - ci_ep_lengths, mean_ep_lengths + ci_ep_lengths, alpha=0.2)
    plt.xlabel("Total Timesteps (x1e6)")
    plt.ylabel("Mean Episode Length")
    plt.title(f"Mean Episode Length over Total Timesteps (Smoothed with {int(confidence_level*100)}% CI)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# def generate_plots(log_dir_base, window_size=10, confidence_level=0.9):
#     all_timesteps = None
#     all_mean_rewards = defaultdict(dict)
#     all_mean_ep_lengths = defaultdict(dict)

#     assignment_methods = os.listdir(log_dir_base)

#     for method in assignment_methods:
#         method_log_dir_base = os.path.join(log_dir_base, method)
#         iteration_dirs = os.listdir(method_log_dir_base)

#         for iteration_dir in iteration_dirs:
#             npz_path = os.path.join(method_log_dir_base, iteration_dir, "evaluations.npz")
#             eval_data = np.load(npz_path)

#             # Extract the relevant data
#             timesteps = eval_data['timesteps']
#             rewards = eval_data['results'].mean(axis=1)  # Mean over n_eval_episodes
#             ep_lengths = eval_data['ep_lengths'].mean(axis=1)  # Mean over n_eval_episodes

#             if all_timesteps is None:
#                 all_timesteps = timesteps

#             # Store the data in the dictionaries
#             all_mean_rewards[method][iteration_dir] = rewards
#             all_mean_ep_lengths[method][iteration_dir] = ep_lengths

#     # Plot results using Matplotlib
#     plt.figure(figsize=(12, 8))

#     plt.subplot(2, 1, 1)
#     for method in assignment_methods:
#         df_mean_rewards = pd.DataFrame(all_mean_rewards[method]).transpose()
#         mean_rewards = df_mean_rewards.mean(axis=0).rolling(window=window_size).mean()
#         sem_rewards = df_mean_rewards.sem(axis=0).rolling(window=window_size).mean()
#         t_value = stats.t.ppf((1 + confidence_level) / 2, df_mean_rewards.count(axis=0) - 1)
#         ci_rewards = sem_rewards * t_value
#         plt.plot(all_timesteps, mean_rewards, label=f"{method}")
#         plt.fill_between(all_timesteps, mean_rewards - ci_rewards, mean_rewards + ci_rewards, alpha=0.2)
#     plt.xlabel("Total Timesteps")
#     plt.ylabel("Mean Reward")
#     plt.title(f"Mean Reward over Total Timesteps (Smoothed with {int(confidence_level*100)}% CI)")
#     plt.legend()
#     plt.grid()

#     plt.subplot(2, 1, 2)
#     for method in assignment_methods:
#         df_mean_ep_lengths = pd.DataFrame(all_mean_ep_lengths[method]).transpose()
#         mean_ep_lengths = df_mean_ep_lengths.mean(axis=0).rolling(window=window_size).mean()
#         sem_ep_lengths = df_mean_ep_lengths.sem(axis=0).rolling(window=window_size).mean()
#         t_value = stats.t.ppf((1 + confidence_level) / 2, df_mean_ep_lengths.count(axis=0) - 1)
#         ci_ep_lengths = sem_ep_lengths * t_value
#         plt.plot(all_timesteps, mean_ep_lengths, label=f"{method}")
#         plt.fill_between(all_timesteps, mean_ep_lengths - ci_ep_lengths, mean_ep_lengths + ci_ep_lengths, alpha=0.2)
#     plt.xlabel("Total Timesteps")
#     plt.ylabel("Mean Episode Length")
#     plt.title(f"Mean Episode Length over Total Timesteps (Smoothed with {int(confidence_level*100)}% CI)")
#     plt.legend()
#     plt.grid()

#     plt.tight_layout()
#     plt.show()

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