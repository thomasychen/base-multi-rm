import numpy as np
from utils.plot_utils import generate_plots
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from pettingzoo.test import parallel_api_test
from pettingzoo_product_env.pettingzoo_product_env import MultiAgentEnvironment
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
import supersuit as ss
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from manager.manager import Manager
from reward_machines.sparse_reward_machine import SparseRewardMachine
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
import threading

lock = threading.Lock()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def extract_states_from_text(text):
    pattern = re.compile(r'\(0, (\d+),')
    matches = pattern.findall(text)
    return list(map(int, matches))

def extract_states_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return extract_states_from_text(content)
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

parser = argparse.ArgumentParser(description="Run reinforcement learning experiments with PettingZoo and Stable Baselines3.")
parser.add_argument('--assignment_methods', type=str, default="ground_truth naive random add multiply UCB", help='The assignment method for the manager. Default is "ground_truth".')
parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations for the experiment. Default is 5.')
parser.add_argument('--wandb', type=str2bool, default=False, help='Turn Wandb logging on or off. Default is off')
parser.add_argument('--timesteps', type=int, default=2000000, help='Number of timesteps to train model. Default is 2000000')
parser.add_argument('--cer', type=str2bool, default=True, help='Turn CER on or off' )
parser.add_argument('--decomposition_file', type=str, default="reward_machines/buttons/aux_buttons.txt",  help="The reward machine file for this decomposition")
args = parser.parse_args()

def run_experiment(method, i, args, log_dir_base):

    method_log_dir_base = os.path.join(log_dir_base, f"{method}")
    os.makedirs(method_log_dir_base, exist_ok=True)

    print(f"Starting experiment with method: {method}, iteration: {i}")
    set_random_seed(i)

    if args.wandb:
        experiment = "test_pettingzoo_sb3"
        config = {
            "policy_type": "MlpPolicy",
            "total_timesteps": 1000000,
            "env_name": "Buttons",
        }

        wandb_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{method}_iteration_{i}_{wandb_timestamp}"

        run = wandb.init(
            project=experiment,
            entity="reinforce-learn",
            config=config,
            sync_tensorboard=True,
            name=run_name
        )

    with open('config/buttons.yaml', 'r') as file:
        buttons_config = yaml.safe_load(file)

    num_agents = 3
    manager = Manager(num_agents=num_agents, assignment_method=method, wandb=args.wandb, seed = i)
    train_rm = SparseRewardMachine(args.decomposition_file)

    buttons_config["initial_rm_states"] = extract_states_from_file(args.decomposition_file)

    train_kwargs = {
        'manager': manager,
        'labeled_mdp_class': HardButtonsLabeled,
        'reward_machine': train_rm,
        'config': buttons_config,
        'max_agents': 3,
        'cer': args.cer,
    }

    lock.acquire()

    env = MultiAgentEnvironment(**train_kwargs)

    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    log_dir = os.path.join(method_log_dir_base, f"iteration_{i}")
    os.makedirs(log_dir, exist_ok=True)
    # new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])


    # eval_log_dir = "./eval_logs/"
    # os.makedirs(eval_log_dir, exist_ok=True)
    eval_kwargs = train_kwargs.copy()
    eval_kwargs['test'] = True

    eval_env = MultiAgentEnvironment(**eval_kwargs)
    eval_env = ss.black_death_v3(eval_env)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, 1, num_cpus=1, base_class="stable_baselines3")
    eval_env = VecMonitor(eval_env)


    eval_callback = EvalCallback(eval_env, best_model_save_path=None,
                            log_path=log_dir, eval_freq=100,
                            n_eval_episodes=10, deterministic=True,
                            render=False)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        exploration_initial_eps= 1,
        exploration_final_eps=0.05, 
        exploration_fraction=0.25,
        batch_size=5000,
        learning_rate=0.0001,
        gamma = 0.9,
        buffer_size=20000,
        target_update_interval=1000,
        tensorboard_log=f"runs/{run.id}" if args.wandb else None,
        max_grad_norm=1,
    )
    # model.set_logger(new_logger)

    manager.set_model(model)
    env.reset()



    # callback_list = None
    # callback_list = CallbackList([eval_callback, WandbCallback(verbose=2,)])


    if args.wandb:
        callback_list = CallbackList([eval_callback, WandbCallback(verbose=2,)])
        print("RETARD\n\n")

    else:
        callback_list = CallbackList([eval_callback])
        print("DUMBASSSS")

    lock.release()

    model.learn(total_timesteps=args.timesteps, callback=callback_list, log_interval=100, progress_bar=False)


    # # model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    # print("Model has been saved.")
    # print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

    # data = np.load('./eval_logs/evaluations.npz')
    # test_steps = data['ep_lengths'].mean(axis=1, keepdims=True)
    # test_reward = data['results'].mean(axis=1,keepdims = True)
    
    # # Log the array to wandb with the index as x-axis
    # for i, length in enumerate(test_steps):
    #     wandb.log({"Test Mean Episode Length": test_steps[i][0], "Test Mean Episode Reward": test_reward[i][0]})

    # Finish your run
    if args.wandb:
        wandb.finish()
    # wandb.finish()
# Main execution block
if __name__ == "__main__":
    assignment_methods = args.assignment_methods.split()
    real_base = "./logs/"
    os.makedirs(real_base, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_base = os.path.join(real_base, f"{timestamp}")
    os.makedirs(log_dir_base, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.num_iterations*len(assignment_methods)) as executor:
        futures = []
        for method in assignment_methods:
            for i in range(1, args.num_iterations + 1):
                futures.append(executor.submit(run_experiment, method, i, args, log_dir_base))

        for future in as_completed(futures):
            try:
                future.result()  # Handle results and exceptions
            except Exception as e:
                print(f"Error during parallel execution: {e}")

