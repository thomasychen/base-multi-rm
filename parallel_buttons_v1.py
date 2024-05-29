import os
import re
import yaml
import wandb
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from stable_baselines3 import DQN, PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
import supersuit as ss
from reward_machines.sparse_reward_machine import SparseRewardMachine
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
from pettingzoo_product_env.pettingzoo_product_env import MultiAgentEnvironment
from manager.manager import Manager
# from threading import Lock
from wandb.integration.sb3 import WandbCallback
from multiprocessing import Lock, Manager as ProcessManager
from concurrent.futures import ProcessPoolExecutor

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

def run_experiment(method, iteration, args, log_dir_base, global_lock):
    global_lock.acquire()
    set_random_seed(iteration)

    if args.wandb:
        experiment = "test_pettingzoo_sb3"
        config = {
            "policy_type": "MlpPolicy",
            "total_timesteps": 1000000,
            "env_name": "Buttons",
        }

        wandb_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{method}_iteration_{iteration}_{wandb_timestamp}"

        run = wandb.init(
            project=experiment,
            entity="reinforce-learn",
            config=config,
            sync_tensorboard=True,
            name=run_name
        )

    with open(f'config/{args.experiment_name}.yaml', 'r') as file:
                buttons_config = yaml.safe_load(file)

    # num_agents = 3
    manager = Manager(num_agents=buttons_config['num_agents'], num_decomps = len(buttons_config["initial_rm_states"]), assignment_method=method, wandb=args.wandb, seed=iteration)
    train_rm = SparseRewardMachine(f"reward_machines/{args.experiment_name}/{args.decomposition_file}")

    # buttons_config["initial_rm_states"] = extract_states_from_file(args.decomposition_file)

    train_kwargs = {
        'manager': manager,
        'labeled_mdp_class': HardButtonsLabeled,
        'reward_machine': train_rm,
        'config': buttons_config,
        'max_agents': buttons_config['num_agents'],
        'cer': args.cer,
    }

    env = MultiAgentEnvironment(**train_kwargs)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    method_log_dir_base = os.path.join(log_dir_base, f"{method}")
    os.makedirs(method_log_dir_base, exist_ok=True)
    log_dir = os.path.join(method_log_dir_base, f"iteration_{iteration}")
    os.makedirs(log_dir, exist_ok=True)

    eval_kwargs = train_kwargs.copy()
    eval_kwargs['test'] = True

    eval_env = MultiAgentEnvironment(**eval_kwargs)
    eval_env = ss.black_death_v3(eval_env)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, 1, num_cpus=1, base_class="stable_baselines3")
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(eval_env, best_model_save_path=None,
                                 log_path=log_dir, eval_freq=200,
                                 n_eval_episodes=10, deterministic=True,
                                 render=False)
    
    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        batch_size=256,
        learning_rate=buttons_config['learning_rate'],
        gamma = buttons_config['gamma'],
        tensorboard_log=f"runs/{run.id}"
    )

    # model = DQN(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     exploration_initial_eps=1,
    #     exploration_final_eps=0.05,
    #     exploration_fraction=0.35,
    #     batch_size=5000,
    #     learning_rate=0.0001,
    #     gamma=0.9,
    #     buffer_size=20000,
    #     target_update_interval=1000,
    #     tensorboard_log=f"runs/{run.id}" if args.wandb else None,
    #     max_grad_norm=1,
    # )

    manager.set_model(model)
    env.reset()

    if args.wandb:
        callback_list = CallbackList([eval_callback, WandbCallback(verbose=2)])
    else:
        callback_list = CallbackList([eval_callback])
    
    global_lock.release()


    model.learn(total_timesteps=args.timesteps, callback=callback_list, log_interval=100, progress_bar=False)

    env.close()

    if args.wandb:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Run reinforcement learning experiments with PettingZoo and Stable Baselines3.")
    parser.add_argument('--assignment_methods', type=str, default="ground_truth naive random add multiply UCB", help='The assignment method for the manager. Default is "ground_truth".')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations for the experiment. Default is 5.')
    parser.add_argument('--wandb', type=str2bool, default=False, help='Turn Wandb logging on or off. Default is off')
    parser.add_argument('--timesteps', type=int, default=2000000, help='Number of timesteps to train model. Default is 2000000')
    parser.add_argument('--cer', type=str2bool, default=True, help='Turn CER on or off' )
    parser.add_argument('--decomposition_file', type=str, default="aux_buttons.txt",  help="The reward machine file for this decomposition")
    parser.add_argument('--experiment_name', type=str, default="buttons", help="Name of config file for environmen")
    args = parser.parse_args()

    assignment_methods = args.assignment_methods.split()
    real_base = "./logs/"
    os.makedirs(real_base, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_base = os.path.join(real_base, f"{timestamp}")
    os.makedirs(log_dir_base, exist_ok=True)

    with ProcessManager() as manager:
        global_lock = manager.Lock()
        with ProcessPoolExecutor(max_workers = 10) as executor:
            futures = []
            for method in assignment_methods:
                for i in range(1, args.num_iterations + 1):
                    futures.append(executor.submit(run_experiment, method, i, args, log_dir_base, global_lock))

            for future in futures:
                future.result()

if __name__ == "__main__":
    main()