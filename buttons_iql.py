from pettingzoo.test import parallel_api_test
from pettingzoo_product_env.pettingzoo_product_env import MultiAgentEnvironment
import yaml
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
from stable_baselines3 import DQN, PPO, SAC, DDPG, HER, HerReplayBuffer
# from stable_baselines3.ddpg import MlpPolicy
from stable_baselines3.ppo import MlpPolicy
# from stable_baselines3.sac import MlpPolicy
import supersuit as ss
import numpy as np
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3.common.logger import configure, read_csv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from manager.manager import Manager
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
from mdp_label_wrappers.easy_buttons_mdp_labeled import EasyButtonsLabeled
from mdp_label_wrappers.challenge_buttons_mdp_labeled import ChallengeButtonsLabeled
from reward_machines.sparse_reward_machine import SparseRewardMachine
from stable_baselines3.common.monitor import Monitor
from pettingzoo.test import parallel_seed_test
from stable_baselines3.common.utils import set_random_seed
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pandas as pd
from utils.plot_utils import generate_plots
import re
import torch as th

## WANDB KILL SWITCH
# ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9


# log_dir = "./logs/"
# os.makedirs(log_dir, exist_ok=True)
# new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Function to extract states connected to 0
def extract_states_from_text(text):
    pattern = re.compile(r'\(0, (\d+),')
    matches = pattern.findall(text)
    return list(map(int, matches))

# Function to read file and extract states
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
parser.add_argument('--decomposition_file', type=str, default="aux_buttons.txt",  help="The reward machine file for this decomposition")
parser.add_argument('--experiment_name', type=str, default="buttons", help="Name of config file for environment")
parser.add_argument('--is_monolithic', type=str2bool, default=False, help="If monolothic RM")
args = parser.parse_args()


if __name__ == "__main__":

    assignment_methods = args.assignment_methods.split()
    real_base = "./logs/"
    os.makedirs(real_base, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir_base = os.path.join(real_base, f"{timestamp}")
    os.makedirs(log_dir_base, exist_ok=True)

    for method in assignment_methods:

        method_log_dir_base = os.path.join(log_dir_base, f"{method}")
        os.makedirs(method_log_dir_base, exist_ok=True)

        for i in range(1, args.num_iterations + 1):
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

            with open(f'config/{args.experiment_name}.yaml', 'r') as file:
                buttons_config = yaml.safe_load(file)

            print(buttons_config)
            # num_agents = 3
            manager = Manager(num_agents=buttons_config['num_agents'], num_decomps = len(buttons_config["initial_rm_states"]),assignment_method=method, wandb=args.wandb, seed = i)
            train_rm = SparseRewardMachine(f"reward_machines/{args.experiment_name}/{args.decomposition_file}")

            # buttons_config["initial_rm_states"] = extract_states_from_file(args.decomposition_file)


            train_kwargs = {
                'manager': manager,
                'labeled_mdp_class': eval(buttons_config['labeled_mdp_class']),
                'reward_machine': train_rm,
                'config': buttons_config,
                'max_agents': buttons_config['num_agents'],
                'cer': args.cer,
                'is_monolithic': args.is_monolithic
            }

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
                                    log_path=log_dir, eval_freq=200,
                                    n_eval_episodes=10, deterministic=True,
                                    render=False)


            policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128])

            model = PPO(
                MlpPolicy,
                env,
                verbose=1,
                batch_size=256,
                learning_rate=buttons_config['learning_rate'],
                gamma = buttons_config['gamma'],
                tensorboard_log=f"runs/{run.id}" if args.wandb else None,
                max_grad_norm=buttons_config['max_grad_norm'],
                vf_coef=buttons_config['vf_coef'],
                # normalize_advantage=True,
                target_kl=buttons_config['target_kl'],
                ent_coef=buttons_config['ent_coef']
            )


            # model = DQN(
            #     "MlpPolicy",
            #     env,
            #     verbose=1,
            #     exploration_initial_eps= 1,
            #     exploration_final_eps=0.05, 
            #     exploration_fraction=0.1,
            #     batch_size=512,
            #     learning_rate=0.001,
            #     gamma = buttons_config['gamma'],
            #     buffer_size=5000,
            #     target_update_interval=100,
            #     tensorboard_log=f"runs/{run.id}" if args.wandb else None,
            #     max_grad_norm=1,
            # )
            
            # model = DQN(
            #     "MlpPolicy",
            #     env,
            #     verbose=1,
            #     exploration_initial_eps= 1,
            #     exploration_final_eps=0.05, 
            #     exploration_fraction=0.25,
            #     batch_size=5000,
            #     learning_rate=0.0001,
            #     gamma = buttons_config['gamma'],
            #     buffer_size=20000,
            #     target_update_interval=1000,
            #     tensorboard_log=f"runs/{run.id}" if args.wandb else None,
            #     max_grad_norm=1,
            # )
            # model = DQN(
            #     "MlpPolicy",
            #     env,
            #     verbose=1,
            #     exploration_initial_eps= 1,
            #     exploration_final_eps=0.05, 
            #     exploration_fraction=0.25,
            #     batch_size=512,
            #     learning_rate=0.0001,
            #     gamma = 0.88,
            #     buffer_size=7000,
            #     target_update_interval=100,
            #     tensorboard_log=f"runs/{run.id}" if args.wandb else None,
            #     max_grad_norm=1,
            #     policy_kwargs=policy_kwargs
            # )
            # model.set_logger(new_logger)
            
            # print("BUTTONSIQL", manager)
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

            model.learn(total_timesteps=args.timesteps, callback=callback_list, log_interval=10, progress_bar=False)


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
            
            # Read the log file for this iteration
            # log_path = os.path.join(log_dir, "progress.csv")
            # data_frame = read_csv(log_path)

            # all_mean_rewards.append(data_frame["eval/mean_reward"])
            # all_mean_ep_lengths.append(data_frame["eval/mean_ep_length"])


