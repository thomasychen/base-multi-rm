from pettingzoo.test import parallel_api_test
from pettingzoo_product_env.pettingzoo_product_env import MultiAgentEnvironment
import yaml
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
from stable_baselines3 import DQN, PPO, SAC, DDPG, HER, HerReplayBuffer
# from stable_baselines3.ddpg import MlpPolicy
from sb3_contrib import QRDQN
from stable_baselines3.ppo import MlpPolicy
# from stable_baselines3.sac import MlpPolicy
import supersuit as ss
import glob
import os
import time
import numpy as np
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from manager.manager import Manager
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
from reward_machines.sparse_reward_machine import SparseRewardMachine
from stable_baselines3.common.monitor import Monitor
from pettingzoo.test import parallel_seed_test
from stable_baselines3.common.utils import set_random_seed
import argparse
from datetime import datetime

## WANDB KILL SWITCH
# ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9


# tmp_path = "/tmp/sb3_log/"
# set up logger
# new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

parser = argparse.ArgumentParser(description="Run reinforcement learning experiments with PettingZoo and Stable Baselines3.")
parser.add_argument('--assignment_method', type=str, default='ground_truth', help='The assignment method for the manager. Default is "ground_truth".')
parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations for the experiment. Default is 5.')
args = parser.parse_args()


if __name__ == "__main__":

    for i in range(1, args.num_iterations + 1):
        set_random_seed(i)

        experiment = "test_pettingzoo_sb3"
        config = {
            "policy_type": "MlpPolicy",
            "total_timesteps": 1000000,
            "env_name": "Buttons",
        }

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{args.assignment_method}_iteration_{i}_{timestamp}"

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
        manager = Manager(num_agents=num_agents, assignment_method=args.assignment_method, seed = i)
        train_rm = SparseRewardMachine("reward_machines/buttons/aux_buttons.txt")

        train_kwargs = {
            'manager': manager,
            'labeled_mdp_class': HardButtonsLabeled,
            'reward_machine': train_rm,
            'config_file_name': "config/buttons.yaml",
            'max_agents': 3
        }

        env = MultiAgentEnvironment(**train_kwargs)
   
        env = ss.black_death_v3(env)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
        env = VecMonitor(env)


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
                                log_path=None, eval_freq=100,
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
            tensorboard_log=f"runs/{run.id}",
            max_grad_norm=1,
        )

        manager.set_model(model)
        env.reset()

        callback_list = CallbackList([eval_callback, WandbCallback(verbose=2,)])
        model.learn(total_timesteps=2000000, callback=callback_list, log_interval=100, progress_bar=True)


        # # model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

        # print("Model has been saved.")
        # print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

        env.close()

        data = np.load('./eval_logs/evaluations.npz')
        test_steps = data['ep_lengths'].mean(axis=1, keepdims=True)
        test_reward = data['results'].mean(axis=1,keepdims = True)
        
        # Log the array to wandb with the index as x-axis
        for i, length in enumerate(test_steps):
            wandb.log({"Test Mean Episode Length": test_steps[i][0], "Test Mean Episode Reward": test_reward[i][0]})

        # Finish your run
        wandb.finish()

    # # Evaluate 10 games (average reward should be positive but can vary significantly)
    # eval(MultiAgentEnvironment, num_games=100, render_mode=None, **env_kwargs)