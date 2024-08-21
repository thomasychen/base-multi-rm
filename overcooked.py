# from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
# from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedEnvPettingZoo
# from overcooked_ai.src.human_aware_rl.rllib.rllib import load_agent_pair
import supersuit as ss
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from pettingzoo_product_env.overcooked_product_env import OvercookedProductEnv
from jaxmarl import make
from jaxmarl.environments.overcooked import Overcooked, overcooked_layouts, layout_grid_to_dict
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import torch.nn as nn
import argparse
from buttons_iql import str2bool
import wandb
from datetime import datetime
import os
from wandb.integration.sb3 import WandbCallback


parser = argparse.ArgumentParser(description="Run reinforcement learning experiments with PettingZoo and Stable Baselines3.")
parser.add_argument('--wandb', type=str2bool, default=False, help='Turn Wandb logging on or off. Default is off')
parser.add_argument('--render', type=str2bool, default=False, help='Enable rendering during training. Default is off')
args = parser.parse_args()

if __name__ == '__main__':
    real_base = "./logs/"
    os.makedirs(real_base, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir_base = os.path.join(real_base, f"{timestamp}")
    os.makedirs(log_dir_base, exist_ok=True)
    if args.wandb:
        experiment = "overcooked_pettingzoo_sb3"
        config = {
            "policy_type": "MlpPolicy",
            "total_timesteps": 4000000,
            "env_name": "OvercookedPettingZoo",
        }

        wandb_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"overcooked_{wandb_timestamp}"

        run = wandb.init(
            project=experiment,
            entity="reinforce-learn",
            config=config,
            sync_tensorboard=True,
            name=run_name
        )
    
    max_steps = 400
    layout = overcooked_layouts["cramped_room"]
    jax_env = make('overcooked', layout=layout, max_steps=max_steps)
    render_mode = "human" if args.render else None
    env = OvercookedProductEnv(jax_env, render_mode = render_mode)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    jax_eval_env = make('overcooked', layout=layout, max_steps=max_steps)
    eval_env = OvercookedProductEnv(jax_eval_env, test=True)
    eval_env = ss.black_death_v3(eval_env)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, 1, num_cpus=1, base_class="stable_baselines3")
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{log_dir_base}/best/",
                                    log_path=log_dir_base, eval_freq=4000, deterministic=True)

    model = PPO(
    MlpPolicy,
    env,
    tensorboard_log=f"runs/{run.id}" if args.wandb else None,
    verbose=1,
    learning_rate=2.5e-4,  # LR
    gamma=0.99,  # GAMMA
    gae_lambda=0.95,  # GAE_LAMBDA
    clip_range=0.2,  # CLIP_EPS
    ent_coef=0.01,  # ENT_COEF
    vf_coef=0.5,  # VF_COEF
    max_grad_norm=0.5,  # MAX_GRAD_NORM
    n_epochs=4,  # UPDATE_EPOCHS
    batch_size=256,  # Can be tuned based on NUM_MINIBATCHES and NUM_ACTORS
    policy_kwargs=dict(
        activation_fn=nn.Tanh
    )
    )
    model.learning_rate = lambda frac: 2.5e-4 * frac

    env.reset()

    if args.wandb:
        callback_list = CallbackList([eval_callback, WandbCallback(verbose=2,)])
    else:
        callback_list = CallbackList([eval_callback])
        print("Disabled Wa&B for this training run. Logging will not occur.")

    model.learn(total_timesteps = 2500000, callback=callback_list, progress_bar=True)
    if args.wandb:
        wandb.finish()
    # env.close()
    
