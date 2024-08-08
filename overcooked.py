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
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn

if __name__ == '__main__':
    max_steps = 100
    layout = overcooked_layouts["cramped_room"]
    jax_env = make('overcooked', layout=layout, max_steps=max_steps)
    env = OvercookedProductEnv(jax_env, render_mode = "human")
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    jax_eval_env = make('overcooked', layout=layout, max_steps=max_steps)
    eval_env = OvercookedProductEnv(jax_eval_env)
    # eval_env = ss.black_death_v3(eval_env)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, 1, num_cpus=1, base_class="stable_baselines3")
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(eval_env, eval_freq=500,
                             deterministic=True, render=False)

    # import pdb; pdb.set_trace();
    logger = configure()

    model = PPO(
    MlpPolicy,
    env,
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
    print("hi")
    model.set_logger(logger)

    env.reset()
    print("hi1")

    model.learn(total_timesteps = 1000000)

    # env.close()
    
