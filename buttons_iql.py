from pettingzoo.test import parallel_api_test

from pettingzoo_product_env.custom_environment.env.custom_environment import MultiAgentEnvironment
import yaml
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
from stable_baselines3 import DQN, PPO, SAC, DDPG
from stable_baselines3.ddpg import MlpPolicy
from sb3_contrib import QRDQN
# from stable_baselines3.ppo import MlpPolicy
# from stable_baselines3.sac import MlpPolicy
import supersuit as ss
import glob
import os
import time
import numpy as np
def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    # env = env_fn.env(render_mode=render_mode, **env_kwargs)
    env = env_fn()

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = DDPG.load(latest_policy)

    reward = {agent: 0 for agent in env.possible_agents}
    steps = {agent:[] for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        observations, _ = env.reset(seed=i)
        # print(observations)
        print("\n\n\n")
        while env.agents:
        # this is where you would insert your policy
            actions = {agent: model.predict(observations[agent])[0] for agent in env.agents}
            print(actions)
            # print(actions)
            observations, rewards, terminations, truncations, infos = env.step(actions)
            # print(terminations, rewards, truncations)
            for agent in rewards:
                reward[agent]+=rewards[agent]
            
            if not all(truncations.values()):
                for i in terminations:
                    if terminations[i]:
                        steps[i].append(infos[i]["timesteps"])
            else:
                for i in truncations:
                    steps[i].append(env.env_config["max_episode_length"])
        env.close()
 

    avg_reward = sum(reward.values()) / len(reward.values())
    assert(len(list(steps.values())[0])  == num_games)
    print("Rewards: ", reward)
    print(f"Avg reward: {avg_reward}")
    print("timesteps per trajectory:", steps)
    for agent in env.possible_agents:
        print(f"avg timesteps for {agent}:",  sum(steps[agent])/len(steps[agent]))
    return avg_reward


if __name__ == "__main__":
    with open('config/buttons.yaml', 'r') as file:
        buttons_config = yaml.safe_load(file)
        
    mdp_labeled = HardButtonsLabeled(buttons_config)
    env = MultiAgentEnvironment()

    env.reset()

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.black_death_v3(env)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    n_actions = env.action_space.shape[-1]

    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = DDPG(
        MlpPolicy,
        env,
        verbose=3,
        action_noise=action_noise,
        # exploration_initial_eps= 1,
        # exploration_final_eps=0, 
        # exploration_fraction=0.96,
        # batch_size=256,
        # learning_rate=0.001,
        # target_update_interval=100,
        # gamma = 0.87,
        # buffer_size=5000,
    )

    model.learn(total_timesteps=100000)


    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

    env_kwargs = {}

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    eval(MultiAgentEnvironment, num_games=100, render_mode=None, **env_kwargs)