# parameters needed:
# num agents
# reward Machine files
# 


from pettingzoo.test import parallel_api_test

from pettingzoo_product_env.custom_environment.env.custom_environment import MultiAgentEnvironment
import yaml
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
from stable_baselines3 import DQN
from stable_baselines3.ppo import MlpPolicy
import supersuit as ss
import glob
import os
import time

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

    model = DQN.load(latest_policy)

    reward = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        env.reset(seed=i)

        # for agent in env.agent_iter():
        #     obs, reward, termination, truncation, info = env.last()

        #     for a in env.agents:
        #         rewards[a] += env.rewards[a]
        #     if termination or truncation:
        #         break
        #     else:
        #         act = model.predict(obs, deterministic=True)[0]

        #     env.step(act)
        while env.agents:
    # this is where you would insert your policy
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent in rewards:
                reward[agent]+=rewards[agent]
        env.close()
 

    avg_reward = sum(reward.values()) / len(reward.values())
    print("Rewards: ", reward)
    print(f"Avg reward: {avg_reward}")
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
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        # learning_rate=1e-3,
        # batch_size=256,
    )

    model.learn(total_timesteps=1000)


    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

    env_kwargs = {}

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    eval(MultiAgentEnvironment, num_games=10, render_mode=None, **env_kwargs)

# # Watch 2 games
# eval(env, num_games=2, render_mode="human")



# env = env_fn.parallel_env()

# env.reset()

# print(f"Starting training on {str(env.metadata['name'])}.")

# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

# # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
# model = PPO(
#     MlpPolicy,
#     env,
#     verbose=3,
#     learning_rate=1e-3,
#     batch_size=256,
# )

# model.learn(total_timesteps=steps)

# from stable_baselines3 import DQN
# from stable_baselines3.common.env_checker import check_env
# from product_env.generic_product_env import MultiAgentEnv
# from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
# import yaml

# if __name__ == '__main__':
#     with open('config/buttons.yaml', 'r') as file:
#         buttons_config = yaml.safe_load(file)

#     mdp_labeled = HardButtonsLabeled(buttons_config)
#     env = MultiAgentEnv(mdp_labeled, "reward_machines/buttons/aux_buttons.txt", 3, buttons_config)
#     check_env(env)

#     model = DQN("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=10000, log_interval=4)
    # model.save("dqn_cartpole")

    # del model # remove to demonstrate saving and loading

    # model = DQN.load("dqn_cartpole")

    # obs, info = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         obs, info = env.reset()
