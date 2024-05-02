# parameters needed:
# num agents
# reward Machine files
# 

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from product_env.generic_product_env import MultiAgentEnv
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
import yaml

if __name__ == '__main__':
    with open('config/buttons.yaml', 'r') as file:
        buttons_config = yaml.safe_load(file)

    mdp_labeled = HardButtonsLabeled(buttons_config)
    env = MultiAgentEnv(mdp_labeled, "reward_machines/buttons/aux_buttons.txt", 3, buttons_config)
    check_env(env)

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    # model.save("dqn_cartpole")

    # del model # remove to demonstrate saving and loading

    # model = DQN.load("dqn_cartpole")

    # obs, info = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         obs, info = env.reset()
