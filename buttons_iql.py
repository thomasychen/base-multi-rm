from pettingzoo.test import parallel_api_test
from pettingzoo_product_env.custom_environment.env.custom_environment import MultiAgentEnvironment
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

## WANDB KILL SWITCH
# ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9

tmp_path = "/tmp/sb3_log/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


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
    steps = {agent:[] for agent in env.possible_agents}

    for i in range(num_games):
        observations, _ = env.reset(seed=i)
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
    experiment = "test_pettingzoo_sb3"
    config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": "Buttons",
    }
    run = wandb.init(project = experiment, entity="reinforce-learn", config=config, sync_tensorboard=True )
    with open('config/buttons.yaml', 'r') as file:
        buttons_config = yaml.safe_load(file)

        
    # mdp_labeled = HardButtonsLabeled(buttons_config)

    num_agents = 3
    manager = Manager(num_agents=num_agents, assignment_method="multiply")
    env = MultiAgentEnvironment(manager=manager)

    # env.set_manager(manager)
    # env.reset()


    print(f"Starting training on {str(env.metadata['name'])}.")



    env = ss.black_death_v3(env)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    goal_selection_strategy = "future"

    model = DQN(
        "MlpPolicy",
        env,
        verbose=3,
        # action_noise=action_noise,
        exploration_initial_eps= 1,
        exploration_final_eps=0.15, 
        exploration_fraction=0.25,
        batch_size=5000,
        learning_rate=0.0001,
        # target_update_interval=100,
        gamma = 0.88,
        buffer_size=5000,
        target_update_interval=5000,
        tensorboard_log=f"runs/{run.id}"
    )

    # model = DQN(
    #     "MlpPolicy",
    #     env,
    #     verbose=3,
    #     # action_noise=action_noise,
    #     exploration_initial_eps= 1,
    #     exploration_final_eps=0.05, 
    #     exploration_fraction=0.25,
    #     batch_size=5000,
    #     learning_rate=0.0001,
    #     replay_buffer_class=HerReplayBuffer,
    #     replay_buffer_kwargs=dict(
    #     n_sampled_goal=4,
    #     goal_selection_strategy=goal_selection_strategy,
    #     ),
    #     # target_update_interval=100,
    #     gamma = 0.9,
    #     buffer_size=10000,
    #     target_update_interval=1000,
    #     tensorboard_log=f"runs/{run.id}"
    # )


    

    # model = PPO2(
    #     MlpPolicy,
    #     env,
    #     verbose=3,
    #     # action_noise=action_noise,
    #     # exploration_initial_eps= 1,
    #     # exploration_final_eps=0.05, 
    #     # exploration_fraction=0.25,
    #     # batch_size=5000,
    #     # learning_rate=0.0001,
    #     # # target_update_interval=100,
    #     # gamma = 0.9,
    #     # buffer_size=10000,
    #     # target_update_interval=1000,
    #     tensorboard_log=f"runs/{run.id}"
    # )


    manager.set_model(model)
    env.reset()


    model.learn(total_timesteps=2000000, callback=WandbCallback(
        verbose=2,
    ))


    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

    env_kwargs = {}

    # # Evaluate 10 games (average reward should be positive but can vary significantly)
    # eval(MultiAgentEnvironment, num_games=100, render_mode=None, **env_kwargs)