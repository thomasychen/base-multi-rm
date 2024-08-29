from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
from pettingzoo_product_env.overcooked_product_env import OvercookedProductEnv
from jaxmarl import make
from stable_baselines3.common.evaluation import evaluate_policy
from jaxmarl.environments.overcooked import Overcooked, overcooked_layouts
from pettingzoo.utils import parallel_to_aec
from reward_machines.sparse_reward_machine import SparseRewardMachine, dfa_to_rm, generate_rm_decompositions
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
from manager.manager import Manager
# from threading import Lock
from wandb.integration.sb3 import WandbCallback
from multiprocessing import Lock, Manager as ProcessManager
from concurrent.futures import ProcessPoolExecutor

def run_trained_model(model_path, steps):
    # Define environment and configuration
    max_steps = 400
    layout = overcooked_layouts["cramped_room"]
    jax_eval_env = make('overcooked', layout=layout, max_steps=max_steps)
    env = OvercookedProductEnv(jax_eval_env, render_mode = "human", test = True)
    # eval_env = ss.black_death_v3(eval_env)
    # eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    # eval_env = ss.concat_vec_envs_v1(eval_env, 1, num_cpus=1, base_class="stable_baselines3")
    # eval_env = VecMonitor(eval_env)

    # def raw_env(render_mode=None):
    #     """
    #     To support the AEC API, the raw_env() function just uses the from_parallel
    #     function to convert from a ParallelEnv to an AEC env
    #     """
    #     # env = parallel_env(render_mode=render_mode)
    #     env = parallel_to_aec(env)
    #     return env

    env = parallel_to_aec(env)


    # Load the trained model
    model = PPO.load(model_path)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    num_games = 1
    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    
    # model.learn(total_timesteps = steps*max_steps)
    # vec_env = model.get_env() 
    # obs = vec_env.reset()
    # obs = eval_env.reset()
    # for i in range(max_steps):
    #     action, _states = model.predict(obs, deterministic=True) 
    #     # print("CURRENT I", i, action)
    #     obs, rewards, dones, info = eval_env.step(action) 
    #     eval_env.render("human")
    # Evaluate the model for the given number of steps
    # mean_reward, std_reward = evaluate_policy(model, eval_env)
    

    # print(f"Mean reward: {mean_reward} +/- {std_reward}")

def test_dfa_generation():
    # load in RM
    brm = SparseRewardMachine("reward_machines/buttons/buttons/team_buttons.txt")
    # convert to DFA
    decomps = generate_rm_decompositions(brm, 3, 2, n_queries=25)
    return decomps