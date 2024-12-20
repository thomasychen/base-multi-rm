from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
from pettingzoo_product_env.overcooked_product_env import OvercookedProductEnv
from jaxmarl import make
from stable_baselines3.common.evaluation import evaluate_policy
from jaxmarl.environments.overcooked import Overcooked, overcooked_layouts
from pettingzoo.utils import parallel_to_aec
from reward_machines.sparse_reward_machine import SparseRewardMachine, dfa_to_rm
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
from manager.manager import Manager
# from threading import Lock
from wandb.integration.sb3 import WandbCallback
from multiprocessing import Lock, Manager as ProcessManager
from concurrent.futures import ProcessPoolExecutor
from mdp_label_wrappers.overcooked_custom_island_labeled import OvercookedCustomIslandLabeled
from mdp_label_wrappers.overcooked_cramped_labeled import OvercookedCrampedLabeled
from mdp_label_wrappers.overcooked_interesting_cramped_labeled  import OvercookedInterestingCrampedLabeled
import yaml
import argparse
from stable_baselines3.ppo import MlpPolicy

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser(description="Run reinforcement learning experiments with PettingZoo and Stable Baselines3.")
parser.add_argument('--assignment_methods', type=str, default="ground_truth naive random add multiply UCB", help='The assignment method for the manager. Default is "ground_truth".')
parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations for the experiment. Default is 5.')
parser.add_argument('--wandb', type=str2bool, default=False, help='Turn Wandb logging on or off. Default is off')
parser.add_argument('--timesteps', type=int, default=2000000, help='Number of timesteps to train model. Default is 2000000')
parser.add_argument('--decomposition_file', type=str, default="aux_buttons.txt",  help="The reward machine file for this decomposition")
parser.add_argument('--experiment_name', type=str, default="buttons", help="Name of config file for environment eg: ")
parser.add_argument('--is_monolithic', type=str2bool, default=False, help="If monolothic RM")
parser.add_argument('--num_candidates', type=int, default=0, help="Use automated decomposition for a monolithic reward machine. If 0, run the monolithic RM as is.")
parser.add_argument('--env', type=str, default="buttons", help="Specify between the buttons grid world or overcooked")
parser.add_argument('--render', type=str2bool, default=False, help='Enable rendering during training. Default is off')
parser.add_argument('--add_mono_file', type=str, default="None", help="Provide a monolithic file for global statekeeping along with a decomposed strategy")

args = parser.parse_args()

def run_trained_model(model_path, steps):
    max_steps = 400

    with open(f'config/{args.env}/{args.experiment_name}.yaml', 'r') as file:
        run_config = yaml.safe_load(file)
    manager = Manager(num_agents=run_config['num_agents'], num_decomps = len(run_config["initial_rm_states"]),assignment_method="ground_truth", wandb=args.wandb, seed = 1)
    run_config["render_mode"] = "human"
    train_rm = SparseRewardMachine(f"reward_machines/{args.env}/{args.experiment_name}/{args.decomposition_file}")
    mono_rm = SparseRewardMachine(f"reward_machines/{args.env}/{args.experiment_name}/{args.add_mono_file}") if args.add_mono_file != "None" else None
    if mono_rm is not None:
        mono_rm.is_monolithic = True
        
    train_kwargs = {
                'manager': manager,
                'labeled_mdp_class': eval(run_config['labeled_mdp_class']),
                'reward_machine': train_rm,
                'config': run_config,
                'max_agents': run_config['num_agents'],
                'is_monolithic': args.is_monolithic,
                'render_mode': run_config["render_mode"],
                'addl_mono_rm': mono_rm,
    }
    env = OvercookedProductEnv(**train_kwargs)

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
                act = model.predict(obs, deterministic=False)[0]

            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)

def test_dfa_generation():
    # load in RM
    brm = SparseRewardMachine("reward_machines/buttons/buttons/team_buttons.txt")
    # convert to DFA
    decomps = generate_rm_decompositions(brm, 3, 2, n_queries=100)
    return decomps

