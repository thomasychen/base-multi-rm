# from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
# from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedEnvPettingZoo
# from overcooked_ai.src.human_aware_rl.rllib.rllib import load_agent_pair
import supersuit as ss
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from mdps.overcooked_mdp import OvercookedWrapperEnv
if __name__ == '__main__':
    env = OvercookedWrapperEnv().env
    # env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    # import pdb; pdb.set_trace();

    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        batch_size=256
    )
    print("hi")

    env.reset()
    print("hi1")

    model.learn(total_timesteps = 2000000)

    env.close()
    
