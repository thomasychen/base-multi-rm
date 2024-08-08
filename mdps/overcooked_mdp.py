import gymnasium as gym
# from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
# from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnvPettingZoo
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent
# from human_aware_rl.rllib.rllib import load_agent_pair
# bc_agent = RlLibAgent(bc_policy, 0, base_env.featurize_state_mdp)


class OvercookedWrapperEnv():
    def __init__(self):
        # mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
        # base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
        # self.env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.lossless_state_encoding_mdp)

        mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
        base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
        a1, a2 = RandomAgent(all_actions=True), RandomAgent(all_actions=True)
        a1.featurize, a2.featurize = base_env.lossless_state_encoding_mdp, base_env.lossless_state_encoding_mdp
        
        # a1.featurize, a2.featurize = base_env.featurize_state_mdp, base_env.featurize_state_mdp
        agent_pair = AgentPair(a1, a2)
        # agent_pair = load_agent_pair("/temp", "ppo", "ppo")
        self.env = OvercookedEnvPettingZoo(base_env, agent_pair, render_mode = "human")
        # self.env = gym.make("Overcooked-v0")
        # self.env.custom_init(base_env=base_env, featurize_fn=base_env.lossless_state_encoding_mdp)

        # self.base_mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
        # self.env = OvercookedEnv.from_mdp(self.base_mdp, horizon=500)

    
        
