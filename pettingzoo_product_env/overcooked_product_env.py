from pettingzoo import ParallelEnv
from jaxmarl.environments.overcooked import Overcooked
import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from gymnasium.spaces import Box, Discrete

class OvercookedProductEnv(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, env:Overcooked, render_mode = None):
        self.possible_agents = ["agent_" + str(r) for r in range(2)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode 
        self.mdp = env
        self.states = []
        self.viz = OvercookedVisualizer()
        self.eps_reward = {agent: 0 for agent in self.possible_agents}
        self.reset_key = None

        # self.key, self.key_r, self.key_a = jax.random.split(key, 3)
    

    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(0, 255, self.mdp.obs_shape)

    def action_space(self, agent):
        # return self.mdp.action_space()
        return Discrete(len(self.mdp.action_set))
    
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        # self.num_moves = 0
        # observations = {agent: NONE for agent in self.agents}
        # infos = {agent: {} for agent in self.agents}
        # self.state = observations
        print("RESETTING")

        self.states = []
        if self.reset_key is None:
            self.reset_key = jax.random.PRNGKey(0)
        self.reset_key, key_r, key_a = jax.random.split(self.reset_key, 3)
        jax_observations, state = self.mdp.reset(key_r)

        print("reset step", state.time)
        self.curr_state = state
        self.states.append([self.curr_state])
        infos = {agent: {} for agent in self.agents}
        observations = {i: jnp.transpose(jax_observations[i], (1,0,2)) for i in jax_observations}
        return observations, infos
    
    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # {
        #     agent_1: 0, 
        #     agent_2: 1,
        # }
        #  {
        #      agent_1: np.array(0), 
             
        #  }

        self.reset_key, key_a0, key_a1, key_s = jax.random.split(self.reset_key, 4)

        jax_obs, state, jax_rewards, jax_dones, jax_infos = self.mdp.step(key_s, self.curr_state, actions)
        # print("step", state.time)
        self.curr_state = state
        self.states[-1].append(state)
        obs = {i: jnp.transpose(jax_obs[i], (1, 0, 2)) for i in jax_obs}
        rewards = {i: float(jax_rewards[i]) for i in jax_rewards}
        for agent, rew in rewards.items():
            self.eps_reward[agent] += rew
        dones = {i: bool(jax_dones[i]) for i in jax_dones}
        # print(dones)

        # If a user passes in actions with no agents, then just return empty observations, etc.
        # if not actions:
        #     self.agents = []
        #     return {}, {}, {}, {}, {}

        # # rewards for all agents are placed in the rewards dictionary to be returned
        # rewards = {}
        # rewards[self.agents[0]], rewards[self.agents[1]] = REWARD_MAP[
        #     (actions[self.agents[0]], actions[self.agents[1]])
        # ]

        # terminations = {agent: False for agent in self.agents}

        # self.num_moves += 1
        # env_truncation = self.num_moves >= NUM_ITERS
        # truncations = {agent: env_truncation for agent in self.agents}

        # # current observation is just the other player's most recent action
        # observations = {
        #     self.agents[i]: int(actions[self.agents[1 - i]])
        #     for i in range(len(self.agents))
        # }
        # self.state = observations

        # # typically there won't be any information in the infos, but there must
        # # still be an entry for each agent
        # infos = {agent: {} for agent in self.agents}
        # env_truncation = (state.time == 0)
        # truncations = {agent: False for agent in self.agents}
        # truncations["__all__"] = False

        if self.render_mode == "human":
            self.render()
        terminations = {i: False for i in actions}
        terminations["__all__"] = False

        if state.time == 0:
            print("episode reward: ", self.eps_reward)
            self.eps_reward = {agent: 0 for agent in self.possible_agents}
            # truncations = {i: True for i in actions}
            # truncations["__all__"] = True
            # jax_infos = {}
            # self.agents = []

        print("infos:", jax_infos)

        
        return obs, rewards, terminations, dones, jax_infos

    def render(self):
        self.viz.render(self.mdp.agent_view_size, self.curr_state, highlight=False)

