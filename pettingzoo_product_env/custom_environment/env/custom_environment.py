from pettingzoo import ParallelEnv
from sparse_reward_machine import SparseRewardMachine
import yaml
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
from gymnasium.spaces import Discrete, Box
import numpy as np
import functools

class MultiAgentEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, render_mode=None):
        # super(MultiAgentEnv, self).__init__()

        self.render_mode = render_mode

        # self.num_agents = 3
        with open('config/buttons.yaml', 'r') as file:
            self.env_config = yaml.safe_load(file)

        self.labeled_mdp = HardButtonsLabeled(self.env_config)
        self.reward_machine = SparseRewardMachine("reward_machines/buttons/aux_buttons.txt")


        #  = env_config

        self.possible_agents = ["agent_" + str(r) for r in range(3)]


        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._action_spaces = {agent: Discrete(5) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Box(low=0, high=99, shape=(2,)) for agent in self.possible_agents
        }

        # # Define observation and action spaces.
        # self.action_space = gym.spaces.Dict({f"agent_{i}":action_space for i in range(self.num_agents)})# 5 possible actions.
        # self.observation_space = gym.spaces.Dict({f"agent_{i}":observation_space for i in range(self.num_agents)})  # Observations include MDP and RM states.


    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.time_step = 0
        
        self.mdp_states = self.env_config["initial_mdp_states"]
        self.rm_states = self.env_config["initial_rm_states"]

        observations = {agent: (self.mdp_states[i], self.rm_states[i]) for i, agent in enumerate(self.agents)}

        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos


    def step(self, actions):
        # If a user passes in actions with no agents, then just return empty observations, etc.
        # print("ACTIONS", actions)
    
        observations = {}
        terminations = {}
        rewards = {}
        
        for i in range(self.num_agents):
            if self.agents[i] not in actions:

                continue
            s = self.mdp_states[i]
            a = actions[self.agents[i]]
            
            s_next = self.labeled_mdp.environment_step(s, a)
            self.mdp_states[i] = s_next
            
            # Get labels and rewards.
            labels = self.labeled_mdp.get_mdp_label(s_next)
            reward = 0

            # advance RM transitions for all agents on a given step to handle team & individual rm machines
            # for j in range(self.num_agents):
            for label in labels:
                u_next = self.reward_machine.get_next_state(self.rm_states[i], label)
                reward += self.reward_machine.get_reward(self.rm_states[i], u_next)
                self.rm_states[i] = u_next

            rewards[self.agents[i]] = reward
            observations[self.agents[i]] = np.array((self.mdp_states[i], self.rm_states[i]))
            terminations[self.agents[i]] = self.reward_machine.is_terminal_state(self.rm_states[i])
        
        self.state = observations

        self.time_step += 1
        env_truncation = self.time_step >= self.env_config["max_episode_length"]
        # print("TRUNCATE", env_truncation)
        # print("TERMINATIONS", terminations)
        # print("\n\nOBS", self.state)
        truncations = {agent: env_truncation for agent in self.agents}

        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []
        else:
            self.agents = []
            for agent in terminations:
                if not terminations[agent]:
                    self.agents.append(agent)
            
        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # return self.observation_spaces[agent]
        return Box(low=0, high=99, shape=(2,))
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # return self.action_spaces[agent]
        return Discrete(5)
