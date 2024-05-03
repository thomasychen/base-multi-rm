from pettingzoo import ParallelEnv
from sparse_reward_machine import SparseRewardMachine
import yaml
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
from gymnasium.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import functools
import copy

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
        # self._action_spaces = {agent: Discrete(5) for agent in self.possible_agents}
        # self._observation_spaces = {
        #     agent: Box(low=0, high=99, shape=(2,)) for agent in self.possible_agents
        # }

        # # Define observation and action spaces.
        # self.action_space = gym.spaces.Dict({f"agent_{i}":action_space for i in range(self.num_agents)})# 5 possible actions.
        # self.observation_space = gym.spaces.Dict({f"agent_{i}":observation_space for i in range(self.num_agents)})  # Observations include MDP and RM states.


    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.time_step = 0
        mdp_state_array =  copy.deepcopy(self.env_config["initial_mdp_states"])
        rm_state_array = copy.deepcopy(self.env_config["initial_rm_states"])
        self.mdp_states = {self.agents[i]:mdp_state_array[i] for i in range(len(self.agents))}
        self.rm_states = {self.agents[i]:rm_state_array[i] for i in range(len(self.agents))}

        observations = {agent: np.array([self.mdp_states[agent], self.rm_states[agent]]) for agent in self.agents}
        # print(observations)

        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos


    def step(self, actions):
        # If a user passes in actions with no agents, then just return empty observations, etc.
        # print("ACTIONS", actions)
        # print(self.rm_states)
        observations = {}
        terminations = {}
        rewards = {}
        # print("ACTIONS", actions)
        for i in range(self.num_agents):
            if self.agents[i] not in actions:
                continue
            curr_agent = self.agents[i]
            s = self.mdp_states[curr_agent]
            ca = actions[curr_agent]
            a = self.discretize_action(ca)
            
            s_next = self.labeled_mdp.environment_step(s, a)
            # print(s_next)
            self.mdp_states[curr_agent] = s_next
            # if len(actions.keys())==1:
            #     print(actions.keys(), self.labeled_mdp.get_mdp_label(s_next), self.labeled_mdp.get_state_description(s_next))
            # Get labels and rewards.
            labels = self.labeled_mdp.get_mdp_label(s_next)
            r= 0

            # advance RM transitions for all agents on a given step to handle team & individual rm machines
            # for j in range(self.num_agents):
            #     for label in labels:
            #         u_next = self.reward_machine.get_next_state(self.rm_states[j], label)
            #         reward += self.reward_machine.get_reward(self.rm_states[j], u_next)
            #         self.rm_states[j] = u_next

            for e in labels:
                # print("HIIIII\n", self.agents[i], self.rm_states[i])
                # Get the new reward machine state and the reward of this step
                u2 = self.reward_machine.get_next_state(self.rm_states[curr_agent], e)
                # print(u2)
                # print(i, self.rm_states[i], u2, self.reward_machine.get_reward(self.rm_states[i], u2))
                r = r + self.reward_machine.get_reward(self.rm_states[curr_agent], u2)
                # Update the reward machine state
                self.rm_states[curr_agent] = u2
    
            observations[curr_agent] = np.array([self.mdp_states[curr_agent], self.rm_states[curr_agent]])
            terminations[curr_agent] = self.reward_machine.is_terminal_state(self.rm_states[curr_agent])
            # if terminations[self.agents[i]] == True and len(actions.keys()) == 2:
            #     import pdb; pdb.set_trace();
            rewards[curr_agent] = r
        # print("TERMINATIONS", terminations)
        
        self.state = observations

        self.time_step += 1
        env_truncation = self.time_step >= self.env_config["max_episode_length"]
        # print("TRUNCATE", env_truncation)
        # print("TERMINATIONS", terminations)
        # print("\n\nOBS", self.state)
        truncations = {agent: env_truncation for agent in self.agents}

        infos = {agent: {"timesteps": self.time_step} for agent in self.agents}

        if env_truncation:
            self.agents = []
            # print("TRUNCATED REWARDS", rewards)
        else:
            self.agents = []
            for agent in terminations:
                if not terminations[agent]:
                    self.agents.append(agent)
            # if not self.agents:
                # print("FINISHED REWARDS", rewards)
        # print(self.rm_states)
            
        # print("\n\n\n", observations, rewards, terminations, truncations)
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return NotImplementedError

    def observation_space(self, agent):
        # return self.observation_spaces[agent]
        return Box(low=0, high=99, shape=(2,))
        # return MultiDiscrete([100, 100])
        # return Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
    
    def action_space(self, agent):
        # return self.action_spaces[agent]
        # return Discrete(5)
        return Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
    def discretize_action(self, continuous_action):
    # Assume continuous_action is a numpy array with values between -1 and 1
    # Scale and round to nearest discrete action
        discrete_action = int(np.round((continuous_action[0] + 1) * 2))  # Scale to range [0, 4]
        return discrete_action
