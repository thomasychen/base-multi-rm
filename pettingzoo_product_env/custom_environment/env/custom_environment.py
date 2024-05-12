from pettingzoo import ParallelEnv
from sparse_reward_machine import SparseRewardMachine
import yaml
from mdp_label_wrappers.buttons_mdp_labeled import HardButtonsLabeled
from gymnasium.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import functools
import copy
import itertools

class MultiAgentEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, manager, render_mode=None):
        MultiAgentEnvironment.manager = manager

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
        with open('config/buttons.yaml', 'r') as file:
            self.env_config = yaml.safe_load(file)

    
        mdp_state_array = copy.deepcopy(self.env_config["initial_mdp_states"])
        
        rm_state_array = copy.deepcopy(self.env_config["initial_rm_states"])
        rm_assignments = MultiAgentEnvironment.manager.get_rm_assignments(mdp_state_array, rm_state_array)

        print("rm assignment", rm_assignments)

        self.mdp_states = {self.agents[i]:mdp_state_array[i] for i in range(len(self.agents))}
        self.rm_states = {self.agents[i]:rm_state_array[rm_assignments[i]] for i in range(len(self.agents))}

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
        infos = {}


        old_rm_states = copy.deepcopy(self.rm_states)
        old_mdp_states = copy.deepcopy(self.mdp_states)


        # print("ACTIONS", actions)
        for i in range(self.num_agents):
            if self.agents[i] not in actions:
                continue
            curr_agent = self.agents[i]
            s = self.mdp_states[curr_agent]
            a = actions[curr_agent]
            # a = self.discretize_action(ca)

            #### FOR UPDATING AGENT FROM CURRENT STATE TO NEXT STATE ####
            current_u = self.rm_states[curr_agent]
            
            s_next = self.labeled_mdp.environment_step(s, a)
            self.mdp_states[curr_agent] = s_next

            labels = self.labeled_mdp.get_mdp_label(s_next)
            r = 0
            for e in labels:
                u2 = self.reward_machine.get_next_state(self.rm_states[curr_agent], e)
                r = r + self.reward_machine.get_reward(self.rm_states[curr_agent], u2)
                self.rm_states[curr_agent] = u2

             #### FOR UPDATING AGENT FROM CURRENT STATE TO NEXT STATE ####


            # # advance RM transitions for all agents on a given step to handle team & individual rm machines
            # # for j in range(self.num_agents):
            # #     for label in labels:
            # #         u_next = self.reward_machine.get_next_state(self.rm_states[j], label)
            # #         reward += self.reward_machine.get_reward(self.rm_states[j], u_next)
            # #         self.rm_states[j] = u_next


    
            observations[curr_agent] =  np.array([self.mdp_states[curr_agent], self.rm_states[curr_agent]])
            terminations[curr_agent] = self.reward_machine.is_terminal_state(self.rm_states[curr_agent])
            # if terminations[self.agents[i]] == True and len(actions.keys()) == 2:
            #     import pdb; pdb.set_trace();
            rewards[curr_agent] = r
        # print("TERMINATIONS", terminations)

        all_permutations = list(itertools.permutations(self.reward_machine.U, 3))
        for perm in all_permutations:
            bools = [(perm[j] == old_rm_states[self.possible_agents[j]] or perm[j] in self.reward_machine.T or perm[j] == self.reward_machine.u0) for j in range(len(self.possible_agents))]
            if any(bools):
                continue
            else:
                # print(actions)
                big_prev_states = []
                big_new_states = []
                big_actions = []
                big_rewards = []
                big_dones = []
                infos = [{}, {}, {}]
                # print(self.num_agents)
                for k in range(len(self.possible_agents)):
                    ag = self.possible_agents[k]
                    s_old = old_mdp_states[ag]
                    if self.possible_agents[k] not in actions:
                        u_old = old_rm_states[ag]
                        u_new = old_rm_states[ag]
                        s_new = old_mdp_states[ag]
                        new_r = 1
                        a = 4
                        done = True
                    else: 
                        u_old = perm[k]
                        s_new = self.mdp_states[ag]
                        new_l = self.labeled_mdp.get_mdp_label(s_new)
                        new_r = 0
                        a = actions[ag]
                        u_temp = u_old
                        u_new = u_old
                        for e in new_l:
                            # Get the new reward machine state and the reward of this step
                            u_new = self.reward_machine.get_next_state(u_temp, u_new)
                            new_r = new_r + self.reward_machine.get_reward(u_temp, u_new)
                            # Update the reward machine state
                            u_temp = u_new

                        done = self.reward_machine.is_terminal_state(u_new)

                    big_prev_states.append(np.array([s_old, u_old]))
                    big_new_states.append(np.array([s_new, u_new]))
                    big_actions.append(np.array([a]))
                    big_rewards.append(new_r)
                    big_dones.append(done)

                MultiAgentEnvironment.manager.model.replay_buffer.add(big_prev_states, big_new_states, np.array(big_actions),np.array(big_rewards), np.array(big_dones), infos)

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
            #     print("FINISHED REWARDS", rewards)
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
        return Discrete(5)
        # return Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
    def discretize_action(self, continuous_action):
    # Assume continuous_action is a numpy array with values between -1 and 1
    # Scale and round to nearest discrete action
        discrete_action = int(np.round((continuous_action[0] + 1) * 2))  # Scale to range [0, 4]
        return discrete_action
