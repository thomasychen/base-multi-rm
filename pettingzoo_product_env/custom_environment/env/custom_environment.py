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


            # s_new = self.mdp_states[curr_agent]
            # # print(f"agent {i} at state {current_u}")
            # # s, a = agent_list[i].get_next_action(epsilon, learning_params)
            # # r, l, s_new = training_environments[i].environment_step(s,a)

            # # u2 = training_environments[i].u
            # # a = training_environments[i].get_last_action() # due to MDP slip
            # # # agent_list[i].update_agent(s_new, a, r, l, learning_params)
            # # if tester.get_current_step() > agent_list[i].buffer.max_: 
            # #     agent_list[i].update_agent(s_new, a, r, l, learning_params, tester.get_current_step())
            # # agent_list[i].buffer.add(s, current_u, a, r, s_new, u2)

            # for u in self.reward_machine.U:
            #     if not (u == current_u) and not (u in self.reward_machine.T) and not (u == self.reward_machine.u0):
            #     # if not (u == current_u) and not (u in agent_list[i].rm.T):
            #         new_l = self.labeled_mdp.get_mdp_label(s_new)
            #         new_r = 0
            #         u_temp = u
            #         u2 = u
            #         for e in new_l:
            #             # Get the new reward machine state and the reward of this step
            #             u2 = self.reward_machine.get_next_state(u_temp, e)
            #             new_r = new_r + self.reward_machine.get_reward(u_temp, u2)
            #             # Update the reward machine state
            #             u_temp = u2
            #         # agent_list[i].update_q_function(s, s_new, u, u2, a, r, learning_params)
            #         # if tester.get_current_step() > agent_list[i].buffer.max_: 
            #         #     agent_list[i].update_q_function(s, s_new, u, u2, a, r, learning_params, tester.get_current_step())
            #         ## keep MDP state the same
            #         prev_state = np.array([s, u])
            #         next_state = np.array([self.mdp_states[curr_agent], u2])

            #         big_prev_state = [np.array([self.mdp_states[agent], self.rm_states[agent]]) for agent in range(self.agents)]
            #         big_prev_state[curr_agent] = prev_state


            #         big_next_state = [np.array([self.mdp_states[agent], self.rm_states[agent]]) for agent in range(self.agents)]
            #         big_next_state[curr_agent] = next_state
                    

            #         is_done = self.reward_machine.is_terminal_state(u2)
            #         import pdb; pdb.set_trace()
            #         MultiAgentEnvironment.manager.model.replay_buffer.add(prev_state, next_state, a, new_r, is_done, infos)
            #         # agent_list[i].buffer.add(s, u, a, new_r, s_new, u2)

    
            observations[curr_agent] =  np.array([self.mdp_states[curr_agent], self.rm_states[curr_agent]])
            terminations[curr_agent] = self.reward_machine.is_terminal_state(self.rm_states[curr_agent])
            # if terminations[self.agents[i]] == True and len(actions.keys()) == 2:
            #     import pdb; pdb.set_trace();
            rewards[curr_agent] = r
        # print("TERMINATIONS", terminations)


        ### COUNTERFACTUAL EXPERIENCE LOOP ###
        for i in range(self.num_agents):
            if self.possible_agents[i] not in actions:
                continue
            curr_agent = self.possible_agents[i]
            current_u = old_rm_states[curr_agent]
            for u in self.reward_machine.U:
                if not (u == current_u) and not (u in self.reward_machine.T) and not (u == self.reward_machine.u0):
                # if not (u == current_u) and not (u in agent_list[i].rm.T):
                    new_l = self.labeled_mdp.get_mdp_label(self.mdp_states[curr_agent])
                    new_r = 0
                    u_temp = u
                    u2 = u
                    for e in new_l:
                        # Get the new reward machine state and the reward of this step
                        u2 = self.reward_machine.get_next_state(u_temp, e)
                        new_r = new_r + self.reward_machine.get_reward(u_temp, u2)
                        # Update the reward machine state
                        u_temp = u2
                    # agent_list[i].update_q_function(s, s_new, u, u2, a, r, learning_params)
                    # if tester.get_current_step() > agent_list[i].buffer.max_: 
                    #     agent_list[i].update_q_function(s, s_new, u, u2, a, r, learning_params, tester.get_current_step())
                    ## keep MDP state the same
                    prev_state = np.array([old_mdp_states[curr_agent], u])
                    next_state = np.array([self.mdp_states[curr_agent], u2])

                    big_prev_state = [np.array([old_mdp_states[agent], old_rm_states[agent]]) for agent in self.possible_agents]
                    big_prev_state[i] = prev_state


                    big_next_state = [np.array([self.mdp_states[agent], self.rm_states[agent]]) for agent in self.possible_agents]
                    big_next_state[i] = next_state
                    

                    big_actions = np.array([actions[agent] if agent in self.agents else 4 for agent in self.possible_agents])
                    
                    big_rewards = np.array([rewards[agent] if agent in self.agents else 0 for agent in self.possible_agents])
                    big_rewards[i] = new_r

                    is_done = self.reward_machine.is_terminal_state(u2)

                    big_done = np.array([terminations[agent] if agent in self.agents else True for agent in self.possible_agents])
                    big_done[i] = is_done

                    # import pdb; pdb.set_trace()
                    MultiAgentEnvironment.manager.model.replay_buffer.add(big_prev_state, big_next_state, big_actions, big_rewards, big_done, [{}, {}, {}])
                    # agent_list[i].buffer.add(s, u, a, new_r, s_new, u2)

        #### COUNTERFACTUAL EXPERIENCE LOOP ###


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
