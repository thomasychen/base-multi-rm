from pettingzoo import ParallelEnv
from reward_machines.sparse_reward_machine import SparseRewardMachine
import yaml
from gymnasium.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import functools
import copy
import itertools
import random
from mdp_label_wrappers.easy_buttons_mdp_labeled import EasyButtonsLabeled

class MultiAgentEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, manager, labeled_mdp_class, reward_machine:SparseRewardMachine, config, max_agents, cer= True, test = False, render_mode=None):
        MultiAgentEnvironment.manager = manager

        # self.manager = manager

        self.render_mode = render_mode

        self.env_config = config
        self.cer = cer

        self.labeled_mdp = labeled_mdp_class(self.env_config)
        self.reward_machine = reward_machine
        self.test = test
        self.max_agents = max_agents
        self.local_manager = None


        self.possible_agents = ["agent_" + str(r) for r in range(self.max_agents)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.time_step = 0

        mdp_state_array = copy.deepcopy(self.env_config["initial_mdp_states"])
        rm_state_array = copy.deepcopy(self.env_config["initial_rm_states"]) if np.array(self.env_config["initial_rm_states"]).ndim == 2 else [copy.deepcopy(self.env_config["initial_rm_states"])]

        if not self.local_manager:
            self.local_manager = MultiAgentEnvironment.manager

        rm_assignments, decomp_idx = self.local_manager.get_rm_assignments(mdp_state_array, rm_state_array, test=self.test)
        self.labeled_mdp.reset(rm_assignments, decomp_idx)
        # rm_assignments = self.manager.get_rm_assignments(mdp_state_array, rm_state_array, test=self.test)


        # print("rm assignment", rm_assignments)

        self.mdp_states = {self.agents[i]:mdp_state_array[i] for i in range(len(self.agents))}
        self.rm_states = {self.agents[i]:rm_state_array[decomp_idx][rm_assignments[i]] for i in range(len(self.agents))}
        # print(self.rm_states, self.mdp_states)
        # print(self.rm_states)

        observations = {agent: np.array([self.mdp_states[agent], self.rm_states[agent]]) for agent in self.agents}
        # print(observations)

        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos



    def step(self, actions):
        # import pdb; pdb.set_trace();
        # If a user passes in actions with no agents, then just return empty observations, etc.
        # print("ACTIONS", actions)
        # print(self.rm_states)
        observations = {}
        terminations = {}
        rewards = {}
        infos = {}


        old_rm_states = copy.deepcopy(self.rm_states)
        old_mdp_states = copy.deepcopy(self.mdp_states)
        # if 10 in old_rm_states.values():
        #     import pdb;pdb.set_trace();

        all_labels = []
        for i in range(len(self.possible_agents)):
            if self.possible_agents[i] not in actions:
                continue
            curr_agent = self.possible_agents[i]
            s = self.mdp_states[curr_agent]
            a = actions[curr_agent]
            s_next = self.labeled_mdp.environment_step(s, a, i+1)
            labels = self.labeled_mdp.get_mdp_label(s_next, i+1, self.rm_states[curr_agent], self.test)
            crazies = [[] for i in range(len(self.possible_agents))]
            if type(self.labeled_mdp) == EasyButtonsLabeled:
                for label in labels:
                    # dont use crazies in our envs
                    if label == 'by' and self.labeled_mdp.get_state_description(s_next) != self.labeled_mdp.env_settings["yellow_button"]:
                        crazies[1].append(label)
                    elif label == 'bg' and self.labeled_mdp.get_state_description(s_next) != self.labeled_mdp.env_settings["green_button"]:
                        crazies[2].append(label)
                    elif label == "br" and self.labeled_mdp.get_state_description(s_next) != self.labeled_mdp.env_settings["red_button"]:
                        crazies[0].append(label)
                    elif label == 'a3br' and self.labeled_mdp.get_state_description(s_next) != self.labeled_mdp.env_settings["red_button"]:
                        crazies[1].append(label)
                    elif label == 'a2br' and self.labeled_mdp.get_state_description(s_next) != self.labeled_mdp.env_settings["red_button"]:
                        crazies[2].append(label)
                    elif label in self.reward_machine.delta_u[self.rm_states[curr_agent]]:
                        all_labels.append(label)
            else: 
                for label in labels:
                    if label in self.reward_machine.delta_u[self.rm_states[curr_agent]]:
                        all_labels.append(label)

        if "bg" in all_labels and self.test:
            print("\n\n\nYAYYYY\n\n\n")
                



        # print("ACTIONS", actions)
        for i in range(len(self.possible_agents)):
            if self.possible_agents[i] not in actions:
                continue
            curr_agent = self.possible_agents[i]
            s = self.mdp_states[curr_agent]
            a = actions[curr_agent]
            # a = self.discretize_action(ca)

            #### FOR UPDATING AGENT FROM CURRENT STATE TO NEXT STATE ####
            current_u = self.rm_states[curr_agent]
            
            s_next = self.labeled_mdp.environment_step(s, a, i+1)
            self.mdp_states[curr_agent] = s_next

            r = 0

            
            for e in all_labels + crazies[i]:
            # for e in all_labels:
                # import pdb; pdb.set_trace()
                u2 = self.reward_machine.get_next_state(self.rm_states[curr_agent], e)
                r = r + self.reward_machine.get_reward(self.rm_states[curr_agent], u2)
                self.rm_states[curr_agent] = u2

            # all_permutations = itertools.permutations(all_labels)

            # max_distance = -float('inf')
            # best_permutation = None

            # # Iterate through all permutations of the labels
            # for perm in all_permutations:
            #     temp_state = self.rm_states[curr_agent]
            #     temp_reward = 0
            #     temp_distance = 0  # To count the number of state transitions
                
            #     for e in perm:
            #         u2 = self.reward_machine.get_next_state(temp_state, e)
            #         temp_reward += self.reward_machine.get_reward(temp_state, u2)
                    
            #         if u2 != temp_state:  # Increment distance only if the state changes
            #             temp_distance += 1
                    
            #         temp_state = u2

            #     # Update the best permutation if this one has more state transitions
            #     if temp_distance > max_distance:
            #         max_distance = temp_distance
            #         best_permutation = perm

            # # Apply the best permutation to the current agent
            # for e in best_permutation:
            #     u2 = self.reward_machine.get_next_state(self.rm_states[curr_agent], e)
            #     r += self.reward_machine.get_reward(self.rm_states[curr_agent], u2)
            #     self.rm_states[curr_agent] = u2


            if hasattr(self.labeled_mdp, "u"):
                self.labeled_mdp.u[i+1] = self.rm_states[curr_agent]

             #### FOR UPDATING AGENT FROM CURRENT STATE TO NEXT STATE ####
            

            # # advance RM transitions for all agents on a given step to handle team & individual rm machines
            # for ak in self.agents:
            #     # print(self.possible_agents[j], self.rm_states[self.possible_agents[j]])
            #     for label in labels:
            #         u_next = self.reward_machine.get_next_state(self.rm_states[ak], label)
            #         r += self.reward_machine.get_reward(self.rm_states[ak], u_next)
            #         self.rm_states[ak] = u_next


    
            observations[curr_agent] =  np.array([self.mdp_states[curr_agent], self.rm_states[curr_agent]])
            terminations[curr_agent] = self.reward_machine.is_terminal_state(self.rm_states[curr_agent])
            # if terminations[self.agents[i]] == True and len(actions.keys()) == 2:
            #     import pdb; pdb.set_trace();
            rewards[curr_agent] = r
            # if self.test:
            #     for ar in self.agents:
            #         rewards[ar] = r
            #     print(rewards)
        # print("TERMINATIONS", terminations)
        # if len(actions) == self.num_agents:
        if not self.test and self.cer:
            all_permutations = list(itertools.permutations(self.reward_machine.U, self.max_agents))
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
                    infos = [{} for _ in range(len(self.possible_agents))]
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
                            new_l = self.labeled_mdp.get_mdp_label(s_new, k+1, u_old, self.test)
                            new_r = 0
                            a = actions[ag]
                            u_temp = u_old
                            u_new = u_old
                            for e in new_l:
                                # Get the new reward machine state and the reward of this step
                                u_new = self.reward_machine.get_next_state(u_temp, e)
                                new_r = new_r + self.reward_machine.get_reward(u_temp, u_new)
                                # Update the reward machine state
                                u_temp = u_new

                            done = self.reward_machine.is_terminal_state(u_new)

                        big_prev_states.append(np.array([s_old, u_old]))
                        big_new_states.append(np.array([s_new, u_new]))
                        big_actions.append(np.array([a]))
                        big_rewards.append(new_r)
                        big_dones.append(done)

                    # if sum([(big_rewards[i_a] == 1 and self.possible_agents[i_a] in actions and rewards[self.possible_agents[i_a]] != 1) for i_a in range(len(big_rewards))])> 0:
                    if sum([(big_rewards[i_a] == 1 and self.possible_agents[i_a] in actions) for i_a in range(len(big_rewards))])> 0:
                    # if sum([(big_rewards[i_a] == 1 and self.possible_agents[i_a] in actions and rewards[self.possible_agents[i_a]] != 1) or (big_rewards[i_a] == 0 and self.possible_agents[i_a] in actions and rewards[self.possible_agents[i_a]] == 1) for i_a in range(len(big_rewards))])> 0:
                        # print("\n\n\nHi\n\n\n")
                        # import pdb; pdb.set_trace()
                        self.local_manager.model.replay_buffer.add(big_prev_states, big_new_states, np.array(big_actions),np.array(big_rewards), np.array(big_dones), infos)

        for curr_agent in self.agents:
            observations[curr_agent] =  np.array([self.mdp_states[curr_agent], self.rm_states[curr_agent]])
            terminations[curr_agent] = self.reward_machine.is_terminal_state(self.rm_states[curr_agent])
            
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
            if self.test:
                for at in terminations:
                    terminations[at] = False
        else:
            self.agents = []
            # if any(terminations.values()) and self.test:
            #     print("\n\n\n\n\n\n", all(terminations.values()), "\n\n\n\n\n\n")
            if not self.test:
                for agent in terminations:
                    if not terminations[agent]:
                        self.agents.append(agent)
                if not self.agents:
                    self.manager.update_rewards(1)
            else:
                if not all(terminations.values()):
                    self.agents = self.possible_agents
                    for at in terminations:
                        terminations[at] = False
    
            # if not self.agents:
            #     print("FINISHED REWARDS", rewards)
        # print(self.rm_states)

        if self.test:
            if all(terminations.values()):
                for ar in rewards:
                    rewards[ar] = 1
            else:
                for ar in rewards:
                    rewards[ar] = 0
        # if "g" in all_labels and self.test and len(all_labels)>1:
        #     print(all_labels, actions, rewards)

        # if any(rewards.values()): 
        #     print(terminations)
        #     print()

        # print("\n\n\n", observations, rewards, terminations, truncations)
        # if all_labels:
        #     print(self.rm_states, actions, self.mdp_states, labels)
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
