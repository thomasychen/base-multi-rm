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
from mdp_label_wrappers.generic_mdp_labeled import MDP_Labeler
import os

class ButtonsProductEnv(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, manager, labeled_mdp_class: MDP_Labeler, reward_machine: SparseRewardMachine, config, max_agents, test=False, is_monolithic=False, addl_mono_rm: SparseRewardMachine=None, render_mode=None, monolithic_weight=1.0, log_dir=None, video=False):
        ButtonsProductEnv.manager = manager

        # self.manager = manager

        self.render_mode = render_mode

        self.env_config = config

        self.labeled_mdp = labeled_mdp_class(self.env_config)
        self.reward_machine = reward_machine
        self.test = test
        self.max_agents = max_agents
        self.local_manager = None

        self.is_monolithic = is_monolithic


        self.possible_agents = ["agent_" + str(r) for r in range(self.max_agents)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Modified parameters post workshop paper
        self.addl_monolithic_rm = addl_mono_rm # Potentially give the monolithic here so everyone know's global states (for potentially dependent dynamics)
        self.monolithic_weight = monolithic_weight
        self.log_dir = log_dir
        self.video = video
        self.traj_mdp_states = []
    
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.time_step = 0
        self.traj_mdp_states = []

        mdp_state_array = copy.deepcopy(self.env_config["initial_mdp_states"])
        rm_array = copy.deepcopy(self.env_config["initial_rm_states"]) if np.array(self.env_config["initial_rm_states"]).ndim == 2 else [copy.deepcopy(self.env_config["initial_rm_states"])]

        rm_state_array = [[self.reward_machine.get_one_hot_encoded_state(state, len(self.possible_agents), idx) for idx, state in enumerate(init_states)] for init_states in rm_array]


        if not self.local_manager:
            self.local_manager = ButtonsProductEnv.manager

        if self.addl_monolithic_rm is not None:
            self.monolithic_rm_state = self.addl_monolithic_rm.get_initial_state()

        decomp_idx = self.local_manager.get_rm_assignments(mdp_state_array, rm_state_array, test=self.test)
        self.labeled_mdp.reset(decomp_idx)
        # rm_assignments = self.manager.get_rm_assignments(mdp_state_array, rm_state_array, test=self.test)


        # print("rm assignment", rm_assignments)

        self.mdp_states = {self.agents[i]:mdp_state_array[i] for i in range(len(self.agents))}
        self.rm_states = {self.agents[i]:rm_array[decomp_idx][i] for i in range(len(self.agents))}
        # print(self.rm_states, self.mdp_states)
        # print(self.rm_states)

        observations = {i: self.flatten_and_add_rm(self.mdp_states[i], self.rm_states[i], idx) for idx, i in enumerate(self.agents)}

        # observations = {agent: np.array([self.mdp_states[agent], self.rm_states[agent]]) for agent in self.agents}
        # print(observations)

        infos = {agent: {} for agent in self.agents}
        self.state = observations

        # import pdb; pdb.set_trace()
        self.traj_mdp_states.append(copy.deepcopy(self.mdp_states))

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
            labels = self.labeled_mdp.get_mdp_label(s_next, i+1, self.rm_states[curr_agent], self.test, self.is_monolithic)
            all_labels.extend(labels)
            # crazies = [[] for i in range(len(self.possible_agents))]
            # if type(self.labeled_mdp) == EasyButtonsLabeled:
            #     for label in labels:
            #         # dont use crazies in our envs
            #         if label == 'by' and self.labeled_mdp.get_state_description(s_next) != self.labeled_mdp.env_settings["yellow_button"]:
            #             crazies[1].append(label)
            #         elif label == 'bg' and self.labeled_mdp.get_state_description(s_next) != self.labeled_mdp.env_settings["green_button"]:
            #             crazies[2].append(label)
            #         elif label == "br" and self.labeled_mdp.get_state_description(s_next) != self.labeled_mdp.env_settings["red_button"]:
            #             crazies[0].append(label)
            #         elif label == 'a3br' and self.labeled_mdp.get_state_description(s_next) != self.labeled_mdp.env_settings["red_button"]:
            #             crazies[1].append(label)
            #         elif label == 'a2br' and self.labeled_mdp.get_state_description(s_next) != self.labeled_mdp.env_settings["red_button"]:
            #             crazies[2].append(label)
            #         elif label in self.reward_machine.delta_u[self.rm_states[curr_agent]]:
            #             all_labels.append(label)
            # else: 
                # for label in labels:
                #     if label in self.reward_machine.delta_u[self.rm_states[curr_agent]]:
                #         all_labels.append(label)

        # if 'g' in all_labels:
        #     import pdb; pdb.set_trace()
        # if all_labels:
        #     print(all_labels, "\n\n")
        # # print()
                  
        if "bg" in all_labels and self.test:
            print("\n\n\nYAYYYY\n\n\n")
            

        # if "br" in all_labels:
        #     import pdb; pdb.set_trace()



        # print("ACTIONS", actions)
        mono_rm_reward = 0

        for i in range(len(self.possible_agents)):
            if self.possible_agents[i] not in actions:
                continue
            # if len(actions) < 3:
            #     import pdb; pdb.set_trace()
            curr_agent = self.possible_agents[i]
            s = self.mdp_states[curr_agent]
            a = actions[curr_agent]
            # a = self.discretize_action(ca)

            #### FOR UPDATING AGENT FROM CURRENT STATE TO NEXT STATE ####
            current_u = self.rm_states[curr_agent]
            
            s_next = self.labeled_mdp.environment_step(s, a, i+1)
            self.mdp_states[curr_agent] = s_next

            r = 0
            for e in all_labels:
            # for e in all_labels:
                # import pdb; pdb.set_trace()
                u2 = self.reward_machine.get_next_state(self.rm_states[curr_agent], e)
                r = r + self.reward_machine.get_reward(self.rm_states[curr_agent], u2)
                self.rm_states[curr_agent] = u2
                if self.addl_monolithic_rm is not None:
                    next_ms = self.addl_monolithic_rm.get_next_state(self.monolithic_rm_state, e) #TODO: check that the order invariance here doesn't matter
                    mono_rm_reward += self.monolithic_weight*self.addl_monolithic_rm.get_reward(self.monolithic_rm_state, next_ms)
                    self.monolithic_rm_state = next_ms

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
            # terminations[curr_agent] = self.reward_machine.is_terminal_state(self.rm_states[curr_agent])
            # if terminations[self.agents[i]] == True and len(actions.keys()) == 2:
            #     import pdb; pdb.set_trace();
            rewards[curr_agent] = r
            # if self.test:

        # import pdb; pdb.set_trace()
        self.traj_mdp_states.append(copy.deepcopy(self.mdp_states))    

        if self.addl_monolithic_rm is not None:
            for agent in rewards:
                rewards[agent] += mono_rm_reward


        for idx, curr_agent in enumerate(self.agents):
            # observations[curr_agent] =  np.array([self.mdp_states[curr_agent], self.rm_states[curr_agent]])
            observations[curr_agent] = self.flatten_and_add_rm(self.mdp_states[curr_agent], self.rm_states[curr_agent], idx)
            # terminations[curr_agent] = self.reward_machine.is_terminal_state(self.rm_states[curr_agent])

        terminations = {i: False for i in self.agents}
        if self.addl_monolithic_rm is not None:
            if self.addl_monolithic_rm.is_terminal_state(self.monolithic_rm_state):
                terminations = {i: True for i in self.agents}
        else:
            if all([self.reward_machine.is_terminal_state(self.rm_states[i]) for i in self.agents]):
                terminations = {i: True for i in self.agents}
        

            
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
                self.manager.update_rewards(0)
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
        # print("BUTTONS RM STATES", self.rm_states)

        if self.test:
            if all(terminations.values()) and any([i > 0 for i in rewards.values()]):
                for ar in rewards:
                    rewards[ar] = 1
            elif not all(terminations.values()):
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

        # import pdb; pdb.set_trace()
        # observations = {i: self.flatten_and_add_rm(self.mdp_states[i], self.rm_states[i]) for i in self.agents}
        if self.render_mode == "human":
            self.render()
        
        if self.video and (env_truncation or all(terminations.values())):
            self.send_animation()


        for k in rewards:
            rewards[k] = max(0, rewards[k])
        return observations, rewards, terminations, truncations, infos

    def render(self):
        self.labeled_mdp.show(self.mdp_states)

    def observation_space(self, agent):
        # return self.observation_spaces[agent]
        mdp_shape = 1
        if self.addl_monolithic_rm is None:
            flattened_shape = [mdp_shape + self.reward_machine.get_one_hot_size(len(self.possible_agents))]
        else:
            flattened_shape = [mdp_shape + self.reward_machine.get_one_hot_size(len(self.possible_agents)) + self.addl_monolithic_rm.get_one_hot_size(len(self.possible_agents))]
        return Box(0, 255, flattened_shape)

        # return Box(low=0, high=99, shape= (2,))
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

    def flatten_and_add_rm(self, obs, rm_state, agent_idx):
        # import pdb; pdb.set_trace();

        # n = len(self.reward_machine.get_states())
        # # Flatten the 3D observation array
        # flattened_obs = obs.flatten()
        obs = np.array([obs])

        # # Create an n-length array of zeros
        # n = len(self.reward_machine.get_states())
        # rm_array = np.zeros(n)

        # # Set the rm_state index to 1
        # rm_array[rm_state] = 1
        rm_ohe = self.reward_machine.get_one_hot_encoded_state(rm_state, len(self.possible_agents), agent_idx)
        if self.addl_monolithic_rm is not None:
            mono_ohe = self.addl_monolithic_rm.get_one_hot_encoded_state(self.monolithic_rm_state, len(self.possible_agents), agent_idx)
            result = np.concatenate((obs, rm_ohe, mono_ohe))
        else:
            # Concatenate the flattened observation and the rm_array
            result = np.concatenate((obs, rm_ohe))

        return result
    def send_animation(self):
        path_dir = f"{self.log_dir}"
        os.makedirs(path_dir, exist_ok=True)
        self.labeled_mdp.animate(self.traj_mdp_states, filename=f"{path_dir}/viz.gif")
