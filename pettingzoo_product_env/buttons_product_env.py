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

        self.mdp_states = {self.agents[i]:mdp_state_array[i] for i in range(len(self.agents))}
        self.rm_states = {self.agents[i]:rm_array[decomp_idx][i] for i in range(len(self.agents))}

        observations = {i: self.flatten_and_add_rm(self.mdp_states[i], self.rm_states[i], idx) for idx, i in enumerate(self.agents)}

        infos = {agent: {} for agent in self.agents}
        self.state = observations

        self.traj_mdp_states.append(copy.deepcopy(self.mdp_states))

        return observations, infos


    def step(self, actions):
        observations = {}
        terminations = {}
        rewards = {}
        infos = {}

        old_rm_states = copy.deepcopy(self.rm_states)
        old_mdp_states = copy.deepcopy(self.mdp_states)

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

        mono_rm_reward = 0

        for i in range(len(self.possible_agents)):
            if self.possible_agents[i] not in actions:
                continue

            curr_agent = self.possible_agents[i]
            s = self.mdp_states[curr_agent]
            a = actions[curr_agent]

            #### FOR UPDATING AGENT FROM CURRENT STATE TO NEXT STATE ####
            current_u = self.rm_states[curr_agent]
            
            s_next = self.labeled_mdp.environment_step(s, a, i+1)
            self.mdp_states[curr_agent] = s_next

            r = 0
            for e in all_labels:
                u2 = self.reward_machine.get_next_state(self.rm_states[curr_agent], e)
                r = r + self.reward_machine.get_reward(self.rm_states[curr_agent], u2)
                self.rm_states[curr_agent] = u2
                if self.addl_monolithic_rm is not None:
                    next_ms = self.addl_monolithic_rm.get_next_state(self.monolithic_rm_state, e) #TODO: check that the order invariance here doesn't matter
                    mono_rm_reward += self.monolithic_weight*self.addl_monolithic_rm.get_reward(self.monolithic_rm_state, next_ms)
                    self.monolithic_rm_state = next_ms

    

            if hasattr(self.labeled_mdp, "u"):
                self.labeled_mdp.u[i+1] = self.rm_states[curr_agent]

             #### FOR UPDATING AGENT FROM CURRENT STATE TO NEXT STATE ####

    
            observations[curr_agent] =  np.array([self.mdp_states[curr_agent], self.rm_states[curr_agent]])
            rewards[curr_agent] = r

        self.traj_mdp_states.append(copy.deepcopy(self.mdp_states))    

        if self.addl_monolithic_rm is not None:
            for agent in rewards:
                rewards[agent] += mono_rm_reward


        for idx, curr_agent in enumerate(self.agents):
            observations[curr_agent] = self.flatten_and_add_rm(self.mdp_states[curr_agent], self.rm_states[curr_agent], idx)

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

        truncations = {agent: env_truncation for agent in self.agents}

        infos = {agent: {"timesteps": self.time_step} for agent in self.agents}
        
        if env_truncation:
            self.agents = []
            if self.test:
                for at in terminations:
                    terminations[at] = False
            else: 
                self.manager.update_rewards(0)
        else:
            self.agents = []
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

        if self.test:
            if all(terminations.values()) and any([i > 0 for i in rewards.values()]):
                for ar in rewards:
                    rewards[ar] = 1
            elif not all(terminations.values()):
                for ar in rewards:
                    rewards[ar] = 0
      
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
        mdp_shape = 1
        if self.addl_monolithic_rm is None:
            flattened_shape = [mdp_shape + self.reward_machine.get_one_hot_size(len(self.possible_agents))]
        else:
            flattened_shape = [mdp_shape + self.reward_machine.get_one_hot_size(len(self.possible_agents)) + self.addl_monolithic_rm.get_one_hot_size(len(self.possible_agents))]
        return Box(0, 255, flattened_shape)

    
    def action_space(self, agent):
        return Discrete(5)
    
    def discretize_action(self, continuous_action):

        discrete_action = int(np.round((continuous_action[0] + 1) * 2))  # Scale to range [0, 4]
        return discrete_action

    def flatten_and_add_rm(self, obs, rm_state, agent_idx):
        obs = np.array([obs])

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
