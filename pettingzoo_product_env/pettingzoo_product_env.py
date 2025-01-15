from abc import ABC, abstractmethod
from pettingzoo import ParallelEnv
from mdp_label_wrappers.generic_mdp_labeled import MDP_Labeler
from reward_machines.sparse_reward_machine import SparseRewardMachine
import numpy as np

class MultiAgentEnvironment(ParallelEnv, ABC):
    def __init__(self, manager, labeled_mdp_class: MDP_Labeler, reward_machine: SparseRewardMachine, config, max_agents, test=False, is_monolithic=False, addl_mono_rm: SparseRewardMachine=None, render_mode=None, monolithic_weight=1.0, log_dir=None, video=False):
        self.render_mode = render_mode
        self.env_config = config
        self.max_agents = max_agents
        self.possible_agents = ["agent_" + str(r) for r in range(self.max_agents)]
        self.labeled_mdp = labeled_mdp_class(config)
        self.reward_machine = reward_machine
        self.test = test
        self.addl_monolithic_rm = addl_mono_rm
        self.monolithic_weight = monolithic_weight
        self.log_dir = log_dir
        self.video = video
        self.traj_mdp_states = []
        self.local_manager = None
        
    @abstractmethod
    def observation_space(self, agent):
        pass
    
    @abstractmethod
    def action_space(self, agent):
        pass

    @abstractmethod
    def reset(self, seed=None, options=None):
        pass
    
    @abstractmethod
    def step(self, actions):
        pass
    
    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def send_animation(self):
        pass

    def flatten_and_add_rm(self, obs, rm_state, agent_idx):
        rm_ohe = self.reward_machine.get_one_hot_encoded_state(rm_state, len(self.possible_agents), agent_idx)
        if self.addl_monolithic_rm is not None:
            mono_ohe = self.addl_monolithic_rm.get_one_hot_encoded_state(self.monolithic_rm_state, len(self.possible_agents), agent_idx)
            result = np.concatenate((obs, rm_ohe, mono_ohe))
        else:
            # Concatenate the flattened observation and the rm_array
            result = np.concatenate((obs, rm_ohe))

        return result