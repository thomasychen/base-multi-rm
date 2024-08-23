from abc import ABC, abstractmethod
from pettingzoo import ParallelEnv

class MultiAgentEnvironment(ParallelEnv, ABC):
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