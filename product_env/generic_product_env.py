import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sparse_reward_machine import SparseRewardMachine

class MultiAgentEnv(gym.Env):
    def __init__(self, labeled_mdp, reward_machine_file, num_agents, env_config, observation_space=spaces.Box(low=0, high=99, shape=(2,), dtype=int), action_space=spaces.Discrete(5)):
        super(MultiAgentEnv, self).__init__()
        self.num_agents = num_agents
        self.labeled_mdp = labeled_mdp
        self.reward_machine = SparseRewardMachine(reward_machine_file)
        self.env_config = env_config

        # Define observation and action spaces.
        self.action_space = gym.spaces.Dict({f"agent_{i}":action_space for i in range(self.num_agents)})# 5 possible actions.
        self.observation_space = gym.spaces.Dict({f"agent_{i}":observation_space for i in range(self.num_agents)})  # Observations include MDP and RM states.

        # Initialize agent states.
        self.reset()

    def reset(self, seed=None, options=None):
        self.mdp_states = self.env_config["initial_mdp_states"]
        self.rm_states = self.env_config["initial_rm_states"]
        assert(len(self.mdp_states) == self.num_agents == len(self.rm_states))
        self.current_step = 0

        obs = {f'agent_{i}': np.array([self.mdp_states[i], self.rm_states[i]]) for i in range(self.num_agents)}
        info = {}
        return obs, info

    def step(self, action_dict):
        rewards = {}
        obs = {}
        terminated = {}
        info = {}

        for i in range(self.num_agents):
            s = self.mdp_states[i]
            a = action_dict[f'agent_{i}']
            
            s_next = self.labeled_mdp.environment_step(s, a)
            self.mdp_states[i] = s_next
            
            # Get labels and rewards.
            labels = self.labeled_mdp.get_mdp_label(s_next)
            reward = 0

            # advance RM transitions for all agents on a given step to handle team & individual rm machines
            for j in range(self.num_agents):
                for label in labels:
                    u_next = self.reward_machine.get_next_state(self.rm_states[j], label)
                    reward += self.reward_machine.get_reward(self.rm_states[j], u_next)
                    self.rm_states[j] = u_next
            

            obs[f'agent_{i}'] = np.array([self.mdp_states[i], self.rm_states[i]])
            if reward > 1: 
                for i in range(self.num_agents):
                    rewards[f'agent_{i}'] = 1
            else:
                rewards[f'agent_{i}'] = reward
            terminated[f'agent_{i}'] = self.reward_machine.is_terminal_state(self.rm_states[i])
        
        terminated["__all__"] = all(terminated.values())
        truncated = self.current_step >= self.env_config["max_episode_length"]
        info['truncated'] = truncated
        self.current_step += 1

        return obs, rewards, terminated, truncated, info

    def render(self):
        # Optional: implement visualization if needed.
        pass

    def close(self):
        # Cleanup resources if needed.
        pass