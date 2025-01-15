import jax
import numpy as np
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from gymnasium.spaces import Box, Discrete
from mdp_label_wrappers.generic_mdp_labeled import MDP_Labeler
from reward_machines.sparse_reward_machine import SparseRewardMachine
import copy
from datetime import datetime
import wandb
import os
from .pettingzoo_product_env import MultiAgentEnvironment
class OvercookedProductEnv(MultiAgentEnvironment):
    metadata = {
        "name": "custom_environment_v0", #TODO: change name
    }

    def __init__(self, manager, labeled_mdp_class: MDP_Labeler, reward_machine: SparseRewardMachine, 
                 config, max_agents, test=False, is_monolithic=False, addl_mono_rm: SparseRewardMachine=None, 
                 render_mode=None, monolithic_weight=1.0, log_dir=None, video=False):
        super().__init__(manager, labeled_mdp_class, reward_machine, 
                         config, max_agents, test, is_monolithic, addl_mono_rm, 
                         render_mode, monolithic_weight, log_dir, video)
        
        OvercookedProductEnv.manager = manager

        ###### FOR VISUALIZING ######
        self.viz = OvercookedVisualizer()
        ###### FOR VISUALIZING ######

        ###### OVERCOOKED SPECIFIC VARIABLES ######
        self.mdp = self.labeled_mdp.jax_env
        self.reset_key = None
        ###### OVERCOOKED SPECIFIC VARIABLES ######

    def observation_space(self, agent):
        if self.addl_monolithic_rm is None:
            flattened_shape = [self.labeled_mdp.obs_shape + self.reward_machine.get_one_hot_size(len(self.possible_agents))]
        else:
            flattened_shape = [self.labeled_mdp.obs_shape + self.reward_machine.get_one_hot_size(len(self.possible_agents)) + self.addl_monolithic_rm.get_one_hot_size(len(self.possible_agents))]
        return Box(0, 255, flattened_shape)
    
    def action_space(self, agent):
        return Discrete(len(self.mdp.action_set))
    
    def reset(self, seed=None, options=None):

        if not self.local_manager:
            self.local_manager = OvercookedProductEnv.manager

        self.agents = self.possible_agents[:]
        self.timestep = 0

        self.traj_mdp_states = []
        if self.reset_key is None:
            self.reset_key = jax.random.PRNGKey(0)
        self.reset_key, key_r, key_a = jax.random.split(self.reset_key, 3)
        jax_observations, state = self.mdp.reset(key_r)
        jax_observations = self.labeled_mdp.trim_observation(jax_observations)

        rm_array = copy.deepcopy(self.env_config["initial_rm_states"]) if np.array(self.env_config["initial_rm_states"]).ndim == 2 else [copy.deepcopy(self.env_config["initial_rm_states"])]
        
        if self.addl_monolithic_rm is not None:
            self.monolithic_rm_state = self.addl_monolithic_rm.get_initial_state()

        rm_state_array = [[self.reward_machine.get_one_hot_encoded_state(state, len(self.possible_agents), idx) for idx, state in enumerate(init_states)] for init_states in rm_array]

        mdp_state_array = [jax_observations[agent].flatten() for agent in self.agents]

        decomp_idx = self.local_manager.get_rm_assignments(mdp_state_array, rm_state_array, test=self.test)

        self.rm_states = {self.agents[i]: rm_array[decomp_idx][i] for i in range(len(self.agents))}
        
        print("reset step", state.time)
        self.curr_state = state
        self.traj_mdp_states.append(self.curr_state)
        infos = {agent: {} for agent in self.agents}

        observations = {}
        for idx, i in enumerate(self.agents):
            observations[i] = self.flatten_and_add_rm(jax_observations[i], self.rm_states[i], idx)
      
        return observations, infos
    
    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        false_actions = {agent: 4 for agent in self.possible_agents}
        for agent, action in actions.items():
            false_actions[agent] = action

        self.reset_key, key_a0, key_a1, key_s = jax.random.split(self.reset_key, 4)

        jax_obs, state, jax_rewards, jax_dones, jax_infos = self.mdp.step(key_s, self.curr_state, false_actions)

        jax_obs, labels = self.labeled_mdp.get_mdp_label(state, jax_rewards)

        rm_rewards = {}
        mono_rm_reward = 0
        for i in range(len(self.possible_agents)):
            agent = self.possible_agents[i]
            r = 0
            for e in labels:
                u2 = self.reward_machine.get_next_state(self.rm_states[agent], e)
                r = r + self.reward_machine.get_reward(self.rm_states[agent], u2)
                self.rm_states[agent] = u2
                if self.addl_monolithic_rm is not None:
                    next_ms = self.addl_monolithic_rm.get_next_state(self.monolithic_rm_state, e) #TODO: check that the order invariance here doesn't matter
                    mono_rm_reward += self.monolithic_weight*self.addl_monolithic_rm.get_reward(self.monolithic_rm_state, next_ms)
                    self.monolithic_rm_state = next_ms
                    
            rm_rewards[agent] = r

        if self.addl_monolithic_rm is not None:
            for agent in self.possible_agents:
                rm_rewards[agent] += mono_rm_reward
        

        terminations = {i: False for i in self.agents}
        if self.addl_monolithic_rm is not None:
            if self.addl_monolithic_rm.is_terminal_state(self.monolithic_rm_state):
                terminations = {i: True for i in self.agents}
        else:
            if all([self.reward_machine.is_terminal_state(self.rm_states[i]) for i in self.agents]):
                terminations = {i: True for i in self.agents}

        
        self.curr_state = state
        self.traj_mdp_states.append(state)
        obs = {i: self.flatten_and_add_rm(jax_obs[i], self.rm_states[i], idx) for idx, i in enumerate(self.agents)}

        self.timestep += 1
        infos =  {agent: {"timesteps": self.timestep} for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        env_truncation = self.timestep >= self.env_config["max_episode_length"]

        truncations = {agent: env_truncation for agent in self.agents}

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
                    self.manager.update_rewards(1*self.env_config["gamma"]**self.timestep)
            else:
                if not all(terminations.values()):
                    self.agents = self.possible_agents[:]
                    for at in terminations:
                        terminations[at] = False
    
        rewards = rm_rewards
        if self.test:
            if all(terminations.values()) and any([i > 0 for i in rewards.values()]):
                for ar in rewards:
                    rewards[ar] = 1
            elif not all(terminations.values()):
                for ar in rewards:
                    rewards[ar] = 0
        if self.video and (env_truncation or all(terminations.values())):
            self.send_animation()

        return obs, rewards, terminations, truncations, infos

    def render(self):
        self.viz.render(self.mdp.agent_view_size, self.curr_state, highlight=False)

    def send_animation(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        path_dir = f"{self.log_dir}"
        os.makedirs(path_dir, exist_ok=True)
        self.viz.animate(self.traj_mdp_states, agent_view_size=5, filename=f"{path_dir}/viz.gif")
        log_dict = {}
        log_dict[f"viz"] = wandb.Video(f"{path_dir}/viz.gif", fps=4, format="gif")

        wandb.log(log_dict)