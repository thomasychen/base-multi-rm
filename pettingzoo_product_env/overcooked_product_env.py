from pettingzoo import ParallelEnv
from jaxmarl.environments.overcooked import Overcooked
import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from gymnasium.spaces import Box, Discrete
from mdp_label_wrappers.generic_mdp_labeled import MDP_Labeler
from reward_machines.sparse_reward_machine import SparseRewardMachine
import copy
from datetime import datetime
import wandb
import os

class OvercookedProductEnv(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, manager, labeled_mdp_class: MDP_Labeler, reward_machine: SparseRewardMachine, config, max_agents, test=False, is_monolithic=False, addl_mono_rm: SparseRewardMachine=None, render_mode=None, monolithic_weight=1.0, log_dir=None, video=False):
        self.possible_agents = ["agent_" + str(r) for r in range(2)]
        self.env_config = config

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode 
        self.labeled_mdp = labeled_mdp_class(config)
        self.mdp = self.labeled_mdp.jax_env
        self.states = []
        self.viz = OvercookedVisualizer()
        # self.eps_reward = {agent: 0 for agent in self.possible_agents}
        self.reset_key = None
        self.test = test
        self.addl_monolithic_rm = addl_mono_rm # Potentially give the monolithic here so everyone know's global states (for potentially dependent dynamics)
        self.monolithic_weight = monolithic_weight
        # self.rm_states = []
        OvercookedProductEnv.manager = manager
        self.local_manager = None
        self.reward_machine = reward_machine
        self.log_dir = log_dir
        self.video = video


        # self.key, self.key_r, self.key_a = jax.random.split(key, 3)
    

    def observation_space(self, agent):
        # flattened_shape = [int(np.prod(self.labeled_mdp.obs_shape) + len(self.reward_machine.get_states()))]
        if self.addl_monolithic_rm is None:
            flattened_shape = [self.labeled_mdp.obs_shape + self.reward_machine.get_one_hot_size(len(self.possible_agents))]
        else:
            flattened_shape = [self.labeled_mdp.obs_shape + self.reward_machine.get_one_hot_size(len(self.possible_agents)) + self.addl_monolithic_rm.get_one_hot_size(len(self.possible_agents))]
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(0, 255, flattened_shape)

    def action_space(self, agent):
        # return self.mdp.action_space()
        return Discrete(len(self.mdp.action_set))
    
    def reset(self, seed=None, options=None):

        if not self.local_manager:
            self.local_manager = OvercookedProductEnv.manager

        self.agents = self.possible_agents[:]
        self.timestep = 0

        print("RESETTING")

        self.states = []
        if self.reset_key is None:
            self.reset_key = jax.random.PRNGKey(0)
        self.reset_key, key_r, key_a = jax.random.split(self.reset_key, 3)
        jax_observations, state = self.mdp.reset(key_r)
        jax_observations = self.labeled_mdp.trim_observation(jax_observations)
        
        # Init variables
        # need to change this and get 
        rm_array = copy.deepcopy(self.env_config["initial_rm_states"]) if np.array(self.env_config["initial_rm_states"]).ndim == 2 else [copy.deepcopy(self.env_config["initial_rm_states"])]
        
        # def one_hot_encode(value, num_classes):
        #     """One-hot encode a single value."""
        #     one_hot = np.zeros(num_classes)
        #     one_hot[value] = 1
        #     return one_hot
        
        # n = len(self.reward_machine.get_states())
        if self.addl_monolithic_rm is not None:
            self.monolithic_rm_state = self.addl_monolithic_rm.get_initial_state()

        rm_state_array = [[self.reward_machine.get_one_hot_encoded_state(state, len(self.possible_agents), idx) for idx, state in enumerate(init_states)] for init_states in rm_array]
        
        # import pdb; pdb.set_trace()

        # rm_state_array = [np.zeros(n) for _ in range(len(self.agents))]
        # for i in range(len(self.agents)):
        #     import pdb; pdb.set_trace()
        #     rm_state_array[i][rm_array[i]] = 1

        mdp_state_array = [jax_observations[agent].flatten() for agent in self.agents]

        decomp_idx = self.local_manager.get_rm_assignments(mdp_state_array, rm_state_array, test=self.test)

        # self.mdp_states = {self.agents[i]:mdp_state_array[i] for i in range(len(self.agents))}
        self.rm_states = {self.agents[i]: rm_array[decomp_idx][i] for i in range(len(self.agents))}

        # print(self.rm_states, self.mdp_states)
        # print(self.rm_states)
        # print("MANAGER LOGS")
        # print(rm_assignments, decomp_idx)
        # print(self.rm_states)
        
        print("reset step", state.time)
        self.curr_state = state
        self.states.append(self.curr_state)
        infos = {agent: {} for agent in self.agents}

        observations = {i: self.flatten_and_add_rm(jax_observations[i], self.rm_states[i], idx) for idx, i in enumerate(self.agents)}
        # import pdb; pdb.set_trace()
        # observations = {self.agents[i]: np.concatenate((mdp_state_array[i], self.rm_states[self.agents[i]])) for i in range(len(self.agents))}
        
        # observations = {i: jnp.transpose(jax_observations[i], (1,0,2)) for i in jax_observations}

        # observations = {i: np.concatenate((flattened_obs, ohe_rm)) for i, flattened_obs, ohe_rm in zip(self.agents, mdp_state_array, rm_state_array)}
        # observations = {i: self.flatten_and_add_rm(observations[i], self.rm_states[i]) for i in observations}
    
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
        # {
        #     agent_1: 0, 
        #     agent_2: 1,
        # }
        #  {
        #      agent_1: np.array(0), 
             
        #  }
        false_actions = {agent: 4 for agent in self.possible_agents}
        for agent, action in actions.items():
            false_actions[agent] = action
        # print("CURRENT STEP", self.timestep)
        self.reset_key, key_a0, key_a1, key_s = jax.random.split(self.reset_key, 4)

        jax_obs, state, jax_rewards, jax_dones, jax_infos = self.mdp.step(key_s, self.curr_state, false_actions)
        # import pdb; pdb.set_trace();

        jax_obs, labels = self.labeled_mdp.get_mdp_label(state, jax_rewards)
        # if "o2" in labels: 
        #     import pdb; pdb.set_trace()
        #     print(labels, self.rm_states)
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

            # if 'o3' in labels:
            #     import pdb; pdb.set_trace()
            # if pseudo_reward == 1:
            #     import pdb; pdb.set_trace()
        if self.addl_monolithic_rm is not None:
            for agent in self.possible_agents:
                rm_rewards[agent] += mono_rm_reward
        
        #TODO: commented this out. we keep all agents going until the end even if they've accomplished their goal
        #terminations = {i: self.reward_machine.is_terminal_state(self.rm_states[i]) for i in self.agents} 
        terminations = {i: False for i in self.agents}
        #TODO: add termination if the monolithic finishes
        if self.addl_monolithic_rm is not None:
            if self.addl_monolithic_rm.is_terminal_state(self.monolithic_rm_state):
                terminations = {i: True for i in self.agents}
        else:
            if all([self.reward_machine.is_terminal_state(self.rm_states[i]) for i in self.agents]):
                terminations = {i: True for i in self.agents}
        
        # print(terminations) if self.reward_machine.is_terminal_state(self.rm_states['agent_0']) else None
        
        # new_agents = []
        # for i in self.agents:
        #     if not terminations[i]:
        #         new_agents.append(i)
        # self.agents = new_agents
        # if terminations and not self.agents and not self.test:
        #     print("completed?", terminations, new_agents, self.timestep)
        #     self.manager.update_rewards(1*(self.env_config['gamma']**self.timestep))


        
        # print("step", state.time)
        self.curr_state = state
        self.states.append(state)
        # obs = {i: jnp.transpose(jax_obs[i], (1, 0, 2)) for i in jax_obs}
        obs = {i: self.flatten_and_add_rm(jax_obs[i], self.rm_states[i], idx) for idx, i in enumerate(self.agents)}
        # rewards = {i: float(jax_rewards[i]) for i in jax_rewards}
        # for agent, rew in rewards.items():
        #     self.eps_reward[agent] += rew
        # dones = {i: bool(jax_dones[i]) for i in jax_dones}
        # if any([i > 0 for i in rewards.values()]): 
        #     dones = {i: True for i in jax_dones}
        #     import pdb; pdb.set_trace();
        self.timestep += 1
        infos =  {agent: {"timesteps": self.timestep} for agent in self.agents}

        if self.render_mode == "human":
            self.render()
        
        # print(dones)
        # if dones["__all__"]:
        #     self.agents = []
            # import pdb; pdb.set_trace();

        # If a user passes in actions with no agents, then just return empty observations, etc.
        # if not actions:
        #     self.agents = []
        #     return {}, {}, {}, {}, {}

        # # rewards for all agents are placed in the rewards dictionary to be returned
        # rewards = {}
        # rewards[self.agents[0]], rewards[self.agents[1]] = REWARD_MAP[
        #     (actions[self.agents[0]], actions[self.agents[1]])
        # ]

        # terminations = {agent: False for agent in self.agents}

        # self.num_moves += 1
        # env_truncation = self.num_moves >= NUM_ITERS
        # truncations = {agent: env_truncation for agent in self.agents}

        # # current observation is just the other player's most recent action
        # observations = {
        #     self.agents[i]: int(actions[self.agents[1 - i]])
        #     for i in range(len(self.agents))
        # }
        # self.state = observations

        # # typically there won't be any information in the infos, but there must
        # # still be an entry for each agent
        # infos = {agent: {} for agent in self.agents}
        # env_truncation = (state.time == 0)
        # truncations = {agent: False for agent in self.agents}
        # truncations["__all__"] = False
        # terminations = {i: False for i in actions}
        # terminations["__all__"] = False

        # if state.time == 0:
        #     # print("episode reward: ", self.eps_reward)
        #     self.eps_reward = {agent: 0 for agent in self.possible_agents}
            # truncations = {i: True for i in actions}
            # truncations["__all__"] = True
            # jax_infos = {}

        # print("infos:", jax_infos)
        
        # if sum(jax_infos["shaped_reward"].values()) > 0:
        #     import pdb; pdb.set_trace();
        # if self.timestep >= self.env_config["max_episode_length"]:
        #     import pdb; pdb.set_trace()
        # rewards = {i: float(jax_infos["shaped_reward"][i]) for i in jax_infos["shaped_reward"]} #if not self.test else rewards

        env_truncation = self.timestep >= self.env_config["max_episode_length"]


        truncations = {agent: env_truncation for agent in self.agents}



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
                    self.manager.update_rewards(1*self.env_config["gamma"]**self.timestep)
            else:
                if not all(terminations.values()):
                    self.agents = self.possible_agents[:]
                    for at in terminations:
                        terminations[at] = False
    
            # if not self.agents:
            #     print("FINISHED REWARDS", rewards)
        # print(self.rm_states)
        rewards = rm_rewards
        if self.test:
            # print("TESTING", self.timestep, truncations, self.agents, terminations)
            if all(terminations.values()) and any([i > 0 for i in rewards.values()]):
                for ar in rewards:
                    rewards[ar] = 1
            elif not all(terminations.values()):
                for ar in rewards:
                    rewards[ar] = 0
        # for visualizing eval trajectories with gifs
        if self.video and (env_truncation or all(terminations.values())):
            self.send_animation()


        # if env_truncation:
        #     self.agents = []
        # rewards = rm_rewards
        # if self.test:
        #     # print("TESTING", rewards)
        #     print("TESTING", self.timestep, truncations, self.agents, terminations)
        #     if all(rm_rewards.values()):
        #         print("TESTING ALL DONE", rewards, terminations)
  
        #     if not all(rm_rewards.values()):
        #          rewards = {i:0 for i in self.possible_agents}
        #          self.agents = self.possible_agents[:]
        #          terminations = {i: False for i in self.possible_agents}
        #TODO: return rm_rewards instead of rewards
        return obs, rewards, terminations, truncations, infos

    def render(self):
        self.viz.render(self.mdp.agent_view_size, self.curr_state, highlight=False)

    def send_animation(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # viz.animate(seq, agent_view_size=5, filename=f"{wandb.run.dir}/{filename}")
        # log_dict[filename] = wandb.Video(f"{wandb.run.dir}/{filename}", fps=4, format="gif")

        # wandb.log(log_dict)
        path_dir = f"{self.log_dir}"
        os.makedirs(path_dir, exist_ok=True)
        self.viz.animate(self.states, agent_view_size=5, filename=f"{path_dir}/viz.gif")
        log_dict = {}
        log_dict[f"viz"] = wandb.Video(f"{path_dir}/viz.gif", fps=4, format="gif")

        wandb.log(log_dict)

    def flatten_and_add_rm(self, obs, rm_state, agent_idx):
        # import pdb; pdb.set_trace();

        # n = len(self.reward_machine.get_states())
        # # Flatten the 3D observation array
        flattened_obs = obs.flatten()

        # # Create an n-length array of zeros
        # n = len(self.reward_machine.get_states())
        # rm_array = np.zeros(n)

        # # Set the rm_state index to 1
        # rm_array[rm_state] = 1
        rm_ohe = self.reward_machine.get_one_hot_encoded_state(rm_state, len(self.possible_agents), agent_idx)
        if self.addl_monolithic_rm is not None:
            mono_ohe = self.addl_monolithic_rm.get_one_hot_encoded_state(self.monolithic_rm_state, len(self.possible_agents), agent_idx)
            result = np.concatenate((flattened_obs, rm_ohe, mono_ohe))
        else:
            # Concatenate the flattened observation and the rm_array
            result = np.concatenate((flattened_obs, rm_ohe))

        return result