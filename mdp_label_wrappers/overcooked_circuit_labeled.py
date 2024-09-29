from mdp_label_wrappers.generic_mdp_labeled import MDP_Labeler
from jaxmarl.environments.overcooked import overcooked_layouts, layout_grid_to_dict
from jaxmarl import make
import numpy as np

class OvercookedCircuitLabeled(MDP_Labeler):
    def __init__(self, run_config):
        
        counter_circuit_grid = """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
"""
        
        self.layout = layout_grid_to_dict(counter_circuit_grid)
        self.jax_env = make('overcooked', layout=self.layout, max_steps=run_config["max_episode_length"])
        print(run_config)
        self.render_mode = run_config["render_mode"]

        # Because we removed 3 layers
        self.obs_shape = list(self.jax_env.observation_space().shape)
        self.obs_shape[2] -= 3
        self.obs_shape = np.product(self.obs_shape)

    def any_elem(self, matrix, num, ignore_i=-1, ignore_j=-1):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == num and not (i == ignore_i and j == ignore_j):
                    return True
        return False
    
    def any_elem_nonzero(self, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] != 0:
                    return True
        return False

    def num_onions(self, obs):
        if self.any_elem(obs[16], 1):
            return "o1"
        elif self.any_elem(obs[16], 2):
            return "o2"
        elif self.any_elem_nonzero(obs[20]) or obs[21][0][2] == 1:
            return "o3"
        else:
            return None
        
    def has_soup(self, obs):
        # 
        if self.any_elem(obs[21], 1, 0, 2):
            return "p"
        return None
        
    def get_mdp_label(self, s_next, reward, *args):
        """
        TODO: IMPLEMENT
        Return the label of the next environment state and current RM state.
        """
        l = []
        old_obs = self.jax_env.get_obs(s_next)
        # import pdb; pdb.set_trace()
        obs = old_obs["agent_0"]
        obs = obs.transpose(2, 0, 1)


        # For onions
        onions = self.num_onions(obs)
        if onions:  
            import pdb; pdb.set_trace()
            l.append(onions)
        # For soup plated
        soup = self.has_soup(obs)
        if soup:
            # import pdb; pdb.set_trace()
            l.append(soup)
        # # for moving agent out of way after placing onions to not obstruct other agent
        # import pdb; pdb.set_trace()
        # move0 = #agent 0 is at non-obstructing
        # if move0 and "move" in :
        #     l.append("move0")
        # move1 = #agent 1 is at non-obstructing
        # if move1:
        #     l.append("move1")
        # For dish done
        if reward["agent_0"] > 0:
            # import pdb; pdb.set_trace()
            l.append("d")

        # self.jax_env.get_obs(s_next)
        # o: 16
        # p: 21
        # d: reward > 0
        # return real_mdp_state, label
        # TODO Masking state + early terminate environment
        
        old_obs = self.trim_observation(old_obs)
        return old_obs, l

    def trim_observation(self, obs):
        def trim_obs(agent_obs):
            agent_obs = agent_obs.transpose(2, 0, 1)
            layers_to_keep = [i for i in range(agent_obs.shape[0]) if i not in [16, 20, 21]]
            agent_obs = agent_obs[layers_to_keep, :, :]
            agent_obs = agent_obs.transpose(1, 2, 0)
            return agent_obs
        obs["agent_0"] = trim_obs(obs["agent_0"])
        obs["agent_1"] = trim_obs(obs["agent_1"])
        return obs
    
    

    