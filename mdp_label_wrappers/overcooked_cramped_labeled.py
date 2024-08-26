from mdp_label_wrappers.generic_mdp_labeled import MDP_Labeler
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl import make

class OvercookedCrampedLabeled(MDP_Labeler):
    def __init__(self, run_config):
        self.layout = overcooked_layouts["cramped_room"]
        self.jax_env = make('overcooked', layout=self.layout, max_steps=run_config["max_episode_length"])
        self.render_mode = run_config["render_mode"]

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
        obs = self.jax_env.get_obs(s_next)
        obs = obs["agent_0"]
        obs = obs.transpose(2, 0, 1)

        # For onions
        onions = self.num_onions(obs)
        if onions:  
            l.append(onions)
        # For soup plated
        soup = self.has_soup(obs)
        if soup:
            import pdb; pdb.set_trace()
            l.append(soup)
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
        return l


    

    