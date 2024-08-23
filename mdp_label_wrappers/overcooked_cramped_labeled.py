from mdp_label_wrappers.generic_mdp_labeled import MDP_Labeler
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl import make

class OvercookedCrampedLabeled(MDP_Labeler):
    def __init__(self, run_config):
        self.layout = overcooked_layouts["cramped_room"]
        self.jax_env = make('overcooked', layout=self.layout, max_steps=run_config["max_episode_length"])
        self.render_mode = run_config["render_mode"]

    def get_mdp_label(self, s_next, *args):
        """
        TODO: IMPLEMENT
        Return the label of the next environment state and current RM state.
        """
        return None

    