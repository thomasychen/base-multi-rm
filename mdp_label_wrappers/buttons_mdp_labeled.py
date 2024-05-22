from mdps.hard_buttons_mdp import HardButtonsEnv

class HardButtonsLabeled(HardButtonsEnv):
    def get_mdp_label(self, s_next):
        """
        Return the label of the next environment state and current RM state.
        """
        row, col = self.get_state_description(s_next)

        l = []

        if (row, col) == self.env_settings['yellow_button']:
            l.append('by')
        if (row, col) == self.env_settings['green_button']:
            l.append('bg')
        if (row, col) == self.env_settings['red_button']:
            l.append('br')

        return l

    
    