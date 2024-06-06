from mdps.challenge_buttons_mdp import ChallengeButtonsEnv

class ChallengeButtonsLabeled(ChallengeButtonsEnv):
    def get_mdp_label(self, s_next, *args):
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

    
    