from mdps.motivating_example_mdp import MotivatingButtonsEnv
import numpy as np
from mdp_label_wrappers.generic_mdp_labeled import MDP_Labeler

class MotivatingExampleLabeled(MotivatingButtonsEnv, MDP_Labeler):

    def get_mdp_label(self, s_next, agent_id=-1, u=-1, test = False, monolithic = False):
        """
        Return the label of the next environment state and current RM state.
        """
        row, col = self.get_state_description(s_next)

        l = []

        thresh = 0.3 #0.3

        # ## append real labels
        if (row, col) == self.env_settings["yellow_button"]:
           l.append('y')
        elif (row, col) == self.env_settings["green_button"]:
            l.append("g")
        elif (row, col) == self.env_settings["red_button"]:
            l.append("r")
        elif (row, col) == self.env_settings["hq_location"] and agent_id == 1:
            MotivatingButtonsEnv.a1hq = True
            l.append("a1hq")
        elif (row, col) == self.env_settings["hq_location"] and agent_id == 2:
            MotivatingButtonsEnv.a2hq = True
            l.append("a2hq")
        elif (row, col) == self.env_settings["hq_location"] and agent_id == 3:
            MotivatingButtonsEnv.a3hq = True
            l.append("a3hq")
        elif MotivatingButtonsEnv.a1hq and not MotivatingButtonsEnv.signal and agent_id == 1 and (row, col) != self.env_settings["hq_location"]:
            MotivatingButtonsEnv.a1hq = False
            l.append("!a1hq")
        elif MotivatingButtonsEnv.a2hq and not MotivatingButtonsEnv.signal and agent_id == 2 and (row, col) != self.env_settings["hq_location"]:
            MotivatingButtonsEnv.a2hq = False
            l.append("!a2hq")
        elif MotivatingButtonsEnv.a3hq and not MotivatingButtonsEnv.signal and agent_id == 3 and (row, col) != self.env_settings["hq_location"]:
            MotivatingButtonsEnv.a3hq = False
            l.append("!a3hq")
        
        if (MotivatingButtonsEnv.a1hq and MotivatingButtonsEnv.a2hq) or (MotivatingButtonsEnv.a2hq and MotivatingButtonsEnv.a3hq) or (MotivatingButtonsEnv.a1hq and MotivatingButtonsEnv.a3hq):
            MotivatingButtonsEnv.signal = True
            l.append("sig")

        return l