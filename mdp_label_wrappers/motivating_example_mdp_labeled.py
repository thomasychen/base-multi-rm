from mdps.motivating_example import MotivatingExampleEnv
import numpy as np
from mdp_label_wrappers.generic_mdp_labeled import MDP_Labeler

class MotivatingExampleLabeled(MotivatingExampleLabeled, MDP_Labeler):

    # def environment_step(self, s, a, agent_id):
    #     s_next, last_action = self.get_next_state(s,a, agent_id)
    #     # self.last_action = last_action

    #     # l = self.get_mdp_label(s_next, agent_id, self.u[agent_id])
    #     # r = 0

    #     # for e in l:
    #     #     # Get the new reward machine state and the reward of this step
    #     #     u2 = self.reward_machine.get_next_state(self.u[agent_id], e)
    #     #     r = r + self.reward_machine.get_reward(self.u[agent_id], u2)
    #     #     # Update the reward machine state
    #     #     self.u[agent_id] = u2

    #     return s_next


    def get_mdp_label(self, s_next, agent_id=-1, u=-1, test = False, monolithic = False):
        """
        Return the label of the next environment state and current RM state.
        """
        row, col = self.get_state_description(s_next)

        l = []

        thresh = 0.3 #0.3

        if monolithic:
            if u == 1 or u == 2 or u == 3:
            # Now check if agents are on buttons
                if agent_id == 1 and  (row,col) == self.env_settings['yellow_button']:
                    l.append('by')
            if u == 4 or u == 11 or u == 18:
                if agent_id == 2 and (row, col) == self.env_settings['green_button']:
                    l.append('bg')
            if u == 5 or u == 12 or u == 19:
                if agent_id == 2 and (row, col) == self.env_settings['red_button']:
                    l.append('a2br')
                if agent_id == 3 and (row, col) == self.env_settings['red_button']:
                    l.append('a3br')
            if u == 6 or u == 13 or u == 20:
                if agent_id == 2 and not ((row, col) == self.env_settings['red_button']):
                    l.append('a2lr')
                if agent_id == 3 and (row, col) == self.env_settings['red_button']:
                    l.append('a3br')
            if u == 7 or u == 14 or u == 21:
                if agent_id == 2 and (row, col) == self.env_settings['red_button']:
                    l.append('a2br')
                if agent_id == 3 and not ((row, col) == self.env_settings['red_button']):
                    l.append('a3lr')
            if u == 8 or u == 15 or u == 22:
                    l.append('br')
            if u == 9:
                # Check if agent 1 has reached the goal
                if agent_id == 1 and (row, col) == self.env_settings['goal_location']:
                    l.append('g')
        else:
            if agent_id == 1:
                if u == 1:
                    if (row, col) == self.env_settings['yellow_button']:
                        l.append('by')
                if u == 2:
                    if np.random.random() <= thresh and not test:
                        l.append('br')
                if u == 3:
                    if (row, col) == self.env_settings['goal_location']:
                        l.append('g')
            elif agent_id == 2:
                if u == 5:
                    if np.random.random() <= thresh and not test:
                        l.append('by')
                if u == 6 and (row,col) == self.env_settings['green_button']:
                    l.append('bg')
                if (u == 7 or u == 8) and (row,col) == self.env_settings['red_button']:
                    l.append('a2br')
                if u == 9: 
                    if not((row,col) == self.env_settings['red_button']):
                        l.append('a2lr')
                    elif np.random.random() <= thresh and not test:
                        l.append('a3br')
            elif agent_id == 3:
                if u == 12:
                    if np.random.random() <= thresh and not test:
                        l.append('bg')
                if (u == 13 or u == 15) and (row,col) == self.env_settings['red_button']:
                    l.append('a3br')
                if u == 14: 
                    if not((row,col) == self.env_settings['red_button']):
                        l.append('a3lr')
                    elif np.random.random() <= thresh and not test:
                        l.append('a2br')

            if u == 16 or u == 10:
                l.append('br')
            
        return l