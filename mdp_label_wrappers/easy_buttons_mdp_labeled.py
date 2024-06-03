from mdps.easy_buttons_mdp import EasyButtonsEnv
import numpy as np

class EasyButtonsLabeled(EasyButtonsEnv):

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


        # thresh = 0.3 #0.3

        # rand = np.random.random()

        # if agent_id == 1:
        #     if not EasyButtonsEnv.yellow_pressed:
        #         if (row, col) == self.env_settings['yellow_button']:
        #             EasyButtonsEnv.yellow_pressed = True
        #             l.append('by')
        #     elif EasyButtonsEnv.yellow_pressed and not EasyButtonsEnv.red_pressed:
        #         if rand <= thresh and not test:
        #             EasyButtonsEnv.red_pressed = True
        #             EasyButtonsEnv.who_at_red = 2
        #             l.append('br')
        #     elif EasyButtonsEnv.yellow_pressed and EasyButtonsEnv.red_pressed:
        #         if (row, col) == self.env_settings['goal_location']:
        #             l.append('g')
        # elif agent_id == 2:
        #     if not EasyButtonsEnv.yellow_pressed:
        #         if rand <= thresh and not test:
        #             EasyButtonsEnv.yellow_pressed = True
        #             l.append('by')
        #     elif EasyButtonsEnv.yellow_pressed and (row,col) == self.env_settings['green_button']:
        #         EasyButtonsEnv.green_pressed = True
        #         l.append('bg')
        #     elif EasyButtonsEnv.yellow_pressed and EasyButtonsEnv.green_pressed and (row,col) == self.env_settings['red_button'] and not EasyButtonsEnv.who_at_red:
        #         EasyButtonsEnv.who_at_red = agent_id
        #         l.append('a2br')
        #     elif EasyButtonsEnv.yellow_pressed and EasyButtonsEnv.green_pressed and not EasyButtonsEnv.red_pressed and EasyButtonsEnv.who_at_red == agent_id : 
        #         if not((row,col) == self.env_settings['red_button']):
        #             EasyButtonsEnv.who_at_red = 0
        #             l.append('a2lr')
        #         elif rand <= thresh and not test:
        #             EasyButtonsEnv.red_pressed = True
        #             l.append('br')
        #     elif EasyButtonsEnv.yellow_pressed and EasyButtonsEnv.green_pressed and not EasyButtonsEnv.red_pressed and EasyButtonsEnv.who_at_red and EasyButtonsEnv.who_at_red != agent_id:
        #         if (row,col) == self.env_settings['red_button']:
        #             EasyButtonsEnv.red_pressed = True
        #             l.append('br')
        # elif agent_id == 3:
        #     if not EasyButtonsEnv.green_pressed:
        #         if rand <= thresh and not test:
        #             EasyButtonsEnv.green_pressed = True
        #             l.append('bg')
        #     elif EasyButtonsEnv.green_pressed and not EasyButtonsEnv.who_at_red and (row,col) == self.env_settings['red_button']:
        #         EasyButtonsEnv.who_at_red = agent_id
        #         l.append('a3br')
        #     elif EasyButtonsEnv.green_pressed and not EasyButtonsEnv.red_pressed and EasyButtonsEnv.who_at_red == agent_id: 
        #         if not((row,col) == self.env_settings['red_button']):
        #             EasyButtonsEnv.who_at_red = 0
        #             l.append('a3lr')
        #         elif rand <= thresh and not test:
        #             EasyButtonsEnv.red_pressed = True
        #             l.append('br')
        #     elif EasyButtonsEnv.yellow_pressed and EasyButtonsEnv.green_pressed and not EasyButtonsEnv.red_pressed and EasyButtonsEnv.who_at_red and EasyButtonsEnv.who_at_red != agent_id:
        #         if (row,col) == self.env_settings['red_button']:
        #             EasyButtonsEnv.red_pressed = True
        #             l.append('br')
        # print("random: ", rand <= thresh, " s_next: ", s_next, " labels: ", l, " agent_id: ", agent_id)
        # thresh = 0.3


        # if (row, col) == self.env_settings['yellow_button']:
        #     l.append('by')
        #     EasyButtonsEnv.yellow_pressed = True
        # elif (row, col) == self.env_settings['green_button']:
        #     l.append('bg')
        #     EasyButtonsEnv.green_pressed = True
        # elif (row, col) == self.env_settings['red_button']:
        #     if EasyButtonsEnv.red_pressed == 1:
        #         l.append('br') 
        #     else: 
        #         l.append(f'a{agent_id}br')
        #         EasyButtonsEnv.who_at_red = agent_id
        #     EasyButtonsEnv.red_pressed += 1
        # elif EasyButtonsEnv.red_pressed == 1 and EasyButtonsEnv.who_at_red == agent_id and (row, col) != self.env_settings['red_button']:
        #     EasyButtonsEnv.red_pressed -= 1
        #     EasyButtonsEnv.who_at_red = -1
        #     l.append(f'a{agent_id}lr')
        # elif not test:
        #     if not EasyButtonsEnv.yellow_pressed and agent_id != 1 and np.random.random() <= thresh:
        #         l.append('by')
        #         EasyButtonsEnv.yellow_pressed = True
        #     elif EasyButtonsEnv.red_pressed < 2 and EasyButtonsEnv.yellow_pressed and agent_id == 1 and np.random.random() <= thresh:
        #         EasyButtonsEnv.red_pressed = 2
        #         l.append('br')
        #     elif EasyButtonsEnv.yellow_pressed and agent_id == 2 and np.random.random() <= thresh:
        #         EasyButtonsEnv.green_pressed = True
        #         l.append('bg')
        #     elif EasyButtonsEnv.red_pressed == 1 and EasyButtonsEnv.who_at_red == agent_id and np.random.random() <= thresh:
        #         EasyButtonsEnv.red_pressed = 2
        #         l.append('br')

        return l