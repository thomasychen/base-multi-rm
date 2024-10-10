import random
import numpy as np
from enum import Enum
import copy

"""
Enum with the actions that the agent can execute
"""
class Actions(Enum):
    up    = 0 # move up
    right = 1 # move right
    down  = 2 # move down
    left  = 3 # move left
    none  = 4 # none 

class EasyButtonsEnv:

    def __init__(self, env_config):
        """
        Initialize environment.

        Parameters
        ----------
        rm_file : string
            File path leading to the text file containing the reward machine
            encoding this environment's reward function.
        agent_id : int
            Index {0,1} indicating which agent
        env_settings : dict
            Dictionary of environment settings
        """

        env_settings = copy.deepcopy(env_config)
        env_settings['Nr'] = 10
        env_settings['Nc'] = 10
        env_settings['initial_states'] = [0, 5, 8]
        env_settings['walls'] = [(0, 3), (1, 3), (2, 3), (3,3), (4,3), (5,3), (6,3), (7,3),
                                    (7,4), (7,5), (7,6), (7,7), (7,8), (7,9),
                                    (0,7), (1,7), (2,7), (3,7), (4,7) ]
        env_settings['goal_location'] = (8,9)
        env_settings['yellow_button'] = (0,2)
        env_settings['green_button'] = (5,6)
        env_settings['red_button'] = (6,9)
        env_settings['yellow_tiles'] = [(2,4), (2,5), (2,6), (3,4), (3,5), (3,6)]
        env_settings['green_tiles'] = [(2,8), (2,9), (3,8), (3,9)]
        env_settings['red_tiles'] = [(8,5), (8,6), (8,7), (8,8), (9,5), (9,6), (9,7), (9,8)]
        env_settings['p'] = 0.95
        self.env_settings = env_settings
        self.p = env_settings["p"]


        self._load_map()

        # self.u = self.reward_machine.get_initial_state()
        # self.last_action = -1 # Initialize last action to garbage value

    def reset(self, decomp_idx):
        EasyButtonsEnv.red_pressed = False
        EasyButtonsEnv.yellow_pressed = False
        EasyButtonsEnv.green_pressed = False
        EasyButtonsEnv.who_at_red = 0

        rm_state_array = copy.deepcopy(self.env_settings["initial_rm_states"]) if np.array(self.env_settings["initial_rm_states"]).ndim == 2 else [copy.deepcopy(self.env_settings["initial_rm_states"])]

        EasyButtonsEnv.u = {i+1:rm_state_array[decomp_idx][i] for i in range(len(self.env_settings["initial_states"]))}
        # print(self.u)

    def _load_map(self):
        """
        Initialize the environment.
        """
        self.Nr = self.env_settings['Nr']
        self.Nc = self.env_settings['Nc']

        self.objects = {}
        self.objects[self.env_settings['goal_location']] = "g" # goal location
        self.objects[self.env_settings['yellow_button']] = 'yb'
        self.objects[self.env_settings['green_button']] = 'gb'
        self.objects[self.env_settings['red_button']] = 'rb'
        self.yellow_tiles = self.env_settings['yellow_tiles']
        self.green_tiles = self.env_settings['green_tiles']
        self.red_tiles = self.env_settings['red_tiles']

        self.num_states = self.Nr * self.Nc

        self.actions = [Actions.up.value, Actions.right.value, Actions.left.value, Actions.down.value, Actions.none.value]
        
        # Define forbidden transitions corresponding to map edges
        self.forbidden_transitions = set()
        
        wall_locations = self.env_settings['walls']

        for row in range(self.Nr):
            self.forbidden_transitions.add((row, 0, Actions.left)) # If in left-most column, can't move left.
            self.forbidden_transitions.add((row, self.Nc - 1, Actions.right)) # If in right-most column, can't move right.
        for col in range(self.Nc):
            self.forbidden_transitions.add((0, col, Actions.up)) # If in top row, can't move up
            self.forbidden_transitions.add((self.Nr - 1, col, Actions.down)) # If in bottom row, can't move down

        # Restrict agent from having the option of moving "into" a wall
        for i in range(len(wall_locations)):
            (row, col) = wall_locations[i]
            self.forbidden_transitions.add((row, col + 1, Actions.left))
            self.forbidden_transitions.add((row, col-1, Actions.right))
            self.forbidden_transitions.add((row+1, col, Actions.up))
            self.forbidden_transitions.add((row-1, col, Actions.down))

    def environment_step(self, s, a, agent_id):
        """
        Execute action a from state s.

        Parameters
        ----------
        s : int
            Index representing the current environment state.
        a : int
            Index representing the action being taken.

        Outputs
        -------
        r : float
            Reward achieved by taking action a from state s.
        l : list
            List of events occuring at this step.
        s_next : int
            Index of next state.
        """
        s_next, _ = self.get_next_state(s,a, agent_id)

        return s_next
    
    # def get_mdp_label(self, s, s_next, u):
    #     """
    #     Return the label of the next environment state and current RM state.
    #     """
    #     row, col = self.get_state_description(s_next)

    #     l = []

    #     thresh = 0.3 #0.3

    #     if self.agent_id == 1:
    #         if u == 0:
    #             if (row, col) == self.env_settings['yellow_button']:
    #                 l.append('by')
    #         if u == 1:
    #             if np.random.random() <= thresh:
    #                 l.append('br')
    #         if u == 2:
    #             if (row, col) == self.env_settings['goal_location']:
    #                 l.append('g')
    #     elif self.agent_id == 2:
    #         if u == 0:
    #             if np.random.random() <= thresh:
    #                 l.append('by')
    #         if u == 1 and (row,col) == self.env_settings['green_button']:
    #             l.append('bg')
    #         if u == 2 and (row,col) == self.env_settings['red_button']:
    #             l.append('a2br')
    #         if u == 3: 
    #             if not((row,col) == self.env_settings['red_button']):
    #                 l.append('a2lr')
    #             elif np.random.random() <= thresh:
    #                 l.append('br')
    #     elif self.agent_id == 3:
    #         if u == 0:
    #             if np.random.random() <= thresh:
    #                 l.append('bg')
    #         if u == 1 and (row,col) == self.env_settings['red_button']:
    #             l.append('a3br')
    #         if u == 2: 
    #             if not((row,col) == self.env_settings['red_button']):
    #                 l.append('a3lr')
    #             elif np.random.random() <= thresh:
    #                 l.append('br')

    #     return l

    def get_next_state(self, s, a, agent_id):
        """
        Get the next state in the environment given action a is taken from state s.
        Update the last action that was truly taken due to MDP slip.

        Parameters
        ----------
        s : int
            Index of the current state.
        a : int
            Action to be taken from state s.

        Outputs
        -------
        s_next : int
            Index of the next state.
        last_action :int
            Last action taken by agent due to slip proability.
        """
        slip_p = [self.p, (1-self.p)/2, (1-self.p)/2]
        check = random.random()

        row, col = self.get_state_description(s)

        # up    = 0
        # right = 1 
        # down  = 2 
        # left  = 3 

        if (check<=slip_p[0]) or (a == Actions.none.value):
            a_ = a

        elif (check>slip_p[0]) & (check<=(slip_p[0]+slip_p[1])):
            if a == 0: 
                a_ = 3
            elif a == 2: 
                a_ = 1
            elif a == 3: 
                a_ = 2
            elif a == 1: 
                a_ = 0

        else:
            if a == 0: 
                a_ = 1
            elif a == 2: 
                a_ = 3
            elif a == 3: 
                a_ = 0
            elif a == 1: 
                a_ = 2

        action_ = Actions(a_)
        if (row, col, action_) not in self.forbidden_transitions:
            if action_ == Actions.up:
                row -= 1
            if action_ == Actions.down:
                row += 1
            if action_ == Actions.left:
                col -= 1
            if action_ == Actions.right:
                col += 1

        s_next = self.get_state_from_description(row, col)

        # If the appropriate button hasn't yet been pressed, don't allow the agent into the colored region
        if agent_id == 1:
            if self.u[agent_id] == 1:
                if (row, col) in self.red_tiles:
                    s_next = s
            if self.u[agent_id] == 2:
                if (row, col) in self.red_tiles:
                    s_next = s
        if agent_id == 2:
            if self.u[agent_id] == 5:
                if (row, col) in self.yellow_tiles:
                    s_next = s
        if agent_id == 3:
            if self.u[agent_id] == 12:
                if (row, col) in self.green_tiles:
                    s_next = s

        last_action = a_
        return s_next, last_action

    def get_state_from_description(self, row, col):
        """
        Given a (row, column) index description of gridworld location, return
        index of corresponding state.

        Parameters
        ----------
        row : int
            Index corresponding to the row location of the state in the gridworld.
        col : int
            Index corresponding to the column location of the state in the gridworld.
        
        Outputs
        -------
        s : int
            The index of the gridworld state corresponding to location (row, col).
        """
        return self.Nc * row + col

    def get_state_description(self, s):
        """
        Return the row and column indeces of state s in the gridworld.

        Parameters
        ----------
        s : int
            Index of the gridworld state.

        Outputs
        -------
        row : int
            The row index of state s in the gridworld.
        col : int
            The column index of state s in the gridworld.
        """
        row = np.floor_divide(s, self.Nr)
        col = np.mod(s, self.Nc)

        return (row, col)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.actions

    # def get_last_action(self):
    #     """
    #     Returns agent's last action
    #     """
    #     return self.last_action

    # def get_initial_state(self):
    #     """
    #     Outputs
    #     -------
    #     s_i : int
    #         Index of agent's initial state.
    #     """
    #     return self.s_i