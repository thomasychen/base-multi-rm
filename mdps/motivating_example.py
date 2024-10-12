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

class MotivatingExampleEnv:

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
        env_settings['Nr'] = 7
        env_settings['Nc'] = 12
        env_settings['initial_states'] = [13, 77, 16]
        env_settings['walls'] = [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
                                 (4, 6),
                                 (3, 6),(3, 7),(3, 8),(3, 9),(3, 10),
                                 (2, 6),
                                 (1, 6)]
        env_settings['hq_location'] = (2,3)
        env_settings['yellow_button'] = (6,8)
        env_settings['green_button'] = (6,0)
        env_settings['red_button'] = (0,8)
        yellow_tiles_x = range(0,7)
        yellow_tiles_y = range(7,12)
        env_settings['yellow_tiles'] = [(x,y) for x in yellow_tiles_x for y in yellow_tiles_y]
        env_settings['p'] = 0.95
        self.env_settings = env_settings
        self.p = env_settings["p"]
        self.in_hazard = False


        self._load_map()

        # self.u = self.reward_machine.get_initial_state()
        # self.last_action = -1 # Initialize last action to garbage value

    def reset(self, rm_assignments, decomp_idx):
        ...

    def _load_map(self):
        """
        Initialize the environment.
        """
        self.Nr = self.env_settings['Nr']
        self.Nc = self.env_settings['Nc']

        self.objects = {}
        self.objects[self.env_settings['hq_location']] = "hq" # goal location
        self.objects[self.env_settings['yellow_button']] = 'yb'
        self.objects[self.env_settings['green_button']] = 'gb'
        self.objects[self.env_settings['red_button']] = 'rb'
        self.yellow_tiles = self.env_settings['yellow_tiles']

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

        in_yellow = (row, col) in self.yellow_tiles

        # check if agent is in the yellow region
        if self.in_hazard is None:
            if in_yellow:
                self.in_hazard = agent_id
        elif self.in_hazard == agent_id:
            if not in_yellow:
                self.in_hazard = None
            
        # If there's already an agent in the yellow region, don't allow the agent into the yellow region
        if self.in_hazard is not None and self.in_hazard != agent_id:
            if in_yellow:
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

    def show(self, s):
        """
        Create a visual representation of the current state of the gridworld.

        Parameters
        ----------
        s : int
            Index of the current state
        """
        display = np.zeros((self.Nr, self.Nc))
        
        # Display the locations of the walls
        for loc in self.yellow_tiles:
            display[loc] = 2

        for loc in self.env_settings['walls']:
            display[loc] = -1

        display[self.env_settings['red_button']] = 7
        display[self.env_settings['green_button']] = 8
        display[self.env_settings['yellow_button']] = 9
        display[self.env_settings['hq_location']] = 6

        # Display the location of the agent in the world
        # row, col = self.get_state_description(s)
        # display[row,col] = self.agent_id

        print(display)

if __name__=='__main__':
    env = MotivatingExampleEnv({})
    env.show(0)