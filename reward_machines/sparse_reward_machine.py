from dfa import DFA, dict2dfa, dfa2dict
from dfa_identify.active import find_dfa_decomposition
import numpy as np

class SparseRewardMachine:
    def __init__(self,file=None):
        # <U,u0,delta_u,delta_r>
        self.U = []       # list of machine states
        self.events = set() # set of events
        self.u0 = None    # initial state
        self.delta_u = {} # state-transition function
        self.delta_r = {} # reward-transition function
        self.T = set()    # set of terminal states (they are automatically detected)
        self.accepting = set() # set of accepting states
        self.max_subtask_size = 0  # to store size of the largest subgraph
        self.state_to_subtask = {}  # to store state -> subtask number
        self.subtask_start_states = {}  # to store the starting state of each subtask
        self.num_subtasks = 0  # total number of subtasks
        self.equivalence_class_name_dict = {} # dict {name: equivalence class} Used in projections only 
        self.state_to_subtask_idx = {}
        self.is_monolithic = False
        if file is not None:
            self._load_reward_machine(file)
        # One hot encoding setup for reward machine states and decomp idx
        self.find_max_subgraph_size_and_assign_subtasks()
        # print(self.state_to_subtask_idx)
        # print(self.state_to_subtask)
        
    def __repr__(self):
        s = "MACHINE:\n"
        s += "init: {}\n".format(self.u0)
        for trans_init_state in self.delta_u:
            for event in self.delta_u[trans_init_state]:
                trans_end_state = self.delta_u[trans_init_state][event]
                s += '({} ---({},{})--->{})\n'.format(trans_init_state,
                                                        event,
                                                        self.delta_r[trans_init_state][trans_end_state],
                                                        trans_end_state)
        if self.equivalence_class_name_dict != {}:
            s += 'State Names for Equivalence Classes: \n'
            s += str(self.equivalence_class_name_dict) + '\n'
        return s

    # Public methods -----------------------------------

    def load_rm_from_file(self, file):
        self._load_reward_machine(file)

    def get_initial_state(self):
        return self.u0

    def get_next_state(self, u1, event):
        if u1 in self.delta_u:
            if event in self.delta_u[u1]:
                return self.delta_u[u1][event]
        return u1

    def get_reward(self,u1,u2,s1=None,a=None,s2=None):
        # print(self.delta_r, u1, u2)
        if u1 in self.delta_r and u2 in self.delta_r[u1]:
            return self.delta_r[u1][u2]
        return 0 # This case occurs when the agent falls from the reward machine

    def get_rewards_and_next_states(self, s1, a, s2, event):
        rewards = []
        next_states = []
        for u1 in self.U:
            u2 = self.get_next_state(u1, event)
            rewards.append(self.get_reward(u1,u2,s1,a,s2))
            next_states.append(u2)
        return rewards, next_states

    def get_states(self):
        return self.U

    def is_terminal_state(self, u1):
        # print(self.T)
        return u1 in self.T

    def get_events(self):
        return self.events

    def is_event_available(self, u, event):
        is_event_available = False
        if u in self.delta_u:
            if event in self.delta_u[u]:
                is_event_available = True
        return is_event_available
    
    def get_origins(self, u_set):
        all_origins = set()

        for state, transitions in self.delta_u.items():
            for u in u_set:
                if u in transitions.values():
                    all_origins.add(state)

        return all_origins 

    def find_max_subgraph_size_and_assign_subtasks(self):
        """Find the largest subgraph connected to u0, store its size in self.max_subtask_size, 
        assign subtasks, and save the number of subtasks and the starting state of each subtask."""  # Track visited states
        largest_size = 0
        subtask_number = 0
        self.subtask_start_states = {}  # Reset subtask start states

        # Iterate through neighbors of u0, treating each as the root of a subgraph
        for event, next_state in self.delta_u.get(self.u0, {}).items():
            visited = set()
            # Record the starting state for this subtask
            self.subtask_start_states[subtask_number] = next_state
            # Calculate subgraph size and assign subtask
            subgraph_size = self._dfs_subgraph_size_and_assign(next_state, visited, subtask_number)
            visited = list(visited)
            for i in range(len(visited)):
                self.state_to_subtask_idx[visited[i]] = i
            largest_size = max(largest_size, subgraph_size)
            subtask_number += 1  # Move to the next subtask for the next unvisited neighbor

        self.max_subtask_size = largest_size
        self.num_subtasks = subtask_number  # Save the total number of subtasks
        return largest_size
    
    def get_one_hot_size(self, num_agents):
        if self.is_monolithic:
            return len(self.get_states())
        return (self.num_subtasks) + num_agents + self.max_subtask_size

    def get_one_hot_encoded_state(self, state, num_agents, agent_idx):
        """Returns 3 one-hot encoded arrays for the given state."""
        """given state number, need to know: which decomp, which subtask,
        first 2 things already done
          how deep in subtask it is."""
        if self.is_monolithic:
            onehot = np.zeros(len(self.get_states()), dtype=int)
            onehot[state] = 1
            return onehot

        if state not in self.state_to_subtask:
            raise ValueError("State not assigned to any subtask!")

        # Get the subtask number for the given state
        subtask_number = self.state_to_subtask[state]
        
        # One-hot encode which decomposition the state belongs to
        decomposition_onehot = np.zeros(self.num_subtasks, dtype=int)
        decomposition_onehot[subtask_number] = 1

        # One-hot encode which subtask this is
        agent_onehot = np.zeros(num_agents, dtype=int)
        agent_onehot[agent_idx] = 1

        # One-hot encode the state's position within the subtask
        # start_state = self.subtask_start_states[subtask_number]
        state_position = self.state_to_subtask_idx[state]
        state_position_onehot = np.zeros(self.max_subtask_size, dtype=int)
        state_position_onehot[state_position] = 1

        concatenated_onehot = np.concatenate((decomposition_onehot, agent_onehot, state_position_onehot))
        
        return concatenated_onehot
    # Private methods -----------------------------------

    def _dfs_subgraph_size_and_assign(self, state, visited, subtask_number):
        """Helper function to perform DFS, return the size of the subgraph, and assign the subtask number."""
        visited.add(state)
        self.state_to_subtask[state] = subtask_number  # Assign the state to its subtask

        size = 1  # Include the current node
        for event, next_state in self.delta_u.get(state, {}).items():
            if next_state not in visited:
                size += self._dfs_subgraph_size_and_assign(next_state, visited, subtask_number)
        return size

    def _load_reward_machine(self, file):
        """
        Example:
            0                  # initial state
            (0,0,'r1',0)
            (0,1,'r2',0)
            (0,2,'r',0)
            (1,1,'g1',0)
            (1,2,'g2',1)
            (2,2,'True',0)

            Format: (current state, next state, event, reward)
        """
        # Reading the file
        f = open(file)
        lines = [l.rstrip() for l in f]
        f.close()
        # setting the DFA
        self._load_reward_machine_from_str_lines(lines)
    
    def _load_reward_machine_from_str_lines(self, lines):
        self.u0 = eval(lines[0])
        # adding transitions
        for e in lines[1:]:
            self._add_transition(*eval(e))
            self.events.add(eval(e)[2]) # By convention, the event is in the spot indexed by 2
        # adding terminal states
        for u1 in self.U:
            if self._is_terminal(u1):
                self.T.add(u1)
        self.U = sorted(self.U)

    def calculate_reward(self, trace):
        total_reward = 0
        current_state = self.get_initial_state()

        for event in trace:
            next_state = self.get_next_state(current_state, event)
            reward = self.get_reward(current_state, next_state)
            total_reward += reward
            current_state = next_state
        return total_reward

    def _is_terminal(self, u1):
        # Check if reward is given for reaching the state in question
        for u0 in self.delta_r:
            if u1 in self.delta_r[u0]:
                if self.delta_r[u0][u1] == 1 or self.delta_r[u0][u1] == -1:
                    return True
        return False
            
    def _add_state(self, u_list):
        for u in u_list:
            if u not in self.U:
                self.U.append(u)

    def _add_transition(self, u1, u2, event, reward):
        # Adding machine state
        self._add_state([u1,u2])
        # Adding state-transition to delta_u
        if u1 not in self.delta_u:
            self.delta_u[u1] = {}
        if event not in self.delta_u[u1]:
            self.delta_u[u1][event] = u2
        else:
            print(u1, u2, event, reward)
            raise Exception('Trying to make rm transition function non-deterministic.')
            # self.delta_u[u1][u2].append(event)
        # Adding reward-transition to delta_r
        if u1 not in self.delta_r:
            self.delta_r[u1] = {}
        self.delta_r[u1][u2] = reward
        if reward > 0:
            self.accepting.add(u2)

    def add_transition_and_reward_only(self, u1, u2, event, reward):
        ''' 
        Similar to _add_transition, this function is public and called when you project and parallelize reward machines. 
        Differences between these two functions:
            add_transition expects:
                (u1, u2) to already be in the state space
                event to already be in the event space

        Inputs:
            u1: element of rm.U
            u2: element of rm.U
            event: string, element of rm.events
            reward: int or float (probably 0 or 1)
        '''
        # add transition 
        if u1 not in self.delta_u:
            self.delta_u[u1] = {}
        if event in self.delta_u[u1]:
            if self.delta_u[u1][event] != u2:
                raise Exception('Trying to make rm transition function non-deterministic.')
        if event not in self.delta_u[u1]:
            self.delta_u[u1][event] = u2

        # Adding reward-transition to delta_r
        if u1 not in self.delta_r:
            self.delta_r[u1] = {}
        self.delta_r[u1][u2] = reward
    
    def get_name_from_class(self, cl):
        '''
        This function finds the "name" of a state that corresponds to a 
        particular class. 

        k is a key in self.equivalence_class_name_dict 
        if (k, cl) is an item of self.equivalence_class_name_dict, return k
        Inputs: 
            cl: (set), element of an equivalence class, value in self.equivalence_class_name_dict
       
        '''
        dict = self.equivalence_class_name_dict
        name_full = [k for k in dict if dict[k]==cl] # should have len = 1

        if len(name_full) == 0:
            sr = 'class ' + str(cl) + 'was not in the equivalence_class_name_dict for this reward machine'
            raise Exception(sr)

        return name_full[0]

def rm_to_dfa(reward_machine):
    '''
    Convert a reward machine object to a DFA for usage in DFA decomposition.
    '''
    dfa_dict = {}
    for state in reward_machine.delta_u:
        if 'True' in reward_machine.delta_u[state]:
            transition_dict = {}
        else:
            transition_dict = reward_machine.delta_u[state]
        for ap in reward_machine.get_events():
            # add stuttering semantics
            if ap not in transition_dict and ap != 'True':
                transition_dict[ap] = state
        dfa_dict[state] = (state in reward_machine.accepting, transition_dict)
    return dict2dfa(dfa_dict, reward_machine.u0)

def dfa_to_rm(dfa_obj):
    '''
    Convert a DFA object to a reward machine object.
    '''
    dfa_dict = dfa2dict(dfa_obj)

    reward_machine = SparseRewardMachine()
    dfa_dict, initial_state = dfa_dict[0], dfa_dict[1]
    for state in dfa_dict:
        original_state = state
        reward_machine._add_state([state])
        reward_machine.delta_u[state] = {}
        reward_machine.delta_r[state] = {}
        for event in dfa_dict[original_state][1]:
            next_state = dfa_dict[original_state][1][event]
            reward_machine.delta_u[state][event] = next_state
            reward_machine.delta_r[state][next_state] = int(dfa_dict[next_state][0])
            if dfa_dict[original_state][0]:
                reward_machine.T.add(state)
                reward_machine.accepting.add(state) 
    reward_machine.u0 = initial_state
    return reward_machine

def combine_to_single_rm(rm_dict, tag="rm", skip_aux=False):
    '''
    Converts a list of DFAs, generated by decomposition, into a decomposition of RMs, where the RMs are held in a single 
    SparseRewardMachine object, but are separate connected components.'''
    subsuming_rm = SparseRewardMachine()
    # add the auxiliary node
    auxiliary_state = 0
    subsuming_rm._add_state([auxiliary_state])
    subsuming_rm.delta_u[auxiliary_state] = {}
    subsuming_rm.delta_r[auxiliary_state] = {}
    subsuming_rm.u0 = auxiliary_state
    current_offset = 1
    sub_rm_initial_states = {}
    for idx, rm_obj in rm_dict.items():
        subsuming_rm._add_state(st + current_offset for st in rm_obj.get_states())
        subsuming_rm.delta_u[auxiliary_state]["to_{}{}".format(tag, idx)] = rm_obj.get_initial_state() + current_offset
        sub_rm_initial_states[idx] = rm_obj.get_initial_state() + current_offset
        subsuming_rm.delta_r[auxiliary_state][rm_obj.get_initial_state() + current_offset] = 0
        for r_k, r_v in rm_obj.delta_r.items():
            subsuming_rm.delta_r[r_k + current_offset] = {}
            for next_st, reward in r_v.items():
                subsuming_rm.delta_r[r_k + current_offset][next_st + current_offset] = reward
        for u_k, u_v in rm_obj.delta_u.items():
            subsuming_rm.delta_u[u_k + current_offset] = {}
            for event, next_st in u_v.items():
                subsuming_rm.delta_u[u_k + current_offset][event] = next_st + current_offset
        current_offset += len(rm_obj.get_states())
    return subsuming_rm, sub_rm_initial_states


def generate_rm_decompositions_using_dfas(monolithic_rm, num_decompositions, num_agents, disregard_list=None, n_queries=25):
    # convert rm to a DFA
    dfa_obj = rm_to_dfa(monolithic_rm)
    # create decomposition generator
    aps = monolithic_rm.get_events()
    if 'True' in aps:
        aps.remove('True')
    decomp_gnr = find_dfa_decomposition(monolithic_dfa=dfa_obj, alphabet=aps, n_dfas=num_agents, n_queries=n_queries)
    disregard_list = [] if disregard_list is None else disregard_list
    # yield decompositions, ignoring ones that we want to throw out
    decomps = []
    decomped_mono_rms = []
    while len(decomps) < num_decompositions:
        candidate = next(decomp_gnr)
        if candidate in disregard_list:
            continue
        else:
            decomped_rms = [dfa_to_rm(dfa_obj) for dfa_obj in candidate]
            subsuming_rm = combine_to_single_rm(decomped_rms)
            decomped_mono_rms.extend(decomped_rms)
            decomps.append(candidate)
    full_subsuming_rm = combine_to_single_rm(decomped_mono_rms, tag='rm')
    return full_subsuming_rm