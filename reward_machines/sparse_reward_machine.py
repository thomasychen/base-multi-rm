from dfa import DFA, dict2dfa, dfa2dict
from dfa_identify.active import find_dfa_decomposition

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
        if file is not None:
            self._load_reward_machine(file)
        
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

    # Private methods -----------------------------------

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
        reward_machine._add_state([state])
        reward_machine.delta_u[state] = {}
        reward_machine.delta_r[state] = {}
        for event in dfa_dict[state][1]:
            next_state = dfa_dict[state][1][event]
            reward_machine.delta_u[state][event] = next_state
            reward_machine.delta_r[state][next_state] = int(dfa_dict[next_state][0])
            if dfa_dict[state][0]:
                reward_machine.T.add(state)
                reward_machine.accepting.add(state) 
    reward_machine.u0 = initial_state
    return reward_machine

def generate_rm_decompositions(monolithic_rm, num_decompositions, num_agents, disregard_list=None, n_queries=25):
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
    while len(decomps) < num_decompositions:
        candidate = next(decomp_gnr)
        if candidate in disregard_list:
            continue
        else:
            rm_set = [dfa_to_rm(dfao) for dfao in candidate]
            decomps.append(rm_set)
    return decomps

def select_top_k_decompositions(decomposition_candidates, k):
    # here, use a selection heuristic to select the top k candidates. 
    # TODO: implement a heuristic.
    pass