import itertools
from reward_machines.sparse_reward_machine import SparseRewardMachine
import reward_machines.task_assignment.helper_functions as hf

class EquivalenceRelation:
    '''
    Relation should be:
        - reflexive: a ~ a
        - symmetric: a ~ b -> b ~ a
        - transitive: a ~ b, b ~ c -> a ~ c
    '''
    def __init__(self, classes = None):
        '''
        Inputs
        classes: list of sets
        
        Attributes
            classes: list holding a set of each equivalence class
        '''
        if classes == None:
            self.classes = []
        else:
            self.classes = classes

        self.check_classes()

    def __repr__(self):
        s = "Equivalence Classes: \n"
        for cl in self.classes:
            s += "    " + str(cl) + "\n"
        return s

        
    def add_relation(self, elements):
        '''
        Inputs
            elements: (list) holds strings of elements to be added to related * MAYBE CHANGE TO TUPLE?
        '''
        #print("     I am adding a relation", elements)
        cls = []
        for e in elements:
            cl = self.find_class(e) 
            if cl:
                if cl not in cls:
                    cls.append(cl)
    
        if cls: # at least one element belongs in a class already
            new_cl = self.merge_classes(cls)  #if only one element in cls, returns cls[0]
            self.add_elements_to_class(elements, new_cl)  
        
        else:
            self.add_new_class(elements) 

        self.check_classes() 

    def find_class(self, element):
        '''
        Finds class the element belongs to
        Input:
            element: (str) (or integer?) element in an equivalence class
        Returns:
            cl: (set) The set containing the element if element is already in a class
                (None) if element is not in a set
        '''
        for cl in self.classes:
            if element in cl:
                return cl
        #print('Element', element, "classes", self.classes)
        return None 
            
    def merge_classes(self, cls):
        '''
        Takes a list of classes, combines them
        Removes existing classes and replaces as ONE 
        Expecting a list of classes at least length 1
        Inputs:
            cls: (list) contains sets that are elements of the list self.classes
        Return:
            megacl: (set) merged classes
        '''
        megacl = set()
        for cl in cls:
            megacl = megacl.union(cl) # build merged class
            self.classes.remove(cl) # remove individual classes

        self.classes.append(megacl) # add merged class
        return megacl
        
    def add_elements_to_class(self, elements, cl):
        '''
        add an element to the set
        Inputs:
            elements: (list) contains strings or int to be added to cl
            cl: (set) equivalence class 
        '''  
        for e in elements:
            cl.add(e)

    def add_new_class(self, elements):
        '''
        Adds a set containting entries in element to list of classes

        elements: List or tuple of elements 
        '''
        new_class = set()
        for e in elements:
            new_class.add(e)
        self.classes.append(new_class)

    def check_classes(self):
        '''
        checks that your equivalence classes are disjoint 
        '''
        full_union = set()
        
        for cl in self.classes:
            for e in cl:
                if e in full_union:
                    raise NameError(" Classes are not disjoint")
                full_union.add(e)

    def are_related(self, elements):

        '''
        Checks if elements belong to the same equivalence class
        Inputs:
            elements: (list or tuple) 
        Returns:
            Bool, true if elements are related, false otherwise 
        '''  
        cls = []
        for e in elements:
            e_cl = self.find_class(e)
            if e_cl == None: # element is not in any equivalence class, elements cannot be related
                return False 
            cls.append(e_cl)
        if len(cls) == 1:
            return True

        if len(cls) == 0: 
            Warning("You are asking if an empty set of elements are related. Returned False")
        
        return False #elements belonged to more than one equivalenc class

    def get_all_related_combos(self):
        '''
        Get all pairs of related elements. Order does not matter.

        For example, if my classes are [{1,2,3}, {4,5}]
        all related combos would be {{1,2}, {1,3}, {2,3}, {4,5}}
        
        Returns: list set of sets
        '''
        all_related_combos = set()
        for cl in self.classes:
            related_combos= set(itertools.combinations(cl, 2))
            for x in related_combos:
                all_related_combos.add(x)
        return all_related_combos    

def get_relation(event_space, rm):
    '''
    generates equivalence class relation given set of events and a reward machine
    Follows definition of equivalence relation outlined in Cyrus' paper (above Def 2)

    Inputs
        event_space: (set) set of strings, subset of rm.events
        rm: (SparseRewardMachine)
    Returns
     (EquivalenceRelation) 

    '''
    relation = EquivalenceRelation()

    state_pairs = list(itertools.combinations_with_replacement(rm.U, 2)) # all combinations length 2 of rm.U (order doesn't matter)
    #print("     state pairs", state_pairs)
    for state_pair in state_pairs: # pair is a tuple
        if relation.are_related(state_pair) == False:
            #print("     ", state_pair , " is related")
            #print("     here", rm.events - event_space )
            u1, u2 = state_pair
            if u1 == u2:
                relation.add_relation(state_pair)

            for e in rm.events - event_space:
                if rm.is_event_available(u1,e):
                    if rm.delta_u[u1][e] == u2:
                        relation.add_relation(state_pair)

                if rm.is_event_available(u2,e):
                    if rm.delta_u[u2][e] == u1: 
                        relation.add_relation(state_pair)
        else:
            print("     ", state_pair, " is not related")
    checked_combos = set()
    all_related_combos = relation.get_all_related_combos()

    while len(all_related_combos) != 0:
        for u1, v1 in all_related_combos:
            for e in event_space: 
                if rm.is_event_available(u1,e) and rm.is_event_available(v1,e): 
                    u2 = rm.delta_u[u1][e]
                    v2 = rm.delta_u[v1][e]
                    relation.add_relation((u2, v2))
            checked_combos.add((u1, v1))
            checked_combos.add((v1, u1))

        all_related_combos = relation.get_all_related_combos() - checked_combos 
    
    return relation

def project_rm(event_space, rm):
    '''
    returns reward machine projected onto the event space. 
    Follows procedure outlined in Definition 2 of "RM for Cooperative MARL" (Cyrus' paper)

    Gets an equivalence relation for the original RM  'rm' states  
    Adds an attribut to new_rm called class_name_dict which saves what class of rm states correspond 
    to the new state names in new_rm. 

    Inputs:
        event_space: (set) hold strings that is a subset of rm.events
        rm: (SparseRewardMachine)
    Return: new_rm (SparseRewardMachine)
    '''
    relation = get_relation(event_space, rm) #put this outside to reuse? Or do I just only project WHEN I NEED TO. 
    #print("event_space in parallel", event_space)
    #print("RELATION IS")
    #rint( relation)
    new_rm = SparseRewardMachine()
    class_name_dict = {}
    new_rm.events = event_space
    for i , cl in enumerate(relation.classes):
        # add states to new_rm
        new_rm.U.append(i)
        class_name_dict[i] = cl # new state name: class of original RM state names it represents 
        
        for u in cl:
            # check if cl should be terminal or inital 
            if rm.is_terminal_state(u):
                new_rm.T.add(i)
            if rm.get_initial_state() == u:
                new_rm.u0 = i
    
    new_rm.equivalence_class_name_dict = class_name_dict 

    # built U, u0, and T, events
    # Now need to build the functions delta_u and delta_r
    
    for v1, v1_transitions in rm.delta_u.items():
        for e, v2 in v1_transitions.items():
            if e in event_space:
                #need to find what class_name for v1 and v2 belongs to. 
                 
                v1_class = relation.find_class(v1)
                v2_class = relation.find_class(v2)

                #print("V1_CLass is", v1_class)
                v1_name = new_rm.get_name_from_class(v1_class)
                v2_name = new_rm.get_name_from_class(v2_class)
                if v2_name in new_rm.T and v1_name not in new_rm.T:
                    reward = 1
                else: 
                    reward = 0
                #print("add transition", v1_name, v2_name, e, reward)
                new_rm.add_transition_and_reward_only(v1_name, v2_name, e, reward)
    
    if rm.dead_transitions != None:
        new_rm.dead_transitions = rm.dead_transitions
    return new_rm  

def put_in_parallel(rm1, rm2):
    '''
    Takes 2 reward machines and puts them in parallel. 
    UNUSED - wrote this for a conceptual first step using 
    Definition 4 in Appendix 1 of "RM for Cooperative MARL" 
    '''
    rm_parallel = SparseRewardMachine()

    U_parallel = itertools.product(rm1.U, rm2.U)
    rm_parallel.U = U_parallel

    events_parallel = rm1.events.union(rm2.events)
    rm_parallel.events = events_parallel

    T_parallel = itertools.product(rm1.T, rm2.T)
    rm_parallel.T = T_parallel

    #delta_u and delta_r:
    for u_pair in rm_parallel.U:
        u1,v1 = u_pair
        for e in rm_parallel.events:
            u2_pair = None
            if rm1.is_event_available(u1, e) and rm2.is_event_available(v1, e) and e in rm1.events.intersection(rm2.events): 
                # Case 1
                u2_pair = (rm1.get_next_state(u1,e), rm2.get_next_state(v1,e))

            elif rm1.is_event_available(u1, e) and e in rm1.events - rm2.events:
                # Case 2
                u2_pair = (rm1.get_next_state(u1,e), v1)    

            elif rm2.is_event_available(v1, e) and e in rm2.events - rm1.events:
                # Case 3
                u2_pair = (u1, rm2.get_next_state(v1,e))

            if u2_pair:
                reward = 0
                if u2_pair in rm_parallel.T and u_pair not in rm_parallel.T:
                    reward = 1 
                rm_parallel.add_transition_and_reward_only( u_pair, u2_pair, e, reward)
    return rm_parallel

def find_defined_transitions(rms, us, e):
    '''
    Given a list of reward machines and a state of their paralell composition, 
    determines which entries of us the event is defined for. 
    
    Example:
        if us = (0, 1, 1) and
        delta_u(0,e) is defined for rm1,
        delta_u( 1,e) NOT defined for rm2
        delta_u(1,e) is defined for rm3
        find_defined_transitions would return (True, False, True)

    Inputs
        rms: list of reward machine types [rm1, rm2, .... rmN]
        us: list of ints. this is and element of rm1.U X rm2.U x ... X rmN.U
        e: str. event of at least one rm in rms
    returns: list of booleans
    '''
    availability = []
    for i, rm in enumerate(rms): 
        if rm.is_event_available(us[i], e):
            availability.append(True)
        else:
            availability.append(False)
    return availability 

def put_many_in_parallel(rms):
    '''
    Finds the parallel compostion of several reward machines. 
    Based on Definition 4 in Appendix 1 of "RM for Cooperative MARL" 


    _______Example of transition function work_______
    rm_p = rm1 || rm2 || rm3
    u1 = (0, 0, 0)
    event = a

    delta_1(0, a) = 2
    delta_2(0, a) = 1
    delta_3(0, a) undefined 

    delta_p(u1 , a) = (delta_1(0, a), delta_2(0, a), 0) if a NOT IN  rm3.events 
                    = undefined if a IN rm3.events

    If delta_2(0,a) is also undefined:
    delta_p(u1, a) = (delta_1(0,a), 0, 0) if a NOT IN rm2.events or rm3.events
                   = undefined otherwise 
    ________________________________________________ 

    Inputs
        rms: (list) holding SparseRewardMachines
    Return:
        rm_parallel: (SparseRewardMachine)
    
    '''
    # Initialize new RM
    rm_parallel = SparseRewardMachine()
    
    # get rm states for each rm 
    U_collection = [rm.U for rm in rms]
    
    # build parallel state space (cartesian product of all of the rm state spaces)
    U_parallel = list(itertools.product(*U_collection))
    rm_parallel.U = U_parallel
    
    # get parallel event space 
    events_collection = [rm.events for rm in rms]
    events_parallel = set()
    events_parallel = events_parallel.union(*events_collection)
    rm_parallel.events = events_parallel

    # get parallel terminal event space 
    T_collection = [rm.T for rm in rms]
    T_parallel = list(itertools.product(*T_collection))
    rm_parallel.T = T_parallel

    # Record meaning of each states' original equivalence class meaning. 
    for i, rm in enumerate(rms):
        rm_parallel.equivalence_class_name_dict[i] = rm.equivalence_class_name_dict  # FIXME
        # this is bad because it is another use for equivalence_class_name_dict ... 
        # keys correspond to "order" in rms, while should correspond to name of states -order is already held in rms list

    # get intial states
    U_init_collection = [rm.u0 for rm in rms]
    rm_parallel.u0 = tuple(U_init_collection)

    # add transitions and rewards between (u_pair, u2_pair)
    for u_pair in rm_parallel.U:
        for e in rm_parallel.events:
            # will do this for each (state , event) in rm_parallel 

            # TODO: talk to Cyrus about the extension of the definition of || rm transition function
            
            # determine if transtion is defined: 
            
            availablilty = find_defined_transitions(rms, u_pair, e) # Ex:              availability = (True, True False)

            defined_event_spaces = set()                            # Ex:      defined_event_spaces = rm_1.events U rm_2.events
            non_defined_event_spaces = set()                        # Ex:  non_defined_event_spaces = rm3.events 
            
            for i, av in enumerate(availablilty):
                rm = rms[i]
                if av:
                    defined_event_spaces = defined_event_spaces.union(rm.events) # union of the event spaces of the RM on which the transition IS locally defined
                else:
                    non_defined_event_spaces = non_defined_event_spaces.union(rm.events) # union of the event spaces of the RM on which the tranisition IS NOT locally defined 

            # find where transition ends (if it exists) 
            u2_pair = None
            if e in defined_event_spaces - non_defined_event_spaces: # Ex: e IN rm_1.events U rm_2.events, e NOT IN rm3.events
                u2 = [] 
                for j , av in enumerate(availablilty):
                    if av: 
                        next_state = rms[j].get_next_state(u_pair[j],e) 
                    else:
                        next_state = u_pair[j]
                    u2.append(next_state)
                u2_pair = tuple(u2)                                 # Ex:   u2_pair = (delta_1(0), delta_2(0), 0)


            # add transition and reward 
            if u2_pair:
                reward = 0
                if u2_pair in rm_parallel.T and u_pair not in rm_parallel.T:
                    reward = 1 
                rm_parallel.add_transition_and_reward_only(u_pair, u2_pair, e, reward)
    
    return rm_parallel

def is_bisimilar(rm_1, rm_2):
    #rm_2 = rm_parallel
    R = [] #line 1 # should this be a set? 
    todo = [] #line1
    rm_1_o = rm_1.get_initial_state()
    rm_2_o = rm_2.get_initial_state()
    todo.append((rm_1_o , rm_2_o)) #line 2

    #print("starting todo ", todo)
    while todo: # line 3
        # need to remove m from todo
        m = todo[0] # line 3.1
        todo.remove(m)
        x, y = m 

        if m not in R:  # line 3.2  really says, if m in R then skip # TODO: check skip meaning 
            # Do everything below 

            if rm_1.is_terminal_state(x) != rm_2.is_terminal_state(y): # line 3.3
                #print("Was false, one of these is terminal and the other is not:")
                #print("in rm_1", x, "terminal is",rm_1.T) 
                #print("in rm_2", y, "terminal is", rm_2.T)

                return False
            for a in rm_1.events:  # line 3.4 # A = rm.events
                x_next = rm_1.get_next_state(x,a) 
                y_next = rm_2.get_next_state(y,a)
                next_m = (x_next, y_next)

                todo.append(next_m) 
            R.append(m)  # line 3.5            
    return True #line 4

def can_win_check(rm, goal_state = None, u0 = None):
    '''
    Checks to see if there is a transition sequence that lands us at '''

    if u0 == None:
        u0= rm.u0
    
    if goal_state != None:
        final_states = {goal_state}
    else:
        terminal_set = rm.T
        final_states = terminal_set.copy()

    for u in final_states:
        if u == u0: #was rm.u0
            return True

    reachability_set = final_states
    keep_searching = True

    while keep_searching:
        origins = rm.get_origins(reachability_set)
        new_origins = origins - reachability_set
        if len(new_origins) == 0:  
            return False
        for u in new_origins:
            if u == u0: #was rm.u0
                return True
            reachability_set.add(u)

    return False

def remove_rm_transitions(rm, strategy_set): # should really be called remove_transitions
    '''
    return a reward machine with only transitions in strategy_set remaining. 
    '''
    strategic_rm = SparseRewardMachine()
    strategic_rm.U = rm.U.copy()
    strategic_rm.events = strategy_set
    strategic_rm.u0 = rm.u0
    strategic_rm.T = rm.T.copy()
    strategic_rm.equivalence_class_name_dict = rm.equivalence_class_name_dict.copy()  

    non_strat_set = rm.events - strategy_set
    #print('non strat set', non_strat_set)
    strategic_rm.delta_u = {}
    strategic_rm.delta_r = {}

    for u_1, transition_dict in rm.delta_u.items():
        if u_1 == 8:
            pause = True
        td_copy = transition_dict.copy()
        rd = rm.delta_r[u_1]
        rd_copy = rd.copy()
        for e in non_strat_set:
            if e in td_copy.keys():
                u2 = td_copy[e]
                get_rid = True
                for se in strategy_set:
                    if se in rm.delta_u[u_1].keys():
                        if rm.delta_u[u_1][se] == u2:
                            get_rid = False
                if u2 in rd_copy.keys():
                    if get_rid == True:
                        rd_copy.pop(u2)
                td_copy.pop(e)
        strategic_rm.delta_u[u_1] = td_copy
        strategic_rm.delta_r[u_1] = rd_copy

        # you have to do a similar thing to reward.

    #print("  ")
    return(strategic_rm)

def remove_unreachable_states(rm):
    unreachable = set()
    for u in rm.U:
        #print("looking at ", u)
        if not can_win_check(rm, u):
            unreachable.add(u)
            if u in rm.delta_u.keys():
                rm.delta_u.pop(u)
            if u in rm.delta_u.keys():
                rm.delta_r.pop(u)

    #print("unreachable is", unreachable) 
    new_U = []
    for u in rm.U:
        if u not in unreachable:
            new_U.append(u)
    rm.U = new_U
    #print("rm U is", rm.U)

def remove_dead_transitions(rm):
    #print("before dead transitions")
    #print(rm)

    dead_states = set()
    bad_dict = {}
    for u in rm.U:
        if can_win_check(rm, u0 = u) == False:
            dead_states.add(u)
            for v in rm.U:
                if v in bad_dict.keys():
                    bad_dict_v = bad_dict[v]
                else:
                    bad_dict_v = {}
                if v != u:
                    if v in rm.delta_u.keys():
                        td = rm.delta_u[v].copy()
                        for e, v2 in td.items():
                            if v2 == u:
                                bad_dict_v[e] = u
                                rm.delta_u[v].pop(e)

                    if v in rm.delta_r.keys():
                        rd = rm.delta_r[v].copy()
                        for v3, r in rd.items():
                            if v3 == u:
                                rm.delta_r[v].pop(u)
                else:
                    if u in rm.delta_u.keys():
                        bad_dict[u] = rm.delta_u[u]
                        rm.delta_u.pop(u)
                    if u in rm.delta_u.keys():
                         rm.delta_r.pop(u)

                bad_dict[v] = bad_dict_v

    new_U = []
    for u in rm.U:
        if u not in dead_states:
            new_U.append(u)
    rm.U = new_U
    rm.dead_transitions = bad_dict 


def is_decomposable(rm_f, rm_p, agent_event_spaces_dict, num_agents, enforced_set = None, prints = False, incompatible_pairs = None, upcomming_events= None): #rm_p should not be an entry in this? yy
    '''
    Checks all three conditions for rm_p and rm_f being 
    a passable decomposition. 
        Check 1: 
            all events need to be assigned somewhere 
    
    Format: check each condition in order return False if any fail
    
    Inputs
        rm_p: (SparseRewardMachine) rm of objects placed in parallel
        rm_f: (SparseRewardMachine) full rm 
        agent_event_spaces_dict: {agent: [events] }

        knapsack: (set of tuples) {(event, agent), ...}
        num_agents: (int) 
        enforced_set: (set of tuples) default None. Should be same format as knapsack. 
                        
    Returns: bool
    '''
    ##### 1 All events are assigned ####
    #if rm_p.events != rm_f.events:
    #    if prints:
    #        print("fails case 1, some events are not assigned")
    #    return False
    
    #### 2 All agents must have an assignment #####
    active_agents = []
    for a, es in agent_event_spaces_dict.items():
        if es:
            active_agents.append(a)

    if len(active_agents) != num_agents:
        if prints:
            print(" fails case 2, some agents not assigned to anything")
        return False
    
    #### 3 each agent has their required events ####
    if enforced_set: 
        for i in enforced_set:
            ev, ag = i 
            if ev not in agent_event_spaces_dict[ag]:
                raise Exception("You were so confident that you were not going to get here you decided to raise an exception \n You had an enforced set problem in is_decomposible")

                if prints:
                    print(" fails case 3, agents don't have required events")
                return False
    ##### 4 incompatible pairs #### give pairs of events that cannot be assigned to the same agents
    if incompatible_pairs == None:
        incompatible_pairs = []
    for pair in incompatible_pairs:
        e1 , e2 = pair
        for a , es in agent_event_spaces_dict.items():
            if e1 in es:
                if e2 in es:
                    if (e1, a) not in upcomming_events: 
                        if (e2, a) not in upcomming_events:
                            if prints:
                                print(" doomed forever" , (e1,a), (e2,a) , upcomming_events, agent_event_spaces_dict)
                            return False
                    
    ##### 5 bisimilarity holds ####
    if not is_bisimilar(rm_p, rm_f):
        if prints:
            print("fails case 5, not bisimilar")
        return False
    
    return True 

def check_restrictions(configs, agent_event_spaces_dict, upcomming_events):  # If it is easier for this to take knapsack we can do that too. 
    '''
    this function checks if a knapsack satisfes the forbidden, enforced, and incompatible assignments. 
    '''
    restrictions = configs.restrictions

    have_hope = True
    restrictions_pass = True
    ########## check all agents assigned something? ### (REMOVE?) ######
    active_agents = []
    for a, es in agent_event_spaces_dict.items():
        if es:
            active_agents.append(a)
    
    if len(active_agents) != configs.num_agents:
        # you have removed too many agents
        restrictions_pass = False
        have_hope = False
        return restrictions_pass, have_hope

    ######### check all enforced events are assigned ################## 
    enforced_assignments = restrictions['enforced_assignments']
    for i in enforced_assignments: 
        ev, ag = i
        if ev not in agent_event_spaces_dict[ag]:
            restrictions_pass = False
            have_hope = False
            return restrictions_pass, have_hope

    ########### Check if incompatible events are violated #############
    incompatible_pairs = restrictions['incompatible_assignments']
    for pair in incompatible_pairs:
        e1 , e2 = pair
        for a , es in agent_event_spaces_dict.items():
            if e1 in es:
                if e2 in es:
                    restrictions_pass = False # this one failed, but is there hope? 
                    if (e1, a) not in upcomming_events: 
                        if (e2, a) not in upcomming_events:
                            # neither event is in upcomming. fails
                            have_hope = False
                            return restrictions_pass, have_hope

    return restrictions_pass, have_hope
                    
def is_decomposible_no_accidents(configs, event_spaces):
    # have already satisfied that it does not break any restrictions. 

    # check if valid "no accidents" strategy 
    strategy_set = set()
    for es in event_spaces:
        strategy_set = strategy_set.union(es)

    strategic_rm = remove_rm_transitions(configs.rm, strategy_set) 
    remove_unreachable_states(strategic_rm)

    if can_win_check(strategic_rm): # check if it is a valid strategy ?
        # yes? continue on to check if bisimilar. 
        rms = []

        for es in event_spaces:
            rm_p = project_rm(es, strategic_rm)  #project each reward machine down onto the event spaces
            rms.append(rm_p)

        rm_parallel = put_many_in_parallel(rms) 
        bisim_check = is_bisimilar(rm_parallel, strategic_rm)

        check_children = True # default is to check all children even if "bisim" fails. Why not. # CHECK 

    else: # You removed an unforgivable edge. You can no longer reach the end. There is no fixing this. 
        bisim_check = False
        check_children = False

    return bisim_check, check_children 

def get_strategy_rm(rm, strategy_set, full_removal = True):
    strategic_rm = remove_rm_transitions(rm, strategy_set)
    #print(strategic_rm)
    if full_removal:
        remove_dead_transitions(strategic_rm) #added this, maybe remove? 
        remove_unreachable_states(strategic_rm)

    return strategic_rm

def get_accident_avoidance_rm(individual_rm, accident_set):
    dead_state = -1
    dead_reward = -1
    acc_rm_u = individual_rm.U.copy()
    acc_rm_e = individual_rm.events.copy()
    acc_rm_T = individual_rm.T.copy()
    acc_rm_u0 = individual_rm.u0
    
    delta_u = {}
    delta_r = {}
    for u, td in individual_rm.delta_u.items():
        delta_u[u] = td.copy()
    for u, rd in individual_rm.delta_r.items():
        delta_r[u] = rd.copy()

    for e in accident_set:
        acc_rm_e.add(e)
        for u in acc_rm_u:
            if u in delta_u.keys():
                td = delta_u[u]
            else:
                td = {}
            td[e] = dead_state

            delta_u[u] = td
        for v in acc_rm_u:
            if v in delta_r.keys():
                rd = delta_r[v]
            else:
                rd = {}
            rd[dead_state] = dead_reward 
            delta_r[v] = rd
            

    acc_rm_u.append(dead_state)
    acc_rm_T.add(dead_state)

    acc_rm = SparseRewardMachine()
    acc_rm.U = acc_rm_u
    acc_rm.events = acc_rm_e
    acc_rm.T = acc_rm_T
    acc_rm.u0 = acc_rm_u0
    acc_rm.delta_u = delta_u
    acc_rm.delta_r = delta_r

    return acc_rm

def get_accident_avoidance_rm_less(individual_rm, accident_set, rm):
    # print("Its about to get crazy buckle up. ")
    dead_state = -1
    dead_reward = -1
    acc_rm_u = individual_rm.U.copy()
    acc_rm_e = individual_rm.events.copy()
    acc_rm_T = individual_rm.T.copy()
    acc_rm_u0 = individual_rm.u0
    acc_rm_equivalence_class_name_dict = individual_rm.equivalence_class_name_dict.copy()

    delta_u = {}
    delta_r = {}
    for u, td in individual_rm.delta_u.items():
        delta_u[u] = td.copy()
    for u, rd in individual_rm.delta_r.items():
        delta_r[u] = rd.copy()


    # look at each state in the individual RM. 
    # for each e in accident: see if it exists for one of the corresponding states in the original reward machine. 
    for u1 in acc_rm_u:
        original_equivalent_states = individual_rm.equivalence_class_name_dict[u1]
        for v in original_equivalent_states:
            if v in rm.delta_u.keys():
                original_possible_transitions = rm.delta_u[v].keys()
                for e in accident_set:
                    if e in original_possible_transitions:
                        acc_rm_e.add(e)
                        if u1 in delta_u.keys():
                            td = delta_u[u1]
                        else:
                            td = {}
                            
                        if u1 in delta_r.keys():
                            rd = delta_r[u1]
                        else:
                            rd = {}
                        td[e] = dead_state
                        rd[dead_state] = dead_reward
                        delta_u[u1] = td
                        delta_r[u1] = rd
                
    
    acc_rm_u.append(dead_state)
    acc_rm_T.add(dead_state)

    acc_rm = SparseRewardMachine()
    acc_rm.U = acc_rm_u
    acc_rm.events = acc_rm_e
    acc_rm.T = acc_rm_T
    acc_rm.u0 = acc_rm_u0
    acc_rm.delta_u = delta_u
    acc_rm.delta_r = delta_r
    acc_rm.equivalence_class_name_dict = acc_rm_equivalence_class_name_dict
    
    return acc_rm


def get_accident_avoidance_rm_less_2(individual_rm, accident_set, rm):
    print("Its about to get crazy buckle up. ")
    dead_state = -1
    dead_reward = -1
    acc_rm_u = individual_rm.U.copy()
    acc_rm_e = individual_rm.events.copy()
    acc_rm_T = individual_rm.T.copy()
    acc_rm_u0 = individual_rm.u0
    acc_rm_equivalence_class_name_dict = individual_rm.equivalence_class_name_dict.copy()

    delta_u = {}
    delta_r = {}
    for u, td in individual_rm.delta_u.items():
        delta_u[u] = td.copy()
    for u, rd in individual_rm.delta_r.items():
        delta_r[u] = rd.copy()

    # look at each state in the individual RM. 
    # for each e in accident: see if it exists for one of the corresponding states in the original reward machine. 
    
    for u1 in acc_rm_u:
        original_equivalent_states = individual_rm.equivalence_class_name_dict[u1]
        for v in original_equivalent_states:
            if v in rm.delta_u.keys():
                original_possible_transitions = rm.delta_u[v].keys()
                
                for e in rm.events:
                    if e in accident_set:
                        if e in original_possible_transitions:
                            acc_rm_e.add(e)
                            if u1 in delta_u.keys():
                                td = delta_u[u1]
                            else:
                                td = {}
                                
                            if u1 in delta_r.keys():
                                rd = delta_r[u1]
                            else:
                                rd = {}
                            td[e] = dead_state
                            rd[dead_state] = dead_reward
                            delta_u[u1] = td
                            delta_r[u1] = rd

                    elif v in individual_rm.dead_transitions.keys():
                        dead_transitions_from_v = individual_rm.dead_transitions[v] # these are transitions I removed that are in my strategy set
                        if e in dead_transitions_from_v.keys():
                            if u1 in delta_u.keys():
                                td = delta_u[u1]
                            else:
                                td = {}
                            if u1 in delta_r.keys():
                                rd = delta_r[u1]
                            else:
                                rd = {}
                            td[e] = dead_state
                            rd[dead_state] = dead_reward
                            delta_u[u1] = td
                            delta_r[u1] = rd
    
    acc_rm_u.append(dead_state)
    acc_rm_T.add(dead_state)

    acc_rm = SparseRewardMachine()
    acc_rm.U = acc_rm_u
    acc_rm.events = acc_rm_e
    acc_rm.T = acc_rm_T
    acc_rm.u0 = acc_rm_u0
    acc_rm.delta_u = delta_u
    acc_rm.delta_r = delta_r
    acc_rm.equivalence_class_name_dict = acc_rm_equivalence_class_name_dict
    
    return acc_rm
    