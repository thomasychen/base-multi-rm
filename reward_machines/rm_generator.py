import itertools
from sparse_reward_machine import SparseRewardMachine, combine_to_single_rm

def generate_le_decompositions(set_x, num_subsets, size_weight=1.0, fairness_weight=0.4, top_k=5):
    # Step 1: Generate all possible subsets of X (including overlapping possibilities)
    powerset = list(itertools.chain.from_iterable(itertools.combinations(set_x, r) for r in range(1, len(set_x)+1)))
    
    # Step 2: Generate all combinations of `num_subsets` subsets from the powerset
    all_decompositions_and_scores = []
    
    for subsets in itertools.combinations_with_replacement(powerset, num_subsets):
        # Step 3: Check if the union of the selected subsets covers the original set X
        union_of_subsets = set().union(*subsets)
        if union_of_subsets == set_x:
            score = score_decomposition_size(len(set_x), num_subsets, subsets) * size_weight + score_decomposition_fairness(subsets) * fairness_weight
            all_decompositions_and_scores.append((subsets, score))
    
    # order based on score
    all_decompositions_and_scores.sort(key=lambda x: x[1], reverse=True)
    top_k = min(top_k, len(all_decompositions_and_scores))
    return all_decompositions_and_scores[:top_k]

def score_decomposition_size(num_events, num_agents, decomposition):
    decomposition_size = 0
    for le_set in decomposition:
        decomposition_size += len(le_set)
    score = ((num_events * num_agents) - decomposition_size) / (num_events * num_agents)
    assert score >= 0 and score <= 1
    return score

def score_decomposition_fairness(decomposition):
    decomposition_size = sum([len(le_set) for le_set in decomposition])
    average_decomposition_size = decomposition_size / len(decomposition)
    fairness_score = 1 - sum([abs(len(le_set) - average_decomposition_size) for le_set in decomposition]) / decomposition_size
    assert fairness_score <= 1
    return fairness_score

def generate_rm_decompositions(monolithic_rm: SparseRewardMachine, num_agents, top_k=5):
    event_decomps_and_scores = generate_le_decompositions(monolithic_rm.get_events(), num_agents, top_k=top_k)
    rm_decomps = []
    for event_sets, score in event_decomps_and_scores:
        decomp = [project_rm(set(event_set), monolithic_rm) for event_set in event_sets]
        rm_decomps.append(combine_to_single_rm(decomp))
    subsuming_rm = combine_to_single_rm(rm_decomps)
    return subsuming_rm

# The below code adapted from https://github.com/smithsophia1688/automated_task_assignment_with_rm
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
    
    # if rm.dead_transitions != None:
    #     new_rm.dead_transitions = rm.dead_transitions
    return new_rm  
    

# Example usage:
X = {1, 2, 3}
ex_rm = SparseRewardMachine("overcooked/asymm_advantages/mono_asymm_adv.txt")
num_agents = 2
result = generate_rm_decompositions(ex_rm, num_agents, top_k=5)
print(result)