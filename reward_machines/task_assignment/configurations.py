import itertools 

import task_assignment.helper_functions as hf


### Holds Configurations class that holds the task assignment experiment details and decomposition metric

class Configurations():
    def __init__(self, num_agents, rm, enforced_set = None, forbidden_set = None, agent_utility_function = None, weights = None , incompatible_pairs = None, include_all = True):
        '''
        This is for experiment parameters that do not change throughout the experiment. 
        Also for defining functions for calculating scores

        num_agents: integer
        rm: (Sparse Reward Machine type)
        enforced_set: {(event, agent)} that must be assigned (not removed in the knapsack) 
                      If this is a dict of the form {agent: [enforced events]} will convert to set of the correct form
        forbidden_set: {(event, agent)}, set that cannot be assigned (must be removed from knapsack) 
        agent_utility_function: (dict)
        weights: list len 3 with ints. [shared events, fairness, utility] weights of importance 
        

        Attributes:
            rm: SparseRewardMachine type
            num_agents: int
            agents: [0, 1,.. ] list of agent names
            
            weights: [shared events weight, fairness weight, utility weight]
            
            enforced_set = {(e,a)} event assignments that must be kept (DO NOT PUT IN KNAPSACK)
            forbidden_set = {(e,a)} event assignments that are not allowed (MUST BE PUT IN KNAPSACK)
            all_events = {(e,a)} rm.events X agents  
            future_events = [all_events - forbidden_set - enforced_set] THIS SETS THE ORDER OF THE TREE SEARCH. tree nodes use this to define the the "tree to come"
            
            max_knapsack_size = |rm.events| x # agents  - # enforced_set
            agent_utility_function: {agent: {event: utility} }
            total_utility_score: sum of all the agents utility 

        '''
        self.rm = rm 
        self.num_agents = num_agents
        self.agents = [i for i in range(num_agents)] 
        self.all_events = set(itertools.product(rm.events, self.agents)) #should be all events
        self.include_all = include_all
        #print("rm events", rm.events)
        #print("agents", self.agents)
        
        if weights: 
            self.weights = weights
        else:
            self.weights = [1,1,1]
        
        if enforced_set:
            if type(enforced_set) == dict: # HACK , should just make the function take a dict lol
                enforced_set = hf.get_sack_from_dict(enforced_set)
            self.enforced_set = enforced_set
        else:
            self.enforced_set = set()
        
        if forbidden_set:
            if type(forbidden_set) == dict: # HACK, should just make the function take a dict lol
                forbidden_set = hf.get_sack_from_dict(forbidden_set)
            self.forbidden_set = forbidden_set
        else:
            self.forbidden_set = set()
        
        tree_events = self.all_events - self.forbidden_set - self.enforced_set
        #print("all", self.all_events)
        #print("forbbiden", self.forbidden_set)
        #print("enforced", self.enforced_set)

        self.future_events = list(tree_events)

        self.max_knapsack_size = len(self.all_events) - len(self.enforced_set)
        if len(self.enforced_set.intersection(self.forbidden_set)) != 0:
            raise Exception("Your enforced assingments and forbidden assignments are not disjoint!!")

        if agent_utility_function:
            self.agent_utility_function = agent_utility_function
        else:
            self.agent_utility_function = {}
            for a in self.agents:
                if a not in self.agent_utility_function:
                    self.agent_utility_function[a] = {}
                for e in self.rm.events:
                    if e not in self.agent_utility_function[a]:
                        self.agent_utility_function[a][e] = 0

        self.total_utility_score = 0
        for a, ed in self.agent_utility_function.items():
            for e, u in ed.items():
                self.total_utility_score += u
        
        if incompatible_pairs:
            self.incompatible_pairs = incompatible_pairs
        else:
            self.incompatible_pairs = []

        self.restrictions = {'enforced_assignments': self.enforced_set, 'incompatible_assignments': self.incompatible_pairs, 'forbidden_assingments': self.forbidden_set}
        self.type = 'no_accidents'

    
    def get_utility(self, agent, event):
        return self.agent_utility_function[agent][event]
        
    def get_shared_event_score(self, knapsack):
        '''
        returns score of shared events
        we are hoping to minimize shared events 
        => maximize relative knapsack size
        '''
        if self.max_knapsack_size == 0:
            return 0
        se_score = len(knapsack) / self.max_knapsack_size # technically this is above the max knapsack size according to our decomposition constraints
        #print("se score", se_score, " max ", config.max_knapsack_size )
        return se_score

    def get_fairness_score(self, knapsack, event_spaces_dict): 
        kept_events = self.all_events - knapsack

        if len(kept_events) == 0:
            return 1

        #event_spaces, event_spaces_dict = self.get_event_spaces_from_knapsack()
        avg = len(kept_events) / self.num_agents

        top_sum = 0
        for i in event_spaces_dict.values():
            top_sum += abs(len(i) - avg)

        return 1 - ( top_sum / len(kept_events))

    def get_utility_score(self, knapsack):
        '''
        calculates a normalized total utility of the "kept events" and
        normalizes it to the total possible utility score
        We are trying to maximize knapsack score
        '''
        if self.total_utility_score == 0:
            return 0
        knapsack_utility_score = 0
        remaining_events = self.all_events - knapsack

        for i in remaining_events: #SHOULD NOT BE KNAPSACK, should be kept events 
            e, a = i 
            us = self.get_utility(a,e) 
            knapsack_utility_score += us

        return knapsack_utility_score / self.total_utility_score

    def get_score(self, knapsack):
        '''
        combine the other three scores, weighted with weights. 
        '''
        event_spaces, event_spaces_dict = hf.get_event_spaces_from_knapsack(self.all_events, knapsack)

        se  = self.get_shared_event_score(knapsack)
        f = self.get_fairness_score(knapsack, event_spaces_dict)
        u = self.get_utility_score(knapsack)
        
        se_weight, f_weight, u_weight = self.weights
        score = se * se_weight + f * f_weight + u * u_weight

        #print(f"se score is:, {se} , f score is: {f}, u score is: {u}, for a total of {score}")
        
        return score 
