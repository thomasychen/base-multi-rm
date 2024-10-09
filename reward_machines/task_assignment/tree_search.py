# holds node class which builds the tree
# also runs the depth first tree search via recursion 

import reward_machines.task_assignment.helper_functions as hf
import reward_machines.task_assignment.bisimilarity_check as bs



class Node:
    def __init__(self, name = None, children = None, value = -1, knapsack = None, future_events = None, all_events = None, depth = 0, is_vaild = True, doomed = False, is_parent_valid = True):
        if name: ## 'root' -> next_event_name= self.future_events[0]
            self.name = name
        else: 
            self.name = 0

        if children: ## NO
            self.children = children
        else: 
            self.children = [] 

        self.value = value ## -1 in 'root' (not passed) -> 1 or 0 (yes/no in knapsack)

        if knapsack: ## configs.forbidden_events in 'root' -> new_knapsack knapsack.union({next_event_name}) ONLY FOR v =1 
            self.knapsack = knapsack
        else:
            self.knapsack = set()

        if type(future_events) == list: ## configs.future_events in 'root' -> next_events = self.future_events[1:]
            self.future_events = future_events
        else:
            self.future_events = []

        if all_events: ## configs.all_events in 'root'  -> self.all_events
            self.all_events = all_events

        else:
            self.all_events = set()
        self.depth = depth ## 0 in 'root' (not passed) -> new_depth (depth +1)
        
        self.doomed = doomed

        self.is_parent_valid = is_parent_valid
        self.is_valid = is_vaild
        self.strategic_rm = None

    def __repr__(self):
        s =  "(" + str(self.name) + ", " + str(self.value) + ")"
        return s 

    def add_children(self, children):
        '''
        adds children to self.children list
        Inputs
            children: (list) holding Node types '''
        self.children.extend(children)
    
    def remove_child(self, child):
        '''
        removes the child from the list of children 
        Inputs
            child: (Node) 
        '''
        if child in self.children:
            self.children.remove(child)
        else: 
            s = str(child) + "was not in" + str(self.name) + "'s children"
            print(s)

    def run_check(self, configs):
        '''
        No forbidden set since tree search already has forbidden set in knapsack and no forbidden events in tree search levels

        '''
        event_spaces, agent_event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, self.knapsack)
        
        # Get projected rm to put in parallel 

        rms = []
        for es in event_spaces:
            rm_p = bs.project_rm(es, configs.rm) 
            rms.append(rm_p)

        rm_parallel = bs.put_many_in_parallel(rms) 

        is_decomp = True
        #if self.value != 0: # only check if you changed something
        is_decomp = bs.is_decomposable(configs.rm, rm_parallel, agent_event_spaces_dict, configs.num_agents, enforced_set = configs.enforced_set, incompatible_pairs = configs.incompatible_pairs, upcomming_events = self.future_events)

        return is_decomp

    def run_check_2(self, configs):

        if configs.type != 'no_accidents':
            print("Freak out not ready for accidents yet")

        event_spaces, agent_event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, self.knapsack)

        restrictions_pass, have_hope = bs.check_restrictions(configs, agent_event_spaces_dict, self.future_events)

        if restrictions_pass: # here, we satisfied all of our restrictions
            bisim_check, check_children = bs.is_decomposible_no_accidents(configs, event_spaces) 

        else:
            if have_hope: 
                bisim_check = False
                check_children = True # default is to check all children 
                # restrictions have failed but there is hope they are fixed. What should I do? 

            else:
                bisim_check = False
                check_children = False

        return bisim_check, check_children #need to incorperate this into my tree search. 
            
    def check_win_ability(self, configs, event_spaces):
        '''
        This function checks if removing the assingments 
        in a given knapsack (leaving the event_spaces) leaves
        a winnable strategy
        '''
        strategy_set = set()
        for es in event_spaces:
            strategy_set = strategy_set.union(es)

        strategic_rm = bs.get_strategy_rm(configs.rm, strategy_set, full_removal = False)
        if  not bs.can_win_check(strategic_rm):
            self.doomed = True
            self.is_valid = False 
            #self.strategic_rm = strategic_rm
            #print("I made it into win ability and it was winable")
        #else: 
            
        
    def check_sufficent_agents(self, configs, agent_event_spaces_dict):
        active_agents = []
        for a, es in agent_event_spaces_dict.items():
            if es:
                active_agents.append(a)
        if len(active_agents) != configs.num_agents:
            # you have removed too many agents
            self.doomed = True
            self.is_valid = False

    def check_incompatible_assignments(self, configs, agent_event_spaces_dict):
        
        incompatible_pairs = configs.restrictions['incompatible_assignments']
        for pair in incompatible_pairs:
            e1 , e2 = pair
            for a , es in agent_event_spaces_dict.items():
                if (e1 in es) and (e2 in es):
                    self.is_valid = False # this node itself is bad but it could get better
                    if ( (e1, a) not in self.future_events) and ( (e2, a) not in self.future_events ): 
                        self.doomed = True
                                          
    def run_check_last_minute(self, configs): 

        event_spaces, agent_event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, self.knapsack)
        if self.is_valid:

            if self.value != 0:
                if not self.doomed:
                    self.check_sufficent_agents(configs, agent_event_spaces_dict) # I would like to remove this? Doing fairness' job. 
                #if not self.doomed:
                #    self.check_incompatible_assignments(configs, agent_event_spaces_dict) # don't need to check since I was valid 
                if not self.doomed:
                    self.check_win_ability(configs, event_spaces)

        else: # not valid 
            if self.value != 0:
                if not self.doomed:
                    self.check_sufficent_agents(configs, agent_event_spaces_dict) # I would like to remove this? Doing fairness' job. 
                if not self.doomed:
                    self.check_incompatible_assignments(configs, agent_event_spaces_dict) # do this check regardless of value
                if not self.doomed:
                    self.check_win_ability(configs, event_spaces)
            if self.value == 0:
                if not self.doomed:
                    self.is_valid = True
                    self.check_incompatible_assignments(configs, agent_event_spaces_dict) # do this check regardless of value
                    
        if self.value == -1:
            self.is_parent_valid = self.is_valid
        
    def run_check_last_minute_2(self, configs):
        event_spaces, agent_event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, self.knapsack)
        self.event_spaces = event_spaces
        self.agent_event_spaces_dict = agent_event_spaces_dict
        if self.value == -1: # this is the root node
            print('value is -1')
            print(f" value is {self.value}, doomed is {self.doomed}, is valid is {self.is_valid}, and is_parent_valid is {self.is_parent_valid}")
            # check everything (it could be root rot)
            if not self.doomed:
                if configs.include_all:
                    print("checking sufficent agents")
                    self.check_sufficent_agents(configs, agent_event_spaces_dict) # I would like to remove this? Doing fairness' job. 
            if not self.doomed:
                print("checking incompatible assignments")
                self.check_incompatible_assignments(configs, agent_event_spaces_dict) # do this check regardless of value
            if not self.doomed:
                print("chekcing win ability")
                self.check_win_ability(configs, event_spaces)

            print(f" updated: value is {self.value}, doomed is {self.doomed}, is valid is {self.is_valid}, and is_parent_valid is {self.is_parent_valid}")
            self.is_parent_valid = self.is_valid #TODO: do I need this line? 

        elif self.value == 1: 
            if self.is_parent_valid: 
                # Do not need to check incompatible assignments since parent is already happy, meaning we removed all incompatible assignments
                if not self.doomed:
                    if configs.include_all:
                        self.check_sufficent_agents( configs, agent_event_spaces_dict)
                if not self.doomed: 
                    self.check_win_ability(configs, event_spaces)
            else:
                if not self.doomed:
                    if configs.include_all:
                        self.check_sufficent_agents( configs, agent_event_spaces_dict)
                if not self.doomed:
                    self.check_incompatible_assignments(configs, agent_event_spaces_dict) # I need to have default is_valid = True
                if not self.doomed: 
                    self.check_win_ability(configs, event_spaces)
        
        elif self.value == 0:
            if not self.is_parent_valid: 
                self.check_incompatible_assignments(configs, agent_event_spaces_dict) # do this check regardless of value

    def small_traverse(self, configs, best_sack = (0, [])):
        '''
        Recursive function
        Completes a depth first tree search
        Inputs:
            num_agents: (int)
            rm: (SparseRewardMachine instance)
            config: (Configurations instance)
            best_sack: (int, list of sets) holds the knapsack with the best score and a list of all knapsacks (sets with elements (e,a)) that have that score
            enforced_set: (set) {(e,a)} sets that cannot go into the knapsack
            forbidden_set: (set {(e,a)} set that must go into the knapsack (In configs, we automatically remove the forbidden assignments from the future events so I DO NOT THINK THIS IS NEEDED. )
        '''
        rm = configs.rm
        
        if self.depth <= 4:
            prints = False
        else: 
            prints = False

        if prints:
            ## PRINT STATEMENTS ##
            s_start = '\t |' * self.depth + "--"
            s = s_start + str(self) + " with " + str(len(self.future_events)) + " future events"
            print(s)
            #######################

        is_decomp = self.run_check(configs)


        if prints:
            ## PRINT STATEMENTS ##
            s_header = "\t |" * (self.depth) + "\t  "
            if is_decomp:
                s = s_header + "... decomposible"
            else:
                s = s_header + "... NOT decomposible"
            print(s)
            #######################

        # if decomposible and remaining events to add, add children with values 1 & 0
        
        if is_decomp: # I think I need to get rid of this condition? or is it I should not fail the decomp.. 

            if self.future_events: 
                # build children 

                next_event_name  = self.future_events[0]
                next_events = self.future_events[1:]
                new_depth = self.depth + 1
                new_knapsack = self.knapsack.union({next_event_name}) # will only be used for child_1

                # add children
                child_1 = Node(name = next_event_name, value = 1, knapsack = new_knapsack, future_events = next_events, all_events = self.all_events, depth = new_depth)
                child_0 = Node(name = next_event_name, value = 0, knapsack = self.knapsack, future_events = next_events, all_events = self.all_events, depth = new_depth)
                self.add_children([child_1, child_0])

            if not self.children:
                if prints:
                    ## PRINT STATEMENTS ## 
                    s = s_header + "SUCCESSFUL TERMINAL, knapsack size = " + str(len(self.knapsack))
                    print(s)
                    #######################

                # Update sucsessful find 
                knap_score = configs.get_score(self.knapsack)

                if knap_score >= best_sack[0]:

                    if knap_score == best_sack[0]:
                        best_sack[1].append(self.knapsack) # this could get really long 
                    else:
                        best_sack = (knap_score, [self.knapsack])

        else: 
            if not self.children:
                if prints:
                    ## PRINT STATEMENTS ## 
                    s = s_header + "is a FAILED TERMINAL "
                    print(s)
                    #######################
                
        # Recursion Step: 
        for child in self.children:
            best_sack = child.small_traverse(configs, best_sack = best_sack)
        
        if prints:
            ## PRINT STATEMENTS ## 
            s = '\t |'* self.depth +  "  Exiting " +str(self)+ " with best sack = " + str(best_sack[0])
            print(s)
            s = "\t |" * (self.depth-  1) + '\t'
            print(s )
            #######################

        return best_sack

    def new_traverse(self, configs, best_sack = (0, [])):

        if self.depth <= 9:
            prints = False
        else: 
            prints = False

        if prints:
            ## PRINT STATEMENTS ##
            s_start = '\t |' * self.depth + "--"
            s = s_start + str(self) + " with " + str(len(self.future_events)) + " future events"
            print(s)
            #######################

        # step 1: check if our knapsack is decomposible and if we should check the kids
        is_decomp, check_kids = self.run_check_2(configs)
        #craft_check = False
        #if ('craft', 2) in self.knapsack: 
        #    craft_check = True
        #    print(self.knapsack, is_decomp, check_kids)

        # step 2: if check kids, add kids. 
        if check_kids: 
            if self.future_events:  # Make children if I can
                next_event_name  = self.future_events[0]
                next_events = self.future_events[1:]
                new_depth = self.depth + 1
                new_knapsack = self.knapsack.union({next_event_name}) # will only be used for child_1

                # Add children
                child_1 = Node(name = next_event_name, value = 1, knapsack = new_knapsack, future_events = next_events, depth = new_depth)
                child_0 = Node(name = next_event_name, value = 0, knapsack = self.knapsack, future_events = next_events, depth = new_depth)
                self.add_children([child_1, child_0])

            else: # there are no future event events to add as kids
                if is_decomp: # If decomposible, I should collect data on this node. 
                    knap_score = configs.get_score(self.knapsack)
                    if knap_score >= best_sack[0]:
                        if knap_score == best_sack[0]:
                            best_sack[1].append(self.knapsack) # this could get really long 
                        else:
                            best_sack = (knap_score, [self.knapsack])

        else:  # don't bother checking kids. 
            if is_decomp: # I don't think this can ever happen, if it does, collect score
                print("WTF ")
                knap_score = configs.get_score(self.knapsack)
                if knap_score >= best_sack[0]:
                    if knap_score == best_sack[0]:
                        best_sack[1].append(self.knapsack) # this could get really long 
                    else:
                        best_sack = (knap_score, [self.knapsack])

        
        # Recursion Step: 
        for child in self.children:
            best_sack = child.new_traverse(configs, best_sack = best_sack)

        return best_sack

    def generate_prints(self):
        ''' generates cool tree prints '''
        if self.depth <= 15:
            prints = False
        else: 
            prints = False
        if prints:
            ## PRINT STATEMENTS ##
            s_start = '\t |' * self.depth + "--"
            #s = s_start + str(self) + " with " + str(len(self.future_events)) + " future events"
            s = s_start + str(len(self.future_events)) + " future events"
            print(s)
            #######################
    def check_is_bisimilar(self, configs):
        #print("I am here")
        strategy_set = set()
        for es in self.event_spaces:
            strategy_set = strategy_set.union(es)
        strategic_rm = bs.get_strategy_rm(configs.rm, strategy_set, full_removal = True)
      
        rms = []
        
        for es in self.event_spaces:
            rm_p = bs.project_rm(es, strategic_rm)  #project each reward machine down onto the event spaces
            rms.append(rm_p)

        rm_parallel = bs.put_many_in_parallel(rms) 
        return bs.is_bisimilar(rm_parallel,strategic_rm)

    def traverse_last_minute_change(self, configs, best_sacks = [(0, {})], num_solutions=1):
        self.generate_prints()

        self.run_check_last_minute_2(configs)
        #print(f"new stats are: value is {self.value}, is valid {self.is_valid}, is doomed is {self.doomed}.")
        
        if not self.doomed:
            if self.future_events:  # Make children if I can
                next_event_name  = self.future_events[0]
                next_events = self.future_events[1:]
                new_depth = self.depth + 1
                new_knapsack = self.knapsack.union({next_event_name}) # will only be used for child_1

                # Add children
                child_1 = Node(name = next_event_name, value = 1, knapsack = new_knapsack, future_events = next_events,  depth = new_depth, is_parent_valid= self.is_valid)
                child_0 = Node(name = next_event_name, value = 0, knapsack = self.knapsack, future_events = next_events,  depth = new_depth, is_parent_valid= self.is_valid)
                self.add_children([child_1, child_0])
            else:
                if self.is_valid: 
                    knap_score = configs.get_score(self.knapsack)
                    if knap_score > best_sacks[0][0] or len(best_sacks) < num_solutions: #fill up the sack
                        if self.check_is_bisimilar(configs):                            
                            best_sacks.append((knap_score, self.knapsack)) # max length of num_solutions
                            if len(best_sacks) > num_solutions:
                                best_sacks.sort(key = lambda x: x[0])
                                best_sacks.pop(0)
        
        ### Recursion Step ###
        for child in self.children:
            best_sacks = child.traverse_last_minute_change(configs, best_sacks = best_sacks, num_solutions=num_solutions)
        
        return best_sacks







        

