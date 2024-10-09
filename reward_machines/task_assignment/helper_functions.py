
### generic functions to move between knapsack and agent:event_spaces dictionaries ###

def get_event_spaces_from_knapsack(all_events, knapsack = None): 
    '''
    all_events = {(event, agent)..}
    knapsack = {(e,a)}, knapsack \subset all_events

    we want the (a,e) pairs from all_events that are not in the knapsack 

    we want them in the form:
        {agent: [events]}
    and 
        [{events for agent 1}, {events for agent 2}, ... ]
    '''
    if not knapsack:
        knapsack = set()

    event_spaces_dict = {}
    remaining_events = all_events - knapsack 

    for e, a in remaining_events:
        if a not in event_spaces_dict.keys():
            event_spaces_dict[a] = [e]
        else:
            event_spaces_dict[a].append(e)
    
    event_spaces = []
    for agent, event_list in event_spaces_dict.items():
        event_spaces.append(set(event_list))

    return event_spaces, event_spaces_dict
    
def get_sack_from_dict(dict):
    '''
    takes a dict with form {a: [e1, e2]}
    and returns set: {(a,e1), (a,e2)}
    '''
    knapsack = set()
    for a, es in dict.items():
        for e in es:
            knapsack.add((e,a))
    return knapsack
        

# this is a silly print function that shows the results of a tree run 
def print_results(configs, bd):
    # bd has form = (score, [knapsacks with that score])
    s = "Kept Event Sets: \n"
    all_events = set(configs.all_events)
    for i, solution in enumerate(bd):
        score, k = solution
        event_spaces, event_spaces_dict = get_event_spaces_from_knapsack(all_events, k) # event_spaces_dict = {agent: [events] }
        labor_division = [len(x) for z, x in event_spaces_dict.items()]
        labor_division_string = ""
        for j, p in enumerate(labor_division):
            labor_division_string += str(p) 
            if j != len(labor_division) - 1: 
                labor_division_string += " vs. "

        kept_ev = all_events - k
        s_mid = str(i) + ":" + str(event_spaces_dict) + " with labor division " + labor_division_string + " \n"

        s += s_mid

    s += "\n"
    ss = "There are " + str(len(configs.future_events)) + " possible knapsack items. \n"
    
    s += ss
    sss = "There are " + str(2**(len(configs.future_events))) + " possible ways to fill a knapsack with these items. \n"
    s += sss

    ssss = "The best viable knapsack(s) have scores " + str([soln[0] for soln in bd]) + " and there are " + str(len(bd)) + " successful ways to do it.\n"
    s += ssss

    print(s)



