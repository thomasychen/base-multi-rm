import numpy as np
# import infrastructure.pytorch_utils as ptu
import torch
import random
import itertools
from stable_baselines3.common.utils import obs_as_tensor
import wandb

class Manager:
    def __init__(self, num_agents, assignment_method = "ground_truth", model=None, seed=None):
        if seed:
            random.seed(seed)
        
        self.curr_assignment = list(np.random.permutation([i for i in range(num_agents)]))
        self.assignment_method = assignment_method
        self.num_agents = num_agents
        self.curr_permutation_qs = {}
        self.epsilon = 1
        self.epsilon_decay = 0.999

        ### UCB Specific ####

        self.permutation_counts = {perm: 0 for perm in itertools.permutations(range(num_agents))}
        self.permutation_total_rewards = {perm: 0.0 for perm in itertools.permutations(range(num_agents))}
        # print("HELLO", self.permutation_counts)
        self.total_selections = 0

        # UCB exploration parameter
        self.ucb_c = 1.5
        self.window = 0
        self.window_cnt = 0

    def set_model(self, model):
        self.model = model


    def get_rm_assignments(self, init_mdp_states, init_rm_states, test=False):
        # self.window_cnt += 1
        # if self.window_cnt % self.window != 0:
        #     return self.curr_assignment
        self.curr_permutation_qs = self.calculate_permutation_qs(init_mdp_states, init_rm_states, True)
        if not test: 
            for perm in self.curr_permutation_qs:
                wandb.log({f"Score for {perm}": self.curr_permutation_qs[perm]})

        if test and self.assignment_method != "naive":
            return self.curr_assignment
        elif test and self.assignment_method == "naive":
            self.curr_permutation_qs = self.calculate_permutation_qs(init_mdp_states, init_rm_states, True)
            self.curr_assignment = list(max(self.curr_permutation_qs, key=self.curr_permutation_qs.get))
            return self.curr_assignment
        elif self.assignment_method == "ground_truth":
            self.curr_assignment = [0,1,2]
        elif self.assignment_method == "random" or self.assignment_method == "naive": 
            self.curr_assignment = list(random.choice(list(self.curr_permutation_qs.keys())))
        elif self.assignment_method == "add":
            self.curr_permutation_qs = self.calculate_permutation_qs(init_mdp_states, init_rm_states, True)

            if random.random() < self.epsilon:
                self.curr_assignment = list(random.choice(list(self.curr_permutation_qs.keys())))
            else:
                self.curr_assignment = list(max(self.curr_permutation_qs, key=self.curr_permutation_qs.get))
            

            self.epsilon *= self.epsilon_decay
        elif self.assignment_method == "multiply":
            self.curr_permutation_qs = self.calculate_permutation_qs(init_mdp_states, init_rm_states, True)

            if random.random() < self.epsilon:
                self.curr_assignment = list(random.choice(list(self.curr_permutation_qs.keys())))
            else:
                self.curr_assignment = list(max(self.curr_permutation_qs, key=self.curr_permutation_qs.get))
            self.epsilon *= self.epsilon_decay

        elif self.assignment_method == "UCB":

            # OVERRIDE WITH UCB SCORES
            ucb_values = {perm: self.calculate_ucb_value(perm) for perm in self.permutation_counts.keys()}
            self.curr_permutation_qs = ucb_values


            if self.total_selections < len(self.permutation_counts):
                # Ensure each permutation is selected at least once in the beginning
                self.curr_assignment = list(self.permutation_counts.keys())[self.total_selections]
            else:
                # Calculate UCB value for each permutation and select the one with the highest UCB value
                # ucb_values = {perm: self.calculate_ucb_value(perm) for perm in self.permutation_counts.keys()}
                self.curr_assignment = list(max(ucb_values, key=ucb_values.get))

            # Update counts and total selections
            perm_tuple = tuple(self.curr_assignment)
            self.permutation_counts[perm_tuple] += 1
            self.total_selections += 1
        else:
            raise Exception("STUPID ASS MF")

        # print(self.curr_permutation_qs)
        return self.curr_assignment


    def calculate_permutation_qs(self, init_mdp_states, init_rm_states, multiply=False):
        res = {}
        for permutation in itertools.permutations(list(range(self.num_agents))):
            accumulator = 1 if multiply else 0

            for i in range(len(permutation)):
                # starting_rm_state = self.start_nodes[permutation[i]]
                # curr_state = np.row_stack(([agent_list[i].s_i], [starting_rm_state])).T
                curr_state = np.array([[init_mdp_states[i], init_rm_states[permutation[i]]]])
                # obs_tensor = torch.tensor(curr_state, dtype=torch.float32).unsqueeze(0)

                # import pdb; pdb.set_trace()

                curr_state = obs_as_tensor(curr_state, device="cpu")
                with torch.no_grad():
                    q_values = self.model.q_net(curr_state)

                # q, max_action = torch.max(q_values, dim=1)
                q = torch.mean(q_values, dim=1)
                # q = 1

                if multiply:
                    accumulator *= q
                else:
                    accumulator += q
            
            res[tuple(permutation)] = accumulator
        return res
    
    
    ### FOR UCB ###
    def update_rewards(self, reward):
        # Update the total reward for a permutation after an assignment is completed
        self.permutation_total_rewards[tuple(self.curr_assignment)] += reward
    
    def calculate_ucb_value(self, permutation):
        # Calculate the UCB value for a given permutation
        if self.permutation_counts[permutation] == 0:
            return float('inf')  # Represents a strong incentive to select this permutation
        
        average_reward = self.permutation_total_rewards[permutation] / self.permutation_counts[permutation]
        confidence = np.sqrt((2 * np.log(self.total_selections)) / self.permutation_counts[permutation])
        return average_reward + self.ucb_c * confidence
    
    def get_curr_assignment(self):
        return self.curr_assignment
                

    