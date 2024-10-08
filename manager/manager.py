import numpy as np
# import infrastructure.pytorch_utils as ptu
import torch
import random
import itertools
from stable_baselines3.common.utils import obs_as_tensor
import wandb
import math

class Manager:
    def __init__(self, num_agents, num_decomps=1, assignment_method = "ground_truth", model=None, wandb=False, seed=None):
        if seed:
            random.seed(seed)
        
        self.curr_decomp = 0
        self.assignment_method = assignment_method
        self.num_agents = num_agents
        self.num_decomps = num_decomps
        self.curr_decomp_qs = {i: 0.0 for i in range(self.num_decomps)}
        # self.decomp_counts = {}

        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.wandb = wandb

        ### UCB Specific ####

        self.decomp_counts = {i: 0 for i in range(self.num_decomps)}

        self.decomp_total_rewards = {i: 0.0 for i in range(self.num_decomps)}
        self.decomp_curr_rewards = {i: 0.0 for i in range(self.num_decomps)}


        self.total_selections = 0

        # UCB exploration parameter
        self.ucb_c = 0.69
        self.window = 0
        self.window_cnt = 0

    def set_model(self, model):
        self.model = model

    # init_rm_states is a list of the rm states of each decomp
    def get_rm_assignments(self, init_mdp_states, init_rm_states, test=False):
        # import pdb; pdb.set_trace();
        # self.window_cnt += 1
        # if self.window_cnt % self.window != 0:
        #     return self.curr_assignment
        

        if test and self.assignment_method != "naive":
            return self.curr_decomp
        elif test and self.assignment_method == "naive":
            for i in range(len(init_rm_states)):
                self.curr_decomp_qs[i] = self.calculate_decomp_qs(init_mdp_states, init_rm_states[i], True)

            self.set_best_assignment()
            return self.curr_decomp
        elif self.assignment_method == "ground_truth":
            self.curr_decomp = 0
        elif self.assignment_method == "random" or self.assignment_method == "naive": 
            self.set_random_assignment()
        elif self.assignment_method == "add":
            for i in range(len(init_rm_states)):
                self.curr_decomp_qs[i] = self.calculate_decomp_qs(init_mdp_states, init_rm_states[i], True)

            if random.random() < self.epsilon:
                self.set_random_assignment()
            else:
                self.set_best_assignment()
    
            self.epsilon *= self.epsilon_decay
        elif self.assignment_method == "multiply":
            for i in range(len(init_rm_states)):
                # print(i)
                self.curr_decomp_qs[i] = self.calculate_decomp_qs(init_mdp_states, init_rm_states[i], True)

            if random.random() < self.epsilon:
                self.set_random_assignment()
            else:
                self.set_best_assignment()
            self.epsilon *= self.epsilon_decay

        elif self.assignment_method == "UCB":

            # OVERRIDE WITH UCB SCORES
            ucb_values = {i: self.calculate_ucb_value(i) for i in range(self.num_decomps)}
            self.curr_decomp_qs = ucb_values 


            if self.total_selections < len(self.decomp_counts):

                decomp_idx = self.total_selections
                # print(decomp_idx)
                self.curr_decomp = decomp_idx
                
            else:
                # Calculate UCB value for each decomp and select the one with the highest UCB value
                best_decomp = None
                best_score = float("-inf")
                for decomp in range(len(self.curr_decomp_qs)):
                    if ucb_values[decomp] > best_score:
                        best_decomp = decomp
                        best_score = ucb_values[decomp]
                self.curr_decomp = best_decomp

            # Update counts and total selections
            self.decomp_counts[self.curr_decomp] += 1
        else:
            raise Exception("Invalid assignment method")


        print("decomp_idx", self.curr_decomp)

        perm_tuple = tuple(self.curr_assignment)
        self.permutation_counts[self.curr_decomp][perm_tuple] += 1
        self.total_selections += 1
        
        if self.wandb:
            for i in range(len(self.curr_decomp_qs)):
                wandb.log({f"selection rate for {i}": self.decomp_counts[i] / self.total_selections})
                wandb.log({f"reward for {i}": self.decomp_curr_rewards[i]})


        return self.curr_decomp


    def calculate_decomp_qs(self, init_mdp_states, init_rm_states, multiply=False):
        res = {}
        accumulator = 1 if multiply else 0

        for i in range(self.num_agents):

            if np.isscalar(init_mdp_states[i]) and np.isscalar(init_rm_states[i]):
                # Convert scalars to 1D arrays and concatenate
                curr_state = np.array([[init_mdp_states[i], init_rm_states[i]]])
            else:
                # Concatenate the 1D lists or arrays
                curr_state = np.append(init_mdp_states[i], init_rm_states[i])
            # obs_tensor = torch.tensor(curr_state, dtype=torch.float32).unsqueeze(0)

            # import pdb; pdb.set_trace()

            curr_state = obs_as_tensor(curr_state, device="cpu")
            with torch.no_grad():
                # q_values = self.model.q_net(curr_state)
                # print(self, "IN MANAGER")
                q = self.model.policy.predict_values(curr_state)

            # q, max_action = torch.max(q_values, dim=1)
            # q_min = q_values.min(dim=1, keepdim=True)[0]
            # q_max = q_values.max(dim=1, keepdim=True)[0]
            # q_normalized = (q_values - q_min) / (q_max - q_min)
            # q = torch.mean(q_normalized, dim=1)

            # q = torch.mean(q_values, dim=1)
            if multiply:
                # q = max(0, q) # cuz it can be negative twice
                q = 1 / (1 + math.exp(-q))


            if multiply:
                accumulator *= q
            else:
                accumulator += q
        
        res = accumulator
        return res
    
    def set_best_assignment(self):

        best_decomp = None
        best_score = float('-inf')
        
        for decomp in range(len(self.curr_decomp_qs)):
            if self.curr_decomp_qs[decomp] > best_score:
                best_decomp = decomp
                best_score = self.curr_decomp_qs[decomp]

        self.curr_decomp = best_decomp

    def set_random_assignment(self):
        decomp = random.choice(list(range(self.num_decomps)))
        self.curr_decomp = decomp
    
    
    ### FOR UCB ###
    def update_rewards(self, reward):
        # Update the total reward for a decomposition after an assignment is completed
        self.decomp_total_rewards[self.curr_decomp] += reward
        self.decomp_curr_rewards = {i: 0.0 for i in range(self.num_decomps)}
        self.decomp_curr_rewards[self.curr_decomp] = reward
    
    def calculate_ucb_value(self, decomp):
        # Calculate the UCB value for a given decomposition
        if self.decomp_counts[decomp] == 0:
            return float('inf')  # Represents a strong incentive to select this decomposition
        
        average_reward = self.decomp_total_rewards[decomp]/ self.decomp_counts[decomp]
        confidence = np.sqrt((2 * np.log(self.total_selections)) / self.decomp_counts[decomp])
        return average_reward + self.ucb_c * confidence
    