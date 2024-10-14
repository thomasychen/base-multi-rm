import numpy as np
import torch
import random
import itertools
from stable_baselines3.common.utils import obs_as_tensor
import wandb
import math

class Manager:
    def __init__(self, num_agents, num_decomps=1, assignment_method = "ground_truth", model=None, wandb=False, seed=None, ucb_c=1.5):
        if seed:
            random.seed(seed)
        
        self.curr_decomp = 0
        self.assignment_method = assignment_method
        self.num_agents = num_agents
        self.num_decomps = num_decomps
        self.curr_decomp_qs = {i: 0.0 for i in range(self.num_decomps)}

        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.wandb = wandb

        ### UCB Specific ####

        self.decomp_counts = {i: 0 for i in range(self.num_decomps)}
        self.decomp_total_rewards = {i: 0.0 for i in range(self.num_decomps)}
        self.decomp_curr_rewards = {i: 0.0 for i in range(self.num_decomps)}


        self.total_selections = 0

        # UCB exploration parameter
        self.ucb_c = ucb_c
        self.ucb_gamma = 0.99
        self.window = 0
        self.window_cnt = 0

    def set_model(self, model):
        self.model = model

    def get_rm_assignments(self, init_mdp_states, init_rm_states, test=False):
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

            ucb_values = {i: self.calculate_ucb_value(i) for i in range(self.num_decomps)}
            self.curr_decomp_qs = ucb_values 

            if self.total_selections < len(self.decomp_counts):

                decomp_idx = self.total_selections
                self.curr_decomp = decomp_idx
                
            else:
                best_decomp = None
                best_score = float("-inf")
                for decomp in range(len(self.curr_decomp_qs)):
                    if ucb_values[decomp] > best_score:
                        best_decomp = decomp
                        best_score = ucb_values[decomp]
                self.curr_decomp = best_decomp

            self.decomp_counts[self.curr_decomp] += 1
        else:
            raise Exception("Invalid assignment method")

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
                curr_state = np.array([[init_mdp_states[i], init_rm_states[i]]])
            else:
                curr_state = np.append(init_mdp_states[i], init_rm_states[i])

            curr_state = obs_as_tensor(curr_state, device="cpu")
            with torch.no_grad():
                q = self.model.policy.predict_values(curr_state)

            if multiply:
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
        self.decomp_total_rewards[self.curr_decomp] = self.decomp_total_rewards[self.curr_decomp] * self.ucb_gamma + reward
        self.decomp_curr_rewards = {i: 0.0 for i in range(self.num_decomps)}
        self.decomp_curr_rewards[self.curr_decomp] = reward
    
    def calculate_ucb_value(self, decomp):
        if self.decomp_counts[decomp] == 0:
            return float('inf')
        
        average_reward = self.decomp_total_rewards[decomp]/ self.decomp_counts[decomp]
        confidence = np.sqrt((2 * np.log(self.total_selections)) / self.decomp_counts[decomp])
        return average_reward + self.ucb_c * confidence
    