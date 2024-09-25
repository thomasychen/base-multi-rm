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
        
        self.curr_assignment = list(np.random.permutation([i for i in range(num_agents)]))
        self.curr_decomp = 0
        self.assignment_method = assignment_method
        self.num_agents = num_agents
        self.num_decomps = num_decomps
        self.curr_permutation_qs = [{} for i in range(self.num_decomps)]
        # self.decomp_counts = {}
        # for i in range(self.num_decomps):
        #     for perm in itertools.permutations(range(num_agents)):
        #         self.decomp_counts[f"{i}_{perm}"] = 0
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.wandb = wandb

        ### UCB Specific ####

        self.permutation_counts = [{perm: 0 for perm in itertools.permutations(range(num_agents))} for i in range(self.num_decomps)]
        # print(self.permutation_counts)
        self.permutation_total_rewards = [{perm: 0.0 for perm in itertools.permutations(range(num_agents))} for i in range(self.num_decomps)]
        self.permutation_curr_rewards = [{perm: 0.0 for perm in itertools.permutations(range(self.num_agents))} for i in range(self.num_decomps)]

        # print(self.permutation_total_rewards)
        # print("HELLO", self.permutation_counts)
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
        
        # if self.wandb:
        #     if not test: 
        #         for perm in self.curr_permutation_qs:
        #             wandb.log({f"Score for {perm}": self.curr_permutation_qs[perm]})

        if test and self.assignment_method != "naive":
            return self.curr_assignment, self.curr_decomp
        elif test and self.assignment_method == "naive":
            for i in range(len(init_rm_states)):
                self.curr_permutation_qs[i] = self.calculate_permutation_qs(init_mdp_states, init_rm_states[i], True)

            self.set_best_assignment()
            return self.curr_assignment, self.curr_decomp
        elif self.assignment_method == "ground_truth":
            self.curr_assignment = list(range(self.num_agents))
            self.curr_decomp = 0
        elif self.assignment_method == "random" or self.assignment_method == "naive": 
            self.set_random_assignment()
        elif self.assignment_method == "add":
            for i in range(len(init_rm_states)):
                self.curr_permutation_qs[i] = self.calculate_permutation_qs(init_mdp_states, init_rm_states[i], False)

            if random.random() < self.epsilon:
                self.set_random_assignment()
            else:
                self.set_best_assignment()
    
            self.epsilon *= self.epsilon_decay
        elif self.assignment_method == "multiply":
            for i in range(len(init_rm_states)):
                # print(i)
                self.curr_permutation_qs[i] = self.calculate_permutation_qs(init_mdp_states, init_rm_states[i], True)

            if random.random() < self.epsilon:
                self.set_random_assignment()
            else:
                self.set_best_assignment()
            self.epsilon *= self.epsilon_decay

        elif self.assignment_method == "UCB":

            # OVERRIDE WITH UCB SCORES
            ucb_values = [{perm: self.calculate_ucb_value(perm, i) for perm in self.permutation_counts[i].keys()} for i in range(len(self.permutation_counts))]
            self.curr_permutation_qs = ucb_values 


            if self.total_selections < len(self.permutation_counts)*len(self.permutation_counts[0]):
                # Ensure each permutation is selected at least once in the beginning
                # print("IN UCB", self.total_selections, len(self.permutation_counts), len(self.permutation_counts[0]))
                decomp_idx = self.total_selections // len(self.permutation_counts[0])
                # print(decomp_idx)
                assign_idx = self.total_selections % len(self.permutation_counts[0])
                self.curr_assignment = list(self.permutation_counts[decomp_idx].keys())[assign_idx]
                self.curr_decomp = decomp_idx
                
            else:
                # Calculate UCB value for each permutation and select the one with the highest UCB value
                # ucb_values = {perm: self.calculate_ucb_value(perm) for perm in self.permutation_counts.keys()}
                best_decomp = None
                best_score = float("-inf")
                for decomp in range(len(self.curr_permutation_qs)):
                    if max(ucb_values[decomp].values()) > best_score:
                        best_decomp = decomp
                        best_score = max(ucb_values[decomp].values())
                self.curr_assignment = list(max(ucb_values[best_decomp], key=ucb_values[best_decomp].get))
                self.curr_decomp = best_decomp

            # Update counts and total selections
            perm_tuple = tuple(self.curr_assignment)
            self.permutation_counts[self.curr_decomp][perm_tuple] += 1
            self.total_selections += 1
        else:
            raise Exception("STUPID ASS MF")

        # print(self.curr_permutation_qs)
        print(self.curr_assignment, "decomp_idx", self.curr_decomp)

        
        if self.wandb:
            for i in range(len(self.permutation_counts)):
                for perm in self.permutation_counts[i]:
                    wandb.log({f"selection rate for {i}_{perm}": self.permutation_counts[i][perm] / self.total_selections})
                    wandb.log({f"reward for {i}_{perm}": self.permutation_curr_rewards[i][perm]})


        return self.curr_assignment, self.curr_decomp


    def calculate_permutation_qs(self, init_mdp_states, init_rm_states, multiply=False):
        res = {}
        for permutation in itertools.permutations(list(range(self.num_agents))):
            accumulator = 1 if multiply else 0

            for i in range(len(permutation)):
                # starting_rm_state = self.start_nodes[permutation[i]]
                # curr_state = np.row_stack(([agent_list[i].s_i], [starting_rm_state])).T
                # curr_state = np.append(init_mdp_states[i], init_rm_states[permutation[i]])
                if np.isscalar(init_mdp_states[i]) and np.isscalar(init_rm_states[permutation[i]]):
                    # Convert scalars to 1D arrays and concatenate
                    curr_state = np.array([[init_mdp_states[i], init_rm_states[permutation[i]]]])
                else:
                    # Concatenate the 1D lists or arrays
                    curr_state = np.append(init_mdp_states[i], init_rm_states[permutation[i]])
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
            
            res[tuple(permutation)] = accumulator
        return res
    
    def set_best_assignment(self):

        best_decomp = None
        best_score = float('-inf')
        
        for decomp in range(len(self.curr_permutation_qs)):
            if max(self.curr_permutation_qs[decomp].values()) > best_score:
                best_decomp = decomp
                best_score = max(self.curr_permutation_qs[decomp].values())

        self.curr_decomp = best_decomp
        self.curr_assignment = list(max(self.curr_permutation_qs[best_decomp], key=self.curr_permutation_qs[best_decomp].get))

    def set_random_assignment(self):
        decomp = random.choice(list(range(self.num_decomps)))
        self.curr_decomp = decomp
        self.curr_assignment = list(random.choice(list(itertools.permutations(list(range(self.num_agents))))))
    
    
    ### FOR UCB ###
    def update_rewards(self, reward):
        # Update the total reward for a permutation after an assignment is completed
        self.permutation_total_rewards[self.curr_decomp][tuple(self.curr_assignment)] += reward
        self.permutation_curr_rewards = [{perm: 0.0 for perm in itertools.permutations(range(self.num_agents))} for i in range(self.num_decomps)]
        self.permutation_curr_rewards[self.curr_decomp][tuple(self.curr_assignment)] = reward
    
    def calculate_ucb_value(self, permutation, decomp):
        # Calculate the UCB value for a given permutation
        if self.permutation_counts[decomp][permutation] == 0:
            return float('inf')  # Represents a strong incentive to select this permutation
        
        average_reward = self.permutation_total_rewards[decomp][permutation] / self.permutation_counts[decomp][permutation]
        confidence = np.sqrt((2 * np.log(self.total_selections)) / self.permutation_counts[decomp][permutation])
        return average_reward + self.ucb_c * confidence
    
    def get_curr_assignment(self):
        return self.curr_assignment
                

    