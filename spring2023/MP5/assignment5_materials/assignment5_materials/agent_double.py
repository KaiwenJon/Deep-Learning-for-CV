import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size
        
        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net and the target net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        # Initialize a target network and initialize the target network to the policy net
        ### CODE ###

        self.target_net = DQN(action_size)
        self.target_net.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    def load_policy_net(self, path):
        self.policy_net = torch.load(path)           

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        ### CODE ###
        self.target_net.load_state_dict(self.policy_net.state_dict())


    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            ### CODE #### 
            # Choose a random action
            a = np.random.randint(self.action_size)
            pass
        else:
            ### CODE ####
            # Choose the best action
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state)
                a = q_values.argmax().item()
            pass
        return a

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8).cuda()
        
        # Your agent.py code here with double DQN modifications
        ### CODE ###
        # print("history.shape", history.shape)
        # print("states.shape", states.shape)
        # print("actions.shape", actions.shape)
        # print("reward.shape", rewards.shape)
        # print("next states shape", next_states.shape)
        # print("Mask shape", mask.shape)


        # Compute Q(s_t, a), the Q-value of the current state
        ### CODE ####
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).cuda()
        # print("Q values shape", q_values.shape)

        # Compute Q function of next state
        ### CODE ####
        next_states = torch.from_numpy(next_states).cuda()
        next_q_values = torch.zeros((batch_size, 1), device=device)
        non_final_next_states = torch.cat([s for s in next_states if s is not None]).cuda()
        non_final_next_states = non_final_next_states.view(-1, 4, WIDTH, HEIGHT)
        # print("non_final_next_states.shape", non_final_next_states.shape)
        with torch.no_grad():
          mask = mask.unsqueeze(1)
          next_action = self.policy_net(non_final_next_states).max(1)[1]
        #   print("next_action.shape", next_action.shape)
          next_q_values[mask] = self.target_net(non_final_next_states).gather(1, next_action.unsqueeze(1)).cuda()[mask]
        #   print("updated mext states values shape", next_q_values.shape)
        
        # Compute the target Q value
        next_q_values = next_q_values.squeeze()
        expected_state_action_values = (next_q_values * self.discount_factor) + rewards
        # print("expected_state_action_values.shape", expected_state_action_values.shape)
    
        # Compute the loss between the predicted and target Q values
        loss = F.smooth_l1_loss(q_values, expected_state_action_values.unsqueeze(1))



        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.scheduler.step()

     
        