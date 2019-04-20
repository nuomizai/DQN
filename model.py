import torch.nn as nn
from torch.nn import init
import numpy as np
import torch
import random


class Q_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        h_size_1 = 128
        h_size_2 = 128
        self.q_val = nn.Sequential(
            nn.Linear(state_dim, h_size_1),
            nn.ReLU(),
            nn.Linear(h_size_1, h_size_2),
            nn.ReLU(),
            nn.Linear(h_size_2, action_dim)
        )
        self.start_epsilon = 0.4
        self.end_epsilon = 0.99
        self.exploration_step = 10000
        self.epsilon = self.start_epsilon
        # init parameters
        for m in self.q_val:
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)
        self.train()

    def forward(self, state):
        return self.q_val(state)

    def get_value(self, state, action):
        Q_value = self.forward(state)
        identity = torch.tensor([[1., 0], [0., 1.]])
        index = identity[action]
        Q_s_a = torch.sum(Q_value * index, -1)
        return Q_s_a

    def epsilon_greedy(self, state):
        # print(state)
        action_val = self.forward(state)
        if self.epsilon < self.end_epsilon:
            self.epsilon = self.epsilon + (self.end_epsilon - self.start_epsilon) / self.exploration_step
        # print('-------epsilon:', self.epsilon)
        if np.random.random() < self.epsilon:  # choose max Q action
            action = torch.argmax(action_val).numpy()
        else:
            action = np.random.randint(0, self.action_dim)  # random choose
        return action

    def get_max_action(self, state):
        action_val = self.forward(state)
        max_action = torch.argmax(action_val, dim=-1)
        max_action = max_action.numpy()
        return max_action

    def get_max_Q(self, state):
        action_val = self.forward(state)
        max_Q = torch.max(action_val, dim=-1)[0]
        return max_Q


class Replay_memory(nn.Module):
    def __init__(self, num_steps, state_dim, mini_batch_size):
        super(Replay_memory, self).__init__()
        self.action = torch.zeros(num_steps, dtype=torch.long)
        self.state = torch.zeros(num_steps + 1, state_dim)
        self.next_state = torch.zeros(num_steps + 1, state_dim)
        self.reward = torch.zeros(num_steps)
        self.masks = torch.zeros(num_steps + 1)
        # self.value = torch.zeros(num_steps)
        self.num_steps = num_steps
        self.mini_batch_size = mini_batch_size
        self.step = 0

    def to(self, device):
        self.state = self.state.to(device)
        self.action = self.action.to(device)
        self.reward = self.reward.to(device)
        self.masks = self.masks.to(device)
        self.next_state = self.next_state.to(device)

    def reset(self, state):
        mask = torch.tensor(0.0)
        self.state[self.step].copy_(state)
        self.masks[self.step].copy_(mask)

    def insert(self, state, action, reward, mask, next_state):
        self.state[self.step].copy_(state)
        self.masks[self.step].copy_(mask)
        self.action[self.step].copy_(torch.tensor(action))
        self.reward[self.step].copy_(torch.tensor(reward))
        self.next_state[self.step].copy_(next_state)
        self.step = (self.step + 1) % self.num_steps
        # self.value [self.step].copy_(value)

    def sample(self, sum_cnt):
        step = min(self.num_steps - 1, sum_cnt)
        index = random.sample(range(step), self.mini_batch_size)
        state_batch = self.state[index]
        action_batch = self.action[index]
        reward_batch = self.reward[index]
        masks_batch = self.masks[index]
        next_state_batch = self.next_state[index]
        return state_batch, action_batch, reward_batch, next_state_batch, masks_batch