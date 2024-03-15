
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
from env import VrepEnvironment_SAC

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, critic_lr):
        super(QNetwork, self).__init__()

        self.fc_s = nn.Linear(state_dim, 256)
        self.fc_a = nn.Linear(action_dim, 256)
        self.fc_1 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, action_dim)

        self.lr = critic_lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, a):
        h1 = F.leaky_relu(self.fc_s(x))
        h2 = F.leaky_relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=-1)
        q = F.leaky_relu(self.fc_1(cat))
        q = self.fc_out(q)
        return q

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr):
        super(PolicyNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 512)
        self.fc_2 = nn.Linear(512, 512)
        # self.bn_1 = nn.BatchNorm1d(512)
        # self.bn_2 = nn.BatchNorm1d(512)
        self.fc_mu = nn.Linear(512, action_dim)
        self.fc_std = nn.Linear(512, action_dim)
        # self.bn_mu = nn.BatchNorm1d(action_dim)
        # self.bn_std = nn.BatchNorm1d(action_dim)


        self.lr = actor_lr

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.max_linear = 0.5
        self.min_linear = 0
        self.max_angular = 1
        self.min_angular = -1
        self.linear_scale = (self.max_linear - self.min_linear) / 2.0
        self.linear_bias = (self.max_linear + self.min_linear) / 2.0
        self.angular_scale = (self.max_angular - self.min_angular) / 2.0
        self.angular_bias = (self.max_angular + self.min_angular) / 2.0
        self.scale = torch.Tensor([self.linear_scale, self.angular_scale])
        self.bias = torch.Tensor([self.linear_bias, self.angular_bias])

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        reparameter = Normal(mean, std)
        #x_t = reparameter.rsample()
        x_t = mean
        y_t = torch.tanh(x_t)
        # action = y_t.clone()
        # action[0] = self.linear_scale * y_t[0] + self.linear_bias
        # action[1] = self.angular_scale * y_t[1] + self.angular_bias
        action = self.scale * y_t + self.bias

        # # Enforcing Action Bound
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(torch.log(self.scale * (1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True)
        #log_prob[1] = log_prob[1] - torch.sum(torch.log(self.angular_scale * (1 - y_t[1].pow(2)) + 1e-6), dim=-1, keepdim=True)

        return torch.Tensor(action), log_prob


class SAC_Agent:
    def __init__(self, weight_file_path):
        self.trained_model  = weight_file_path
        self.state_dim      = 17  # [cos(theta), sin(theta), theta_dot]
        self.action_dim     = 2  # [torque] in[-2,2]
        self.lr_pi          = 0.001
        self.DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device : ", self.DEVICE)

        self.PI  = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi).to(self.DEVICE)
        self.PI.load_state_dict(torch.load(self.trained_model))

    def choose_action(self, s):
        with torch.no_grad():
            action, log_prob = self.PI.sample(s.to(self.DEVICE))
        return action, log_prob


if __name__ == '__main__':

    log_name = '12-3/'
    weight_name = 'sac_actor_step_500000.pt'

    weight_file_path = 'weights_sac/' + log_name + weight_name
    agent = SAC_Agent(weight_file_path)

    fix_pos_list = [(0, -3.5), (4, -0.5), (0.5,-0.2)]
    env = VrepEnvironment_SAC(rate=1, is_testing=True, fix_pos=(0, 0))
    state = env.reset()
    for fix_pos in fix_pos_list:
        env.target_pos = fix_pos
        print(env.target_pos)
        step = 0
        while True:
            action, log_prob = agent.choose_action(torch.FloatTensor(state))
            action = action.detach().cpu().numpy()  # GPU
            state_prime, reward, done, info = env.step(action)
            print(info)
            step += 1
            if done:
                break
            state = state_prime
