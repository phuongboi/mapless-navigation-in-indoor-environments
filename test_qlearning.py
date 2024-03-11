import numpy as np
from collections import deque
import random
import torch
from torch import nn
import os
from env import VrepEnvironment
from matplotlib import pyplot as plt
from IPython.display import clear_output

class Network(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

class DQN:
    def __init__(self, model_path, env, lr, batch_size, gamma, eps_decay, eps_start, eps_end, initial_memory, memory_size):

        self.env = env
        self.model_path = model_path
        self.lr = lr
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.initial_memory = initial_memory

        self.replay_buffer = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.num_actions = 3
        self.input_shape = 29
        self.model = self.make_model()
        self.model.load_state_dict(torch.load(self.model_path))

    def make_model(self):
        model = Network(self.input_shape, self.num_actions)
        return model

    def agent_policy(self, state):

        q_value = self.model(torch.from_numpy(state))
        action = np.argmax(q_value.detach().numpy())
        return action

    def test(self):
        #self.model.cuda().train()
        self.model.eval()
        for episode in range(num_episodes):
            state = env.reset()
            reward_for_episode = 0
            while True:
                received_action = self.agent_policy(state)
                # print("received_action:", received_action)
                next_state, reward, terminal, info = env.step(received_action)
                print(info)
                # add up rewards
                reward_for_episode += reward
                state = next_state

                if terminal:
                    print("Episode: {} done, Reward: {}".format(episode, reward_for_episode))
                    state = env.reset()
                    break

        #env.close()





if __name__ == "__main__":
    env = VrepEnvironment(speed=1.0, turn=0.25, rate=1)

    # setting up params
    lr = 0.0001
    batch_size = 32
    eps_decay = 30000
    eps_start = 1
    eps_end = 0.1
    initial_memory = 1000
    memory_size = 5000#20 * initial_memory
    gamma = 0.99
    num_episodes = 1
    model_path = "weights103/steps_58001.pth"
    print('Start testing')
    model = DQN(model_path, env, lr, batch_size, gamma, eps_decay, eps_start, eps_end,initial_memory, memory_size)
    model.test()
