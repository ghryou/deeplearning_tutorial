#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
from IPython import display
import matplotlib.pyplot as plt

env = gym.make('Breakout-v0')
env.reset()
for _ in range(1000):
    plt.imshow(env.render(mode='rgb_array'))
    display.clear_output(wait=True)
    display.display(plt.gcf())
    env.step(env.action_space.sample())
