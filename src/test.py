#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import cv2

import gym
# from IPython import display
import matplotlib.pyplot as plt

env = gym.make('Breakout-v0')
env.reset()

for _ in range(1000):
    env.step(env.action_space.sample())
    # plt.figure()
    # plt.imshow(env.render(mode='rgb_array'))
    size = 200, 200
    frame = np.zeros(size, dtype=np.uint8)
    cv2.imshow("test", frame)
    cv2.waitKey(30)
    # plt.show()
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
