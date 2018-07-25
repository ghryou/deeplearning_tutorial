#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gym

env = gym.make('Breakout-v0')
env.reset()

for _ in range(1000):
    env.step(env.action_space.sample())
    size = 200, 200
    frame = np.zeros(size, dtype=np.uint8)

    plt.figure()
    plt.imshow(frame)
    # cv2.imshow("test", cv2.cvtColor(env.render(mode='rgb_array'), cv2.COLOR_BGR2RGB))
    # cv2.waitKey(30)
    plt.show()
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
