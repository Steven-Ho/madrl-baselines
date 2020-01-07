import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, max_episode_len):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.max_episode_len = max_episode_len

    # push once when an episode ends
    def push(self, data_tuples):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data_tuples
        self.position = (self.position + 1) % self.capacity

    def sample(self, num_episodes):
        batch = random.sample(self.buffer, num_episodes)
        return batch

    def __len__(self):
        return len(self.buffer)