import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, data_tuple):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data_tuple
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*batch))

    def __len__(self):
        return len(self.buffer)
