import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, max_episode_len, num_agents, obs_shape, action_shape):
        self.capacity = capacity
        self.length = 0
        self.episode = 0
        self.t = 0
        self.max_episode_len = max_episode_len
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer = dict()
        self.buffer['obs'] = np.zeros((capacity, max_episode_len, num_agents, obs_shape), dtype=np.float32)
        self.buffer['actions'] = np.zeros((capacity, max_episode_len, num_agents, action_shape), dtype=np.float32)
        self.buffer['reward'] = np.zeros((capacity, max_episode_len), dtype=np.float32)
        self.buffer['obs_next'] = np.zeros((capacity, max_episode_len, num_agents, obs_shape), dtype=np.float32)
        self.buffer['mask'] = np.zeros((capacity, max_episode_len), dtype=np.float32)
        self.buffer['done'] = np.zeros((capacity, max_episode_len), dtype=np.float32)

    # push once per step
    def push(self, obs, actions, reward, obs_next, done):
        self.buffer['obs'][self.episode][self.t] = obs
        self.buffer['actions'][self.episode][self.t] = actions
        self.buffer['reward'][self.episode][self.t] = reward
        self.buffer['obs_next'][self.episode][self.t] = obs_next
        self.buffer['mask'][self.episode][self.t] = 1.
        self.buffer['done'][self.episode][self.t] = done
        self.t += 1

    def end_episode(self):
        if self.length < self.capacity:
            self.length += 1
        self.episode = (self.episode + 1) % self.capacity
        self.t = 0
        # clear previous data
        self.buffer['obs'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.obs_shape), dtype=np.float32)
        self.buffer['actions'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
        self.buffer['reward'][self.episode] = np.zeros((self.max_episode_len,), dtype=np.float32)
        self.buffer['obs_next'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.obs_shape), dtype=np.float32)
        self.buffer['mask'][self.episode] = np.zeros((self.max_episode_len,), dtype=np.float32)
        self.buffer['done'][self.episode] = np.zeros((self.max_episode_len,), dtype=np.float32)

    def sample(self, num_episodes):
        assert num_episodes <= self.length
        batch_indices = np.random.choice(self.length, size=num_episodes, replace=False)
        obs_batch = np.take(self.buffer['obs'], batch_indices, axis=0)
        actions_batch = np.take(self.buffer['actions'], batch_indices, axis=0)
        reward_batch = np.take(self.buffer['reward'], batch_indices, axis=0)
        obs_next_batch = np.take(self.buffer['obs_next'], batch_indices, axis=0)
        mask_batch = np.take(self.buffer['mask'], batch_indices, axis=0)
        done_batch = np.take(self.buffer['done'], batch_indices, axis=0)

        return obs_batch, actions_batch, reward_batch, obs_next_batch, mask_batch, done_batch

    def __len__(self):
        return self.length