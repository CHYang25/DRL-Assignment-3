import cv2
from collections import deque
import torch
import numpy as np
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, horizon, obs_shape, action_shape, capacity=50000, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = torch.zeros((capacity, horizon, *obs_shape), dtype=torch.float32)
        self.action = torch.zeros((capacity, action_shape), dtype=torch.int64)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_obs = torch.zeros((capacity, horizon, *obs_shape), dtype=torch.float32)
        self.done = torch.zeros((capacity, 1), dtype=torch.bool)

        self.multistep_wrapper = Multistep_Wrapper(horizon, obs_shape, device)
        self.multistep_wrapper_next = Multistep_Wrapper(horizon, obs_shape, device)
        self.obs_shape = obs_shape
        self.action_shape = action_shape

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, done):
        obs = self.multistep_wrapper.stack(preprocess_frame(obs))
        self.obs[self.ptr] = obs
        self.action[self.ptr] = F.one_hot(torch.tensor(action), self.action_shape)
        self.reward[self.ptr] = reward
        next_obs = self.multistep_wrapper_next.stack(preprocess_frame(next_obs))
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = {
            'obs': self.obs[idxs],
            'action': self.action[idxs],
            'reward': self.reward[idxs],
            'next_obs': self.next_obs[idxs],
            'done': self.done[idxs], 
        }
        return batch

class Multistep_Wrapper:

    def __init__(self, horizon, obs_shape, device='cpu'):
        self.obss = torch.zeros((horizon, *obs_shape), device=device, dtype=torch.float32)
        self.horizon = horizon
        self.obs_shape = obs_shape
        self.cnt = 0
    
    def stack(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)

        self.cnt += 1

        start_idx = -min(self.horizon, self.cnt)
        if start_idx == -1:
            self.obss[start_idx] = obs.unsqueeze(0)
        else:
            self.obss[start_idx:] = torch.cat((self.obss[start_idx+1:], obs.unsqueeze(0)), dim=0)
        
        if self.cnt < self.horizon:
            self.obss[:start_idx] = self.obss[start_idx]

        return self.obss
    
    def reset(self):
        self.cnt = 0

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # # Resize to 84x84
    # resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Normalize to [0, 1] and convert to float32
    return gray.astype(np.float32) / 255.0