import cv2
from collections import deque
import torch
import numpy as np
import torch.nn.functional as F

FRAME_RESIZE = 84

class ReplayBuffer:
    def __init__(self, horizon, obs_shape, action_shape, capacity=50000, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = torch.zeros((capacity, FRAME_RESIZE, FRAME_RESIZE), dtype=torch.uint8)
        self.action = torch.zeros((capacity, action_shape), dtype=torch.int64)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_obs = torch.zeros((capacity, FRAME_RESIZE, FRAME_RESIZE), dtype=torch.uint8)
        self.done = torch.zeros((capacity, 1), dtype=torch.bool)

        self.obs_shape = (FRAME_RESIZE, FRAME_RESIZE)
        self.action_shape = action_shape
        self.horizon = horizon

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, done):
        obs = torch.from_numpy(preprocess_frame(obs))
        next_obs = torch.from_numpy(preprocess_frame(next_obs))

        self.obs[self.ptr] = obs
        self.action[self.ptr] = F.one_hot(torch.tensor(action), self.action_shape)
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_obs_sequence(self, idx):
        start = idx - self.horizon + 1
        end = idx + 1

        if start < 0:
            obs = torch.zeros((self.horizon, *self.obs_shape), dtype=torch.uint8)
            next_obs = torch.zeros((self.horizon, *self.obs_shape), dtype=torch.uint8)
            dones = torch.zeros((self.horizon, 1), dtype=torch.bool)
            if self.size == self.capacity:
                obs[:abs(start)] = self.obs[start:]
                obs[abs(start):] = self.obs[:end]
                next_obs[:abs(start)] = self.obs[start:]
                next_obs[abs(start):] = self.obs[:end]                
                dones[:abs(start)] = self.done[start:]
                dones[abs(start):] = self.done[:end]
            else:
                obs[:abs(start)] = self.obs[0]
                obs[abs(start):] = self.obs[:end]
                next_obs[:abs(start)] = self.obs[0]
                next_obs[abs(start):] = self.obs[:end]
                dones[:abs(start)] = self.done[0]
                dones[abs(start):] = self.done[:end]
        else:
            obs = self.obs[start:end]
            next_obs = self.next_obs[start:end]
            dones = self.done[start:end]

        cut = torch.nonzero(dones.flatten())
        if cut.shape[0] > 0:
            if cut[-1] == self.horizon - 1:
                next_obs[cut[-1]] = next_obs[cut[-1]-1]
            else:
                obs[:cut[-1]+1] = obs[cut[-1]+1]
                next_obs[:cut[-1]] = next_obs[cut[-1]]

        return obs, next_obs
            

    def sample(self, batch_size, norm_img_obs=True):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obss = []
        next_obss = []
        
        # TODO
        for idx in idxs:
            obs, next_obs = self._get_obs_sequence(idx)
            obss.append(obs)
            next_obss.append(next_obs)
        
        obss = torch.stack(obss, dim=0).to(torch.float32)
        next_obss = torch.stack(next_obss, dim=0).to(torch.float32)

        if norm_img_obs:
            obss /= 255.0
            next_obss /= 255.0

        batch = {
            'obs': obss,
            'action': self.action[idxs],
            'reward': self.reward[idxs],
            'next_obs': next_obss,
            'done': self.done[idxs], 
        }
        return batch

class Multistep_Wrapper:

    def __init__(self, horizon, obs_shape, device='cpu'):
        self.obss = torch.zeros((horizon, FRAME_RESIZE, FRAME_RESIZE), device=device, dtype=torch.float32)
        self.horizon = horizon
        self.obs_shape = (FRAME_RESIZE, FRAME_RESIZE)
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

class SkipFrame_Wrapper:
    def __init__(self, skip):
        """Return only every `skip`-th frame"""
        self.cnt = 0
        self._skip = skip
        self.action = None

    def can_set_action(self):
        res = (self.cnt % self._skip == 0)
        self.cnt += 1
        return res
    
    def set_action(self, action):
        self.action = action

    def reset(self):
        self.cnt = 0

from torchvision import transforms as T
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # # Resize to 84x84
    resized = cv2.resize(gray, (FRAME_RESIZE, FRAME_RESIZE), interpolation=cv2.INTER_AREA)
    # Normalize to [0, 1] and convert to float32
    return resized.astype(np.uint8)

class RewardShaper:
    def __init__(self):
        self.prev_info = None

    def reset(self):
        self.prev_info = None

    def compute_reward(self, info):
        if self.prev_info is None:
            self.prev_info = info
            return 0.0

        reward = 0.0

        # 2. Coin collection
        dcoins = info["coins"] - self.prev_info["coins"]
        reward += 1.0 * dcoins

        # 3. Score increase
        dscore = info["score"] - self.prev_info["score"]
        reward += 0.01 * dscore  # scaled down to prevent high variance

        # 5. Status improvement
        status_values = {"small": 0, "tall": 1, "fireball": 2}
        dstatus = status_values[info["status"]] - status_values[self.prev_info["status"]]
        reward += 2.0 * dstatus

        # 6. Reached the flag
        if info["flag_get"] and not self.prev_info["flag_get"]:
            reward += 50.0

        # 8. Stage/world progression reward
        world_stage_prev = self.prev_info["world"] * 10 + self.prev_info["stage"]
        world_stage_curr = info["world"] * 10 + info["stage"]
        dstage = world_stage_curr - world_stage_prev
        if dstage > 0:
            reward += 30.0 * dstage

        # 7. Life lost penalty
        self.prev_info = info
        return reward
