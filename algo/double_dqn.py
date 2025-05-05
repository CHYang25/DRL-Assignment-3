from .base_dqn import Base_DQN_Agent
from .model import QNet
from .utils import Multistep_Wrapper, SkipFrame_Wrapper, preprocess_frame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import random

class Double_DQN_Agent(Base_DQN_Agent):

    def __init__(self,
            in_channels,
            feature_dim,
            n_actions,
            n_hidden,
            horizon,
            obs_shape,
            lr,
            device,
            skip,
            gamma=0.99,
            tau=1e-3):
        
        super().__init__()
        self.model = QNet(in_channels, feature_dim, n_actions, n_hidden).to(device)
        self.model_target = QNet(in_channels, feature_dim, n_actions, n_hidden).to(device)
        self.multistep_wrapper = Multistep_Wrapper(horizon, obs_shape)
        self.skip_frame_wrapper = SkipFrame_Wrapper(skip)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.multistep_wrapper.reset()
        self.skip_frame_wrapper.reset()
        self.action_list = list(range(n_actions))

    def act(self, observation, epsilon=0.0):
        if self.skip_frame_wrapper.can_set_action():
            if epsilon < random.uniform(0, 1):
                with torch.no_grad():
                    observation = self.multistep_wrapper.stack(preprocess_frame(observation)).to(self.device)
                    out = self.model(observation.unsqueeze(dim=0)).squeeze(dim=0)
                    action = torch.argmax(out).cpu().item()
            else:
                action = random.choice(self.action_list)

            self.skip_frame_wrapper.set_action(action)

        return self.skip_frame_wrapper.action

    def train(self, batch, update_target = None):
        obs_batch = batch['obs'].to(self.device)
        action_batch = batch['action'].to(self.device)
        reward_batch = batch['reward'].to(self.device)
        next_obs_batch = batch['next_obs'].to(self.device)
        done_batch = batch['done'].to(self.device)

        next_state_values = torch.zeros(action_batch.shape, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions = self.model(next_obs_batch).argmax(dim=1, keepdim=True)  # Keep as indices
            next_state_values = self.model_target(next_obs_batch).gather(1, next_actions)

        state_action_values = self.model(obs_batch).gather(1, action_batch)
        expected_state_action_values = reward_batch + (next_state_values * self.gamma) * (1 - done_batch.float())
        # TODO: Compute loss and update the model
        loss = self.loss_fn(state_action_values.float(), expected_state_action_values.float())

        # TODO: Update target network periodically
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        if update_target is not None and update_target:
            self.model_target.load_state_dict(self.model.state_dict())
        else:
            q_target_network_state_dict = self.model_target.state_dict()
            q_network_state_dict = self.model.state_dict()
            for key in q_network_state_dict:
                q_target_network_state_dict[key] = q_network_state_dict[key]*self.tau + q_target_network_state_dict[key]*(1-self.tau)

            self.model_target.load_state_dict(q_target_network_state_dict)
        return loss.cpu().detach().item()