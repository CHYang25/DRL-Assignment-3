import gym
from algo.double_dqn import Double_DQN_Agent
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import torch

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(Double_DQN_Agent):
    """Agent that acts randomly."""
    def __init__(self):
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        super().__init__(
            in_channels=4,
            feature_dim=512,
            n_actions=12,
            n_hidden=128,
            horizon=4,
            obs_shape=env.observation_space.shape[:-1],
            lr=0,
            device='cpu',
            skip=4,
            gamma=0.99,
            tau=1e-3
        )
        self.action_space = gym.spaces.Discrete(12)
        self.model.load_state_dict(torch.load("output/2025.05.04-23.37.05/checkpoints/ckpt_2400.pt", "rb"))
        self.multistep_wrapper.reset()
        self.skip_frame_wrapper.reset()

    def act(self, observation):
        return super().act(observation)