import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from algo import Double_DQN_Agent
import os
import torch
import click
from PIL import Image

@click.command()
@click.option('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on')
@click.option('--feature-dim', default=512, help='Dimension of CNN feature extractor output')
@click.option('--hidden-size', default=128, help='Hidden layer size in Q-network')
@click.option('--horizon', default=4, help='Number of stacked frames')
@click.option('--gamma', default=0.99)
@click.option('--checkpoint', required=True)
def eval(device, feature_dim, hidden_size, horizon, gamma, checkpoint):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    frames = []
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    obs_space = env.observation_space.shape[:-1]
    action_space = env.action_space.n
    action_list = list(range(action_space))

    agent = Double_DQN_Agent(
        in_channels=horizon,
        feature_dim=feature_dim,
        n_actions=action_space,
        n_hidden=hidden_size,
        horizon=horizon,
        obs_shape=obs_space,
        lr=0.0,
        device=device,
        gamma=gamma,
        tau=None
    )
    
    agent.model.load_state_dict(torch.load(checkpoint))
    agent.model.to(device)

    total_reward = 0
    obs = env.reset()
    done = False
    step = 0
    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        frames.append(Image.fromarray(env.render(mode='rgb_array')))
        step += 1

    print("Total reward:", total_reward)
    print("Steps:", step)
    gif_path = "mario.gif"
    img = frames[0]
    img.save(gif_path, save_all=True, append_images=frames[1:], duration=20, loop=0)

if __name__ == '__main__':
    eval()