import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import imageio
import click
from algo import Double_DQN_Agent
from algo.utils import ReplayBuffer
import torch
import math
import wandb
from datetime import datetime
from tqdm import tqdm
import numpy as np
import random
import os

@click.command()
@click.option('--seed', default=42)
@click.option('--buffer-size', default=30000, help='Replay buffer capacity')
@click.option('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on')
@click.option('--feature-dim', default=512, help='Dimension of CNN feature extractor output')
@click.option('--hidden-size', default=128, help='Hidden layer size in Q-network')
@click.option('--learning-rate', default=2.5e-4, help='Learning rate for optimizer')
@click.option('--horizon', default=4, help='Number of stacked frames')
@click.option('--batch-size', default=64, help='Batch size for training')
@click.option('--train-interval', default=4, help='Train the agent every N steps')
@click.option('--num-episodes', default=10000, help='Number of training episodes')
@click.option('--save-interval', default=100, help='Logging and checkpoint interval (in episodes)')
@click.option('--gamma', default=0.99)
@click.option('--tau', default=1e-4)
@click.option('--max-steps', default=10000)
@click.option('--train-initial-step', default=10000)
@click.option('--eps-decay-rate', default=0.9995)
@click.option('--eps-minimum', default=0.01)
def train(seed, buffer_size, device, feature_dim, hidden_size, learning_rate,
          horizon, batch_size, train_interval, num_episodes, save_interval, 
          gamma, tau, max_steps, train_initial_step, eps_decay_rate, eps_minimum):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    run_name = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")

    ckpt_dir = f"./output/{run_name}/checkpoints/"
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb_run = wandb.init(
        dir=f'./output/{run_name}',
        project='drl_hw3',
        mode="online",
        name=run_name,
    )
    wandb.config.update(
        {
            "output_dir": f'./output/{run_name}',
        }
    )

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    obs_space = env.observation_space.shape[:-1]
    action_space = env.action_space.n
    action_list = list(range(action_space))

    replay_buffer = ReplayBuffer(
        horizon=horizon,
        obs_shape=obs_space, 
        action_shape=action_space, 
        capacity=buffer_size,
        device='cpu'
    )

    agent = Double_DQN_Agent(
        in_channels=horizon,
        feature_dim=feature_dim,
        n_actions=action_space,
        n_hidden=hidden_size,
        horizon=horizon,
        obs_shape=obs_space,
        lr=learning_rate,
        device=device,
        gamma=gamma,
        tau=tau
    )

    steps = 0
    reward_history = []
    best_return = -math.inf
    epsilon = 1.0
    loss = 0.0
    with tqdm(range(num_episodes), desc=f"Training episode 0", leave=False) as tepisode:
        for episode in tepisode:

            obs = env.reset()
            done = False
            total_reward = 0
            episode_step = 0
            while not done:

                if epsilon < random.uniform(0, 1):
                    action = agent.act(obs)
                else:
                    action = random.choice(action_list)

                next_obs, reward, done, _ = env.step(action)
                total_reward += reward
                replay_buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs.copy()

                steps += 1
                episode_step += 1

                wandb_run.log({
                    'steps': steps,
                }, step=steps)

                if steps % train_interval == 0 and len(replay_buffer) >= batch_size and steps > train_initial_step:
                    batch = replay_buffer.sample(batch_size)
                    loss = agent.train(batch)
                    wandb_run.log({
                        'loss': loss,
                        'lr': learning_rate,
                    }, step=steps)

                if max_steps and episode_step >= max_steps:
                    break

            reward_history.append(total_reward)
            replay_buffer.multistep_wrapper.reset()
            replay_buffer.multistep_wrapper_next.reset()
            agent.multistep_wrapper.reset()

            wandb_run.log({
                'total_reward': total_reward,
                'episode': episode,
                'epsilon': epsilon,
                'episode_length': episode_step
            }, step=steps)

            mean_span = min(100, episode)
            avg_return = np.mean(reward_history[-mean_span:])
            tepisode.set_postfix(Reward=avg_return, loss=loss)
            tepisode.set_description(f"Training episode {episode}")
            epsilon = max(epsilon*eps_decay_rate, eps_minimum)

            if (episode+1) % save_interval == 0 and avg_return > best_return:
                # Save the checkpoint
                with open(os.path.join(ckpt_dir, f"ckpt_{episode+1}.pt"), "+wb") as f:
                    torch.save(agent.model.state_dict(), f)
                best_return = avg_return

if __name__ == "__main__":
    train()