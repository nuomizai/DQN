from model import Q_network, Replay_memory
import torch
import gym
import torch.optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse


def train(render):
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    gamma = 0.9
    batch_size = 32
    replay_capacity = 10000
    replay = Replay_memory(replay_capacity, state_dim, batch_size)
    dqn = Q_network(state_dim, action_dim)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=0.0005)
    sum_cnt = 0
    av_reward = []
    for i in tqdm(range(2000)):
        sum_reward = 0.0
        av_loss = 0
        cur_state = env.reset()
        cur_state = torch.tensor(cur_state, dtype=torch.float)
        done = False
        while not done:
            action = dqn.epsilon_greedy(cur_state)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float)
            mask = torch.tensor(0.0 if done else 1.0)
            replay.insert(cur_state, action, reward, mask, next_state)
            sum_reward += reward
            cur_state = next_state
            sum_cnt += 1
            if sum_cnt < batch_size:
                continue
            state_batch, action_batch, reward_batch, next_state_batch, masks_batch = replay.sample(sum_cnt)
            Q_val = dqn.get_value(state_batch, action_batch)
            next_Q = dqn.get_max_Q(next_state_batch)
            target_y = reward_batch + gamma * masks_batch * next_Q
            loss = torch.mean((target_y - Q_val) ** 2)
            av_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i % 10 == 0:
            total_reward = 0
            for _ in range(10):
                state = env.reset()
                for _ in range(300):
                    state = torch.tensor(state, dtype=torch.float)
                    action = dqn.get_max_action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            av_reward.append(total_reward / 10)
    av_reward = np.array(av_reward)
    plt.plot(av_reward)
    plt.title('Returns for Cartpole')
    plt.xlabel('episode')
    plt.ylabel('returns')
    plt.savefig('result/returns.jpg')
    print('results saved under result/')
    if render:
        for _ in range(10):
            state = env.reset()
            for _ in range(300):
                env.render()
                state = torch.tensor(state, dtype=torch.float)
                action = dqn.get_max_action(state)
                state, reward, done, _ = env.step(action)
                if done:
                    break
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DQN training code')
    # Configurations
    parser.add_argument('--render', type=bool, default=False, help='train config file path')
    args = parser.parse_args()
    train(args.render)
