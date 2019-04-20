from model import Q_network, Replay_memory
import torch
import gym
import torch.optim
import torch.nn as nn
import visdom


def train():
    vis = visdom.Visdom(env='dqn-CartPole-v0')
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
    for i in range(10000):
        # print('episode:', i)
        # env.render()
        sum_reward = 0.0
        av_loss = 0
        cnt = 0
        # print('into loop')
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
            cnt += 1
            # print(cnt)
            if sum_cnt < batch_size:
                continue
            state_batch, action_batch, reward_batch, next_state_batch, masks_batch = replay.sample(sum_cnt)
            # print(action_batch)
            Q_val = dqn.get_value(state_batch, action_batch)
            next_Q = dqn.get_max_Q(next_state_batch)
            target_y = reward_batch + gamma * masks_batch * next_Q
            loss = torch.mean((target_y - Q_val) ** 2)
            av_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i % 100 == 0:
            total_reward = 0
            for _ in range(10):
                state = env.reset()
                for _ in range(300):
                    env.render()
                    state = torch.tensor(state, dtype=torch.float)
                    action = dqn.get_max_action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            av_reward = total_reward / 10
            print(i, ': ', sum_reward)
            vis.line(X=torch.tensor([i]), Y=torch.tensor([av_loss / cnt]), update='append' if i > 0 else None,
                     win='loss-profile',
                     opts=dict(title='loss-profile', xlabel='step', ylabel='loss'))
            vis.line(X=torch.tensor([i]), Y=torch.tensor([av_reward]), update='append' if i > 0 else None,
                     win='reward-profile',
                     opts=dict(title='reward-profile', xlabel='step', ylabel='reward'))


if __name__ == '__main__':
    train()
