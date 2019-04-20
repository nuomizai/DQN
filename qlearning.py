import gym
import torch




env = gym.make('CartPole-v0')
state = env.reset()
# for _ in range(1000):
#     env.render()
#     state, reward, done, _, = env.step(env.action_space.sample())  # take a random action
#     print(state)
# env.close()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space[0]
print(action_dim)