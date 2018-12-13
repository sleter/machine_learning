import gym
import time

env = gym.make('CartPole-v0')
obj = env.reset()
env.render()

time.sleep(10000)