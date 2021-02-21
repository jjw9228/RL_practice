import gym
import numpy as np
from matplotlib import pyplot as plt
import random

def random_argmax(arr):
    m = np.max(arr)
    ind = [k for k in range(len(arr)) if arr[k] == m]
    return random.choice(ind)

env = gym.make('FrozenLake-v0')        # is_slippery False

Q = np.zeros([env.observation_space.n, env.action_space.n])
N = 2000
discounted_rate = 0.99
lr = 0.85
rList = []
for i in range(N):
    state = env.reset()
    done = False
    rAll = 0

    while not done:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
        new_state, reward, done, _ = env.step(action)
        Q[state, action] += lr * (reward + discounted_rate * np.max(Q[new_state, :]) - Q[state, action])

        state = new_state
        rAll += reward

    rList.append(rAll)

print("Success rate : " + str(sum(rList) / N))
print("Final Q-Table Value")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rList)), rList, color='blue')
plt.show()