import gym
from gym.envs.registration import register
import numpy as np
from matplotlib import pyplot as plt
import random


register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

def random_argmax(arr):
    m = np.max(arr)
    ind = [k for k in range(len(arr)) if arr[k] == m]
    return random.choice(ind)

env = gym.make('FrozenLake-v3')        # is_slippery False

Q = np.zeros([env.observation_space.n, env.action_space.n])
N = 2000
rList = []
for i in range(N):
    state = env.reset()
    done = False
    rAll = 0

    while not done:
        action = random_argmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)
        Q[state, action] = reward + np.max(Q[new_state, :])

        state = new_state
        rAll += reward

    rList.append(rAll)

print("Success rate : " + str(sum(rList) / N))
print("Final Q-Table Value")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rList)), rList, color='blue')
plt.show()