import gym
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def one_hot(x):
    return tf.Variable(np.identity(16, dtype='float32')[x:x+1])


env = gym.make('FrozenLake-v0')        # is_slippery False

input_size = env.observation_space.n
output_size = env.action_space.n
lr = 0.1

W = tf.Variable(tf.random.uniform([input_size, output_size], minval=0, maxval=0.01, dtype='float32'))
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
num_episode = 2000
dis = .99
rList = []

for i in range(num_episode):
    s = env.reset()
    e = 1. / ((i / 50) + 10)
    rAll = 0
    done = False

    if i % 100 == 0:
        print(i)
    while not done:
        var = one_hot(s)
        Qs = tf.matmul(var, W).numpy()

        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)
        var1 = one_hot(s1)
        if done:
            Qs[0, a] = reward
        else:
            Qs1 = tf.matmul(var1, W).numpy()
            Qs[0, a] = reward + dis * np.max(Qs1)

        loss = lambda: tf.reduce_sum(tf.square(Qs - tf.matmul(var, W)))
        optimizer.minimize(loss, var_list=[W])

        rAll += reward
        s = s1
    rList.append(rAll)

print("Success rate : " + str(sum(rList) / num_episode))
plt.bar(range(len(rList)), rList, color='blue')
plt.show()