import gym
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
import time
import sys
import matplotlib.pyplot as plt
import seaborn


def preprocess(I):
    I = I[35:195] # crop score bar
    I = I[::2, ::2, 0] # down sampling
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to
    return I


if __name__ == '__main__':

    episodes = 10
    size = 80  # image size
    e = 0.00  # e-greedy policy
    k = 1  # the agent sees and selects an action every kth frame
    m = 4  # number of frames looked at each moment
    plot = False
    game = sys.argv[1]

    # create enviroment
    env = gym.make(game)

    # Initialize action value function with random with random weights
    print("creating Q network")
    Q = Sequential()
    Q.add(Convolution2D(32, 8, 8, border_mode='same', subsample=[4, 4], input_shape=[4, size, size]))
    Q.add(Activation('relu'))
    Q.add(Convolution2D(64, 4, 4, border_mode='same', subsample=[2, 2]))
    Q.add(Activation('relu'))
    Q.add(Convolution2D(64, 3, 3, border_mode='same', subsample=[1, 1]))
    Q.add(Activation('relu'))
    Q.add(Flatten())
    Q.add(Dense(512, activation='relu'))
    Q.add(Dense(env.action_space.n, activation='linear', init='zero'))

    print("compiling Q network")
    Q.compile(loss="mse", optimizer='adadelta')
    Q.summary()
    Q.load_weights(game.lower())

    if plot:
        labels = env.get_action_meanings() #['NOOP', 'UP', 'DOWN']
        actions = list(range(env.action_space.n))#[0, 2, 3]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        width = 0.5
        li = ax.bar(np.arange(len(actions)), np.zeros(len(actions)), width)
        # ax.bar(np.arange(env.action_space.n), np.zeros(env.action_space.n), width)
        ax.set_ylim([-1, 2])
        ax.set_ylabel(game.upper())
        ax.set_ylabel('Q value')
        ax.set_xlabel('Actions')
        ax.set_xticks(np.arange(len(actions)) + width/2)
        ax.set_xticklabels(labels)
        # draw and show it
        fig.canvas.draw()
        plt.show(block=False)

    for episode in range(episodes):
        obs = np.zeros([m, size, size], dtype=np.int8)
        obs[:] = preprocess(env.reset())
        t = 0
        action = env.action_space.sample()
        done = False
        while not done:
            if (t % k) == 0 and  np.random.rand() < e:
                action = env.action_space.sample()
            elif (t % k) == 0:
                qval = Q.predict(np.array([obs]), verbose=0)[0]
                action = qval.argmax()
                print(action, qval)
            (ob, reward, done, _info) = env.step(action)

            # update state
            obs[1:] = obs[:m-1]
            obs[0] = preprocess(ob)

            env.render()

            if plot:
                fig.canvas.draw()
                for i, v in enumerate(qval[actions]):
                    li.patches[i].set_height(v)
                    li.patches[i].set_color('blue')
                li.patches[np.argmax(qval[actions])].set_color('r')
            else:
                time.sleep(0.01)
