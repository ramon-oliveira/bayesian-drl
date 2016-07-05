import gym
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
import time

def preprocess(I):
    I = I[35:195] # crop score bar
    I = I[::2, ::2, 0] # down sampling
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to
    return I


if __name__ == '__main__':

    episodes = 1
    size = 80  # image size
    e = 0.00  # e-greedy policy
    k = 1  # the agent sees and selects an action every kth frame
    m = 4  # number of frames looked at each moment
    render = False
    # create enviroment
    env = gym.make('Pong-v0')

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
    Q.add(Dense(6, activation='linear', init='zero'))

    print("compiling Q network")
    Q.compile(loss="mse", optimizer='adadelta')
    Q.summary()
    Q.load_weights('breakout.h5')

    for episode in range(episodes):
        obs0 = np.zeros([m, size, size], dtype=np.int8)
        obs1 = np.zeros([m, size, size], dtype=np.int8)
        obs0[:] = obs1[:] = preprocess(env.reset())

        action = env.action_space.sample()
        done = False
        t = 0
        while not done:
            if (t % k) == 0:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    qval = Q.predict(np.array([obs0]), verbose=0)[0]
                    action = qval.argmax()
                    print(action, qval)
            (ob, reward, done, _info) = env.step(action)

            # update state
            obs1[1:] = obs1[:m-1]
            obs1[0] = preprocess(ob)

            # set last state
            obs0[:] = obs1[:]
            env.render()
            time.sleep(0.01)
            t += 1
