import sys
import os
import gym
import numpy as np
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.callbacks import Progbar
from keras import backend as K
from keras import objectives


def preprocess(I):
    I = I[35:195] # crop score bar
    I = I[::2, ::2, 0] # down sampling
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to
    return I


def clipped_mse(y_true, y_pred):
    return K.clip(objectives.mean_squared_error(y_true, y_pred), -1, 1)

if __name__ == '__main__':

    # parameters
    total_frames = 10000000
    max_episodes = 10000
    max_steps = 5000 # maximum number of steps per episode
    size = 80  # image size
    e = 0.1  # e-greedy policy, drops from e=1 to e=0.1
    k = 1  # the agent sees and selects an action every kth frame
    m = 4  # number of frames looked at each moment
    replay_size = 100000 # replay memory size
    batch_size = 32 # batch size
    gamma = 0.99  #discount factor for future rewards Q function
    C = 10000 # frequency target update
    render = False
    resume = True
    game = sys.argv[1]

    # create enviroment
    env = gym.make(game)

    # populates replay memory with some random sequences
    state0_rm = np.zeros([replay_size, m, size, size], dtype=np.int8)
    action_rm = np.zeros([replay_size], dtype=np.int8)
    reward_rm = np.zeros([replay_size], dtype=np.int8)
    state1_rm = np.zeros([replay_size, m, size, size], dtype=np.int8)
    terminal_rm = np.zeros([replay_size], dtype=np.bool)

    # Initialize action value function with random with random weights
    print('creating Q network')
    Q = Sequential()
    Q.add(Convolution2D(32, 8, 8, border_mode='same', subsample=[4, 4], input_shape=[m, size, size]))
    Q.add(Activation('relu'))
    Q.add(Convolution2D(64, 4, 4, border_mode='same', subsample=[2, 2]))
    Q.add(Activation('relu'))
    Q.add(Convolution2D(64, 3, 3, border_mode='same', subsample=[1, 1]))
    Q.add(Activation('relu'))
    Q.add(Flatten())
    Q.add(Dense(512, activation='relu'))
    Q.add(Dense(env.action_space.n, activation='linear', init='zero'))

    print('compiling Q network')
    Q.compile(loss=clipped_mse, optimizer='adadelta')
    Q.summary()

    if resume and os.path.isfile(game.lower()):
        print('Loading weights from', game)
        Q.load_weights(game.lower())

    # initialize target action-value function ^Q with same wieghts
    print('copying Q to Q_target')
    Q_target = model_from_yaml(Q.to_yaml())
    Q_target.set_weights(Q.get_weights())


    # keep track variables
    updates = 0
    idx_rm = 0
    steps = 0
    idxs_rm = list(range(replay_size))

    eh_nois = 0

    print('Starting to Train')
    for episode in range(max_episodes):
        ##Initialize sequence game and sequence s1 pre-process sequence
        obs0 = np.zeros([m, size, size], dtype=np.int8)
        obs1 = np.zeros([m, size, size], dtype=np.int8)
        obs0[:] = obs1[:] = preprocess(env.reset())

        t = 0
        treward = 0
        pb = Progbar(5000)

        action = env.action_space.sample()
        done = False
        while not done:
            if (t % k) == 0:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    qval = Q.predict(np.array([obs0]), verbose=0)
                    action = qval.argmax()
            (ob, reward, done, _info) = env.step(action)
            steps += 1

            # update state
            obs1[1:] = obs1[:m-1]
            obs1[0] = preprocess(ob)
            treward += reward
            reward = np.clip(reward, -1, 1)

            # save replay memory
            state0_rm[idx_rm] = obs0[:]
            action_rm[idx_rm] = action
            reward_rm[idx_rm] = reward
            state1_rm[idx_rm] = obs1[:]
            terminal_rm[idx_rm] = int(done)

            # set last state
            obs0[:] = obs1[:]

            # if t % 50 == 0:
            #     plt.imsave('tst.png', np.concatenate(obs0, axis=1), cmap=plt.cm.binary)

            eh_nois = min(eh_nois + 1, replay_size)

            if eh_nois >= batch_size and (t % k) == 0:
                # sample random minibatch of transitions from D
                idxs = np.random.choice(idxs_rm[:eh_nois], size=batch_size)

                qamax = np.max(Q_target.predict(state1_rm[idxs]), axis=1)
                y_Q = Q.predict(state0_rm[idxs])
                y_Q_target = reward_rm[idxs] + (1.0-terminal_rm[idxs])*gamma*qamax

                # print(qamax.shape, y_Q.shape, y_Q_target.shape)
                for i, a in enumerate(action_rm[idxs]):
                    y_Q[i, a] = y_Q_target[i]

                # train on batch
                train_loss = Q.train_on_batch(state0_rm[idxs], y_Q)
                updates += 1
                pb.add(1, [['clipped_mse', train_loss]])

                # update Q_target every C trains
                if (updates % C) == 0:
                    Q_target.set_weights(Q.get_weights())

            # update replay idx
            idx_rm = (idx_rm + 1) % replay_size

            # set e-greedy policy adjust
            if e > 0.1:
                e -= 0.0000009

            if render:
                env.render()

            t += 1

        pb.target = t
        pb.update(t, [['clipped_mse', train_loss]], force=True)
        if (episode % 100) == 0:
            Q.save_weights(game.lower(), overwrite=True)

        stats = 'Episode {0}\t| points {1}\t| episode-frames {2}\t| total-frames {3}\t| e-greedy {4:.2f}\n'
        print(stats.format(episode+1, treward, t, steps, e))
