import gym
import numpy as np
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.callbacks import Progbar


def preprocess(I):
    I = I[35:195] # crop score bar
    I = I[::2, ::2, 0] # down sampling
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to
    return I


def populate_memory(env, D, m, k, max_steps):
    state0_rm, action_rm, reward_rm, state1_rm, terminal_rm = D
    replay_size = state0_rm.shape[0]

    n = 0
    obs0 = np.zeros([m, 80, 80], dtype=np.int8)
    obs1 = np.zeros([m, 80, 80], dtype=np.int8)
    while n < replay_size:
        obs0[:] = obs1[:] = preprocess(env.reset())
        action = env.action_space.sample()
        for i in range(max_steps):
            if i % k == 0:
                action = env.action_space.sample()
            (ob, reward, done, _info) = env.step(action)

            obs1[1:] = obs0[:m-1]
            obs1[0] = preprocess(ob)

            # save memory
            state0_rm[n] = obs0[:]
            action_rm[m] = action
            reward_rm[n] = reward
            state1_rm[n] = obs1[:]
            terminal_rm[n] = done
            n += 1

            obs0[:] = obs1[:]

            if (n % 1000) == 0:
                print('Replay memory lenght', n)
            if done or n >= replay_size:
                break


if __name__ == '__main__':

    # parameters
    total_frames = 10000000
    max_episodes = 10000
    max_steps = 5000 # maximum number of steps per episode
    size = 80  # image size
    e = 1.0  # e-greedy policy, drops from e=1 to e=0.1
    k = 2  # the agent sees and selects an action every kth frame
    m = 4  # number of frames looked at each moment
    replay_size = 100000 # replay memory size
    batch_size = 32 # batch size
    gamma = 0.99  #discount factor for future rewards Q function
    C = 10000 # frequency target update
    render = False

    # create enviroment
    env = gym.make('Breakout-v0')

    # populates replay memory with some random sequences
    state0_rm = np.zeros([replay_size, m, size, size], dtype=np.int8)
    action_rm = np.zeros([replay_size], dtype=np.int8)
    reward_rm = np.zeros([replay_size], dtype=np.int8)
    state1_rm = np.zeros([replay_size, m, size, size], dtype=np.int8)
    terminal_rm = np.zeros([replay_size], dtype=np.bool)
    D = [state0_rm, action_rm, reward_rm, state1_rm, terminal_rm]
    populate_memory(env, D, m, k, max_steps)

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

    # initialize target action-value function ^Q with same wieghts
    print("copying Q to Q_target")
    Q_target = model_from_yaml(Q.to_yaml())
    Q_target.set_weights(Q.get_weights())

    print("Starting to Train")
    ##Starts Playing and training
    updates = 0
    idx_rm = 0
    steps = 0
    idxs_rm = list(range(replay_size))
    for episode in range(max_episodes):
        ##Initialize sequence game and sequence s1 pre-process sequence
        obs0 = np.zeros([m, size, size], dtype=np.int8)
        obs1 = np.zeros([m, size, size], dtype=np.int8)
        obs0[:] = obs1[:] = preprocess(env.reset())

        treward = 0
        pb = Progbar(5000)

        action = env.action_space.sample()
        for t in range(max_steps):
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

            # save replay memory
            state0_rm[idx_rm] = obs0[:]
            action_rm[idx_rm] = action
            reward_rm[idx_rm] = reward
            state1_rm[idx_rm] = obs1[:]
            terminal_rm[idx_rm] = done

            # set last state
            obs0[:] = obs1[:]

            if (t % k) == 0:
                # sample random minibatch of transitions from D
                idxs = np.random.choice(idxs_rm, size=batch_size)
                if idx_rm not in idxs:
                    idxs[0] = idx_rm

                y_Q = Q.predict(state0_rm[idxs])
                maxs = np.max(Q_target.predict(state1_rm[idxs]), axis=1)
                # maxs = np.max(Q.predict(state1_rm[idxs]), axis=1)
                y_Q_target = reward_rm[idxs] + (1.0-terminal_rm[idxs])*gamma*maxs
                y_Q[:, action_rm[idxs]] = y_Q_target

                ##train on batch
                train_loss = Q.train_on_batch(state0_rm[idxs], y_Q)
                updates += 1
                pb.update(t, [['mse', train_loss]])

            # update replay idx
            idx_rm = (idx_rm + 1) % replay_size

            # set e-greedy policy adjust
            if e > 0.1:
                e -= 0.0000009

            # update Q_target every C trains
            if (updates % C) == 0:
                Q_target.set_weights(Q.get_weights())

            if render:
                env.render()

            if done:
                break

        pb.target = t
        pb.update(t, [['mse', train_loss]], force=True)
        if (episode % 100) == 0:
            Q.save_weights('breakout.h5', overwrite=True)
        print("\nEpisode", episode+1, "\tpoints =", treward, "\ttotal frames", steps,
              "\te-greedy", e)
        print("")

# if __name__ == '__main__':
#     train()
