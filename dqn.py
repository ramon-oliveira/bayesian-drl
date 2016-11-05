import sys
import os
import gym
import numpy as np
import pickle
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.callbacks import Progbar
from keras import backend as K
from keras import objectives
import matplotlib.pyplot as plt


actions_pong = [0, 2, 3]
actions_meanings_pong = ['NOOP', 'UP', 'DOWN']

actions_breakout = [0, 1, 2, 3]
actions_meanings_breakout = ['NOOP', 'FIRE', 'RIGTH', 'LEFT']

def preprocess(I):
    # I = I[35:195] # crop score bar
    I = I[::2, ::2, 0] # down sampling
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to
    return I


def clipped_mse(y_true, y_pred):
    return K.clip(objectives.mean_squared_error(y_true, y_pred), -1.0, 1.0)

def evaluate(Q, actions, k):
    episodes = 30
    print ('Evaluating')
    re = np.zeros(episodes)
    pb = Progbar(episodes)
    e = 0.05
    for episode in range(episodes):
        obs = np.zeros([m, heigth, width], dtype=np.int8)
        obs[:] = preprocess(env.reset())
        t = 0
        treward = 0
        done = False
        while not done:
            if (t % k) == 0 and  np.random.rand() < e:
                action_idx = np.random.randint(low=0, high=len(actions))
            elif (t % k) == 0:
                qval = Q.predict(np.array([obs]), verbose=0)[0]
                action_idx = qval.argmax()

            ob, reward, done, info = env.step(actions[action_idx])
            treward += reward
            t += 1
            
            if e == 1:
               qval = Q.predict(np.array([obs]), verbose=0)[0]

            # update state
            obs[1:] = obs[:m-1]
            obs[0] = preprocess(ob)
        pb.update(episode+1, [['points', treward]], force=True)
        re[episode] = treward
    
    stats = 'Mean {0:.2f}\t| std {1:.2f}\t| Games {2}\n'
    print(stats.format(np.mean(re), np.std(re), episodes))
    meanpoints.append(np.mean(re))
    std.append(np.std(re))
    maxpoints.append(re.max())

if __name__ == '__main__':
	
    game = sys.argv[1]
    
    # create enviroment
    env = gym.make(game)


    # parameters
    max_episodes = 100000
    heigth = env.observation_space.shape[0]/2
    width = env.observation_space.shape[1]/2
    e = 1.0  # e-greedy policy, drops from e=1 to e=0.1
    k = 4  # the agent sees and selects an action every kth frame
    m = 4  # number of frames looked at each moment
    replay_size = 5000 # replay memory size
    batch_size = 32 # batch size
    gamma = 0.99  #discount factor for future rewards Q function
    C = 10000 # frequency target network update
    render = False 
    resume = False

    if 'pong' in game.lower():
        actions = np.array(actions_pong)
        actions_meanings = actions_meanings_pong
    elif 'breakout' in game.lower():
        actions = np.array(actions_breakout)
        actions_meanings = actions_meanings_breakout
    else:
        actions = np.arange(env.action_space.n)
        actions_meanings = env.get_action_meanings()

    print('Actions:', actions_meanings)

    # populates replay memory with some random sequences
    state0_rm = np.zeros([replay_size, m, heigth, width], dtype=np.int8)
    action_rm = np.zeros([replay_size], dtype=np.int8)
    reward_rm = np.zeros([replay_size], dtype=np.int8)
    state1_rm = np.zeros([replay_size, m, heigth, width], dtype=np.int8)
    terminal_rm = np.zeros([replay_size], dtype=np.bool)

    # Initialize action value function with random with random weights
    print('creating Q network')
    Q = Sequential()
    Q.add(Convolution2D(32, 8, 8, border_mode='same', subsample=[4, 4], input_shape=[m, heigth, width]))
    Q.add(Activation('relu'))
    Q.add(Convolution2D(64, 4, 4, border_mode='same', subsample=[2, 2]))
    Q.add(Activation('relu'))
    Q.add(Convolution2D(64, 3, 3, border_mode='same', subsample=[1, 1]))
    Q.add(Activation('relu'))
    Q.add(Flatten())
    Q.add(Dense(512, activation='relu'))
    Q.add(Dense(len(actions), activation='linear'))

    print('compiling Q network')
    Q.compile(loss=clipped_mse, optimizer='adadelta')
    Q.summary()

    # keep track variables
    updates = 0
    idx_rm = 0
    steps = 0
    idxs_rm = np.arange(replay_size)
    idxs_batch = np.arange(batch_size)

    nb_active_rm = 0
    ep = 0
    maxpoints = []
    meanpoints =[]
    std = []

    loa = game.lower() + '-Prog/{0}'
    if resume and os.path.isfile(wei.format(game.lower())):
        print('Loading weights from', game)
        Q.load_weights(wei.format(game.lower()))
        state0_rm    = np.load(loa.format('state0.npy'))
        action_rm    = np.load(loa.format('action.npy'))
        reward_rm    = np.load(loa.format('reward.npy'))
        state1_rm    = np.load(loa.format('state1.npy'))
        terminal_rm  = np.load(loa.format('terminal.npy'))
        ep           = np.load(loa.format('episode.npy'))
        e            = np.load(loa.format('e.npy'))
        nb_active_rm = np.load(loa.format('nb_active_rm.npy'))
        updates      = np.load(loa.format('updates.npy'))
        t            = np.load(loa.format('t.npy'))
        idx_rm       = np.load(loa.format('idx_rm.npy'))
        with open(loa.format('maxpoints.pickle'),'rb') as f:
            maxpoints = pickle.load(f)
        with open(loa.format('meanpoints.pickle'),'rb') as f:
            meanpoints = pickle.load(f)
        with open(loa.format('std.pickle'), 'rb') as f:
            std = pickle.load(f)


    # initialize target action-value function ^Q with same wieghts
    print('copying Q to Q_target')
    Q_target = model_from_yaml(Q.to_yaml())
    Q_target.set_weights(Q.get_weights())

	
	
    print('Starting to Train')
    for episode in range(ep, max_episodes):
        
        obs0 = np.zeros([m, heigth, width], dtype=np.int8)
        obs1 = np.zeros([m, heigth, width], dtype=np.int8)
        obs0[:] = obs1[:] = preprocess(env.reset())
	
        t = 0
        treward = 0
        pb = Progbar(5000)

        done = False
        while not done:
            if (t % k) == 0:
                if np.random.rand() < e:
                    action_idx = np.random.randint(low=0, high=len(actions))
                else:
                    qval = Q.predict(np.array([obs0]), verbose=0)
                    action_idx = qval.argmax()
            ob, reward, done, info = env.step(actions[action_idx])
            steps += 1

            # update state
            obs1[1:] = obs1[:m-1]
            obs1[0] = preprocess(ob)
            treward += reward
            reward = np.clip(reward, -1, 1)

            # save replay memory
            state0_rm[idx_rm] = obs0[:]
            action_rm[idx_rm] = action_idx
            reward_rm[idx_rm] = reward
            state1_rm[idx_rm] = obs1[:]
            terminal_rm[idx_rm] = int(done)

            # if t % 100 == 0:
            #     plt.imsave('obs.png', np.concatenate(obs0, axis=1), cmap=plt.cm.binary)

            # set last state
            obs0[:] = obs1[:]

            nb_active_rm = min(nb_active_rm + 1, replay_size)

            if nb_active_rm >= batch_size and (t % k) == 0:
                # sample random minibatch of transitions from D
                idxs = np.random.choice(idxs_rm[:nb_active_rm], size=batch_size)

                qamax = np.max(Q_target.predict(state1_rm[idxs]), axis=1)
                y_Q = Q.predict(state0_rm[idxs])
                y_Q_target = reward_rm[idxs] + (1.0-terminal_rm[idxs])*gamma*qamax
                y_Q[idxs_batch, action_rm[idxs]] = y_Q_target

                # train on batch
                train_loss = Q.train_on_batch(state0_rm[idxs], y_Q)
                updates += 1
                pb.add(k, [['clipped_mse', train_loss]])

                # update Q_target every C trains
                if (updates % C) == 0:
                    Q_target.set_weights(Q.get_weights())

                # set e-greedy policy adjust
                e = max(e - 0.0000009, 0.1)

            # update replay idx
            idx_rm = (idx_rm + 1) % replay_size
            t += 1
            if render: env.render()

        pb.target = t
        pb.update(t, [['clipped_msre', train_loss]], force=True)
        if (episode % 600) == 0:
            sav = game.lower() + '-Prog/{0}'
            Q.save_weights(sav.format(game.lower()), overwrite=True)
            evaluate(Q, actions, k) 
            np.save(sav.format('state0'),state0_rm)
            np.save(sav.format('action'),action_rm)
            np.save(sav.format('reward'),reward_rm)
            np.save(sav.format('state1'),state1_rm)
            np.save(sav.format('terminal'),terminal_rm)
            np.save(sav.format('episode'),episode)
            np.save(sav.format('e'),e)
            np.save(sav.format('nb_active_rm'), nb_active_rm)
            np.save(sav.format('updates'), updates)
            np.save(sav.format('t'), t)
            np.save(sav.format('idx_rm'), idx_rm)
            with open(sav.format('maxpoints.pickle'),'wb') as f:
               pickle.dump(maxpoints, f)
            with open(sav.format('meanpoints.pickle'),'wb') as f:
                pickle.dump(meanpoints, f)
            with open(sav.format('std.pickle'), 'wb') as f:
                pickle.dump(std, f)

        stats = 'Episode {0}\t| points {1}\t| episode-frames {2}\t| total-frames {3}\t| e-greedy {4:.2f}\n'
        print(stats.format(episode+1, treward, t, steps, e))
