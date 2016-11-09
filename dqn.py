import os
import sys
import gym
import tqdm
import argparse
import pickle as pkl
import numpy as np
from keras import backend as K
from keras import objectives
from keras.models import Sequential, model_from_yaml
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.utils.generic_utils import Progbar
from layers import OSBayesian, OSBayesianConvolution2D
import objectives
import utils

parser = argparse.ArgumentParser(description='Atari2600 DQN experiments.')
parser.add_argument('--game', dest='game', required=True,
                    choices=['pong', 'breakout', 'space-invaders'])
parser.add_argument('--model', dest='model', required=True,
                    choices=['maximum-likelihood', 'onesample-bayesian'])
parser.add_argument('--render', dest='render', action='store_true')

parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)
parser.add_argument('--replay-size', dest='replay_size', type=int, default=5000)
parser.add_argument('--nb-batch', dest='nb_batch', type=int, default=64)
parser.add_argument('--nb-frame', dest='nb_frame', type=int, default=5000,
                    help='number of frames the network is trained on')
parser.add_argument('--C', dest='C', type=int, default=10000,
                    help='target network update frequency')
parser.add_argument('--gamma', dest='gamma', type=float, default=0.99,
                    help='discount factor in Q-learning')
parser.add_argument('--frame-skip', dest='frame_skip', type=int, default=4,
                    help='skip frames this many times after the agent select an action')
parser.add_argument('--nb-frame-state', dest='nb_frame_state', type=int, default=4,
                    help='number of frames to compose the state')
parser.add_argument('--update-frequency', dest='update_frequency', type=int, default=4,
                    help='number of actions made by the agente between updates')


def game_config(args):
    if args.game == 'pong':
        actions = [0, 2, 3]
        meanings = ['NOOP', 'UP', 'DOWN']
        enviroment = gym.make('Pong-v0')
    elif args.game == 'breakout':
        actions = [0, 1, 2, 3]
        meanings = ['NOOP', 'FIRE', 'RIGTH', 'LEFT']
        enviroment = gym.make('Breakout-v0')
    elif args.game == 'space-invaders':
        actions = [0, 1, 2, 3]
        meanings = ['NOOP', 'FIRE', 'RIGTH', 'LEFT']
        enviroment = gym.make('SpaceInvaders-v0')
    else:
        raise Exception('Unknown game')

    shape = enviroment.observation_space.shape
    screen = args.nb_frame_state, shape[0]//2, shape[1]//2
    return {
        'actions': actions,
        'meanings': meanings,
        'enviroment': enviroment,
        'state_shape': screen,
        'preprocessing': utils.preprocessing,
    }


def create_model(args, game_config):
    input_shape = game_config['state_shape']
    output_dim = len(game_config['actions'])
    model = Sequential()
    if args.model == 'maximum-likelihood':
        model.add(Convolution2D(32, 8, 8, border_mode='same', subsample=[4, 4],
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, border_mode='same', subsample=[2, 2]))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=[1, 1]))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(output_dim, activation='linear'))
        loss = 'mse'
    elif args.model == 'onesample-bayesian':
        mean_prior = 0.0
        std_prior = 0.05
        model.add(OSBayesianConvolution2D(mean_prior, std_prior, 32, 8, 8,
                                                 border_mode='same', subsample=[4, 4],
                                                 input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(OSBayesianConvolution2D(mean_prior, std_prior, 64, 4, 4,
                                                 border_mode='same', subsample=[2, 2]))
        model.add(Activation('relu'))
        model.add(OSBayesianConvolution2D(mean_prior, std_prior, 64, 3, 3,
                                                 border_mode='same', subsample=[1, 1]))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(OSBayesian(512, mean_prior, std_prior))
        model.add(Activation('relu'))
        model.add(OSBayesian(output_dim, mean_prior, std_prior))
        loss = objectives.explicit_bayesian_loss(model, mean_prior, std_prior,
                                                 batch_size, nb_batch)

    else:
        raise Exception('Unknown model type: {0}'.format(args.model))
    model.compile(loss=loss, optimizer='adam')
    return model


def train(args, game_config):
    e = 1.0  # e-greedy policy, drops from e=1 to e=0.1
    state_shape = game_config['state_shape']
    env = game_config['enviroment']
    preprocessing = game_config['preprocessing']
    actions = game_config['actions']

    # populates replay memory with some random sequences
    state0_rm = np.zeros([args.replay_size, *state_shape], dtype=np.int8)
    action_rm = np.zeros([args.replay_size], dtype=np.int8)
    reward_rm = np.zeros([args.replay_size], dtype=np.int8)
    state1_rm = np.zeros([args.replay_size, *state_shape], dtype=np.int8)
    terminal_rm = np.zeros([args.replay_size], dtype=np.bool)

    # Initialize action value function with random with random weights
    model = create_model(args, game_config)

    # initialize target action-value function ^Q with same wieghts
    model_target = model_from_yaml(model.to_yaml())
    model_target.set_weights(model.get_weights())

    # keep track variables
    episodes = 0
    model_updates = 0
    idx_rm = 0
    steps = 0
    idxs_rm = np.arange(args.replay_size)
    idxs_batch = np.arange(args.batch_size)
    nb_active_rm = 0
    total_frames = 0

    pbar_frames = tqdm.tqdm(total=args.nb_frame, leave=True, position=0)
    while total_frames < args.nb_frame:
        obs0 = np.zeros(state_shape, dtype=np.int8)
        obs1 = np.zeros(state_shape, dtype=np.int8)
        obs0[:] = obs1[:] = preprocessing(env.reset())

        a = max(1, int(total_frames/max(1, episodes)))
        pbar_episode = tqdm.tqdm(total=a, leave=False, position=1)
        t = 0
        treward = 0
        done = False
        nb_actions = 0
        while not done and total_frames < args.nb_frame:
            if (t % args.frame_skip) == 0:
                if np.random.rand() < e:
                    action_idx = np.random.randint(low=0, high=len(actions))
                else:
                    qval = model.predict(np.array([obs0]), verbose=0)
                    action_idx = qval.argmax()
                nb_actions += 1

            ob, reward, done, info = env.step(actions[action_idx])
            treward += reward

            if (t % args.frame_skip) == 0:
                # update state
                obs1[1:] = obs1[:-1]
                obs1[0] = preprocessing(ob)
                reward = np.clip(reward, -1, 1)

                # save replay memory
                state0_rm[idx_rm] = obs0[:]
                action_rm[idx_rm] = action_idx
                reward_rm[idx_rm] = reward
                state1_rm[idx_rm] = obs1[:]
                terminal_rm[idx_rm] = int(done)

                # set last state
                obs0[:] = obs1[:]
                nb_active_rm = min(nb_active_rm + 1, args.replay_size)

                # update replay idx
                idx_rm = (idx_rm + 1) % args.replay_size

            if nb_active_rm >= args.batch_size and nb_actions == args.update_frequency:
                nb_actions = 0

                # sample random minibatch of transitions from D
                idxs = np.random.choice(idxs_rm[:nb_active_rm], size=args.batch_size)

                qamax = np.max(model_target.predict(state1_rm[idxs]), axis=1)
                y_q = model.predict(state0_rm[idxs])
                y_q_target = reward_rm[idxs] + (1.0-terminal_rm[idxs])*args.gamma*qamax
                y_q[idxs_batch, action_rm[idxs]] = y_q_target

                # train on batch
                train_loss = model.train_on_batch(state0_rm[idxs], y_q)
                model_updates += 1

                # update Q_target every C updates
                if (model_updates % args.C) == 0:
                    model_target.set_weights(model.get_weights())

                # set e-greedy policy adjust
                e = max(e - 0.00009, 0.1)

            t += 1
            if args.render: env.render()
            pbar_episode.total = max(pbar_episode.total, t)
            pbar_episode.update(1)
            pbar_frames.update(1)

        pbar_episode.close()

        # stats = 'Episode {0}\t| points {1}\t| total-frames {2}\t| e-greedy {3:.2f}\n'
        # print(stats.format(episodes+1, treward, steps, e))

        episodes += 1
        total_frames += t

    pbar_frames.close()



if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])

    game_config = game_config(args)
    train(args, game_config)



"""
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
"""
