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
parser.add_argument('--resume', dest='resume', action='store_true')
parser.add_argument('--weights-file', dest='weights_file')

parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)
parser.add_argument('--replay-size', dest='replay_size', type=int, default=500000)
parser.add_argument('--nb-batch', dest='nb_batch', type=int, default=64)
parser.add_argument('--nb-frame', dest='nb_frame', type=int, default=10000000,
                    help='number of frames the network is trained on')
parser.add_argument('--C', dest='C', type=int, default=10000,
                    help='target network update frequency')
parser.add_argument('--frame-skip', dest='frame_skip', type=int, default=4,
                    help='skip frames this many times after the agent select an action')
parser.add_argument('--nb-frame-state', dest='nb_frame_state', type=int, default=4,
                    help='number of frames to compose the state')
parser.add_argument('--update-frequency', dest='update_frequency', type=int, default=4,
                    help='number of actions made by the agente between updates')
parser.add_argument('--gamma', dest='gamma', type=float, default=0.99,
                    help='discount factor in Q-learning')
parser.add_argument('--initial-epsilon', dest='initial_epsilon', type=float, default=1.0)
parser.add_argument('--final-epsilon', dest='final_epsilon', type=float, default=0.1)
parser.add_argument('--final-epsilon-annealing', dest='final_epsilon_annealing',
                    type=int, default=1000000)


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


def get_newest_file(path):
    return max([os.path.join(path, f) for f in os.listdir(path)], key=os.path.getctime)

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

    if args.weights_file is not None:
        model.load_weights(args.weights_file)
    elif args.resume:
        f = 'weights/{0}/{1}'.format(args.game, args.model)
        model.load_weights(get_newest_file(f))

    return model


def train(args, game_config):
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
    epsilon = args.initial_epsilon
    epsilon_reduction = (args.initial_epsilon - args.final_epsilon)/args.final_epsilon_annealing
    episodes = 0
    model_updates = 0
    idx_rm = 0
    steps = 0
    idxs_rm = np.arange(args.replay_size)
    idxs_batch = np.arange(args.batch_size)
    nb_active_rm = 0
    total_frames = 0

    pbar_frames = tqdm.tqdm(total=args.nb_frame, desc='0000 episodes', leave=True, position=0)
    while total_frames < args.nb_frame:
        obs0 = np.zeros(state_shape, dtype=np.int8)
        obs1 = np.zeros(state_shape, dtype=np.int8)
        obs0[:] = obs1[:] = preprocessing(env.reset())

        a = max(1, int(total_frames/max(1, episodes)))
        pbar_episode = tqdm.tqdm(total=a, desc='   0   reward', leave=False, position=1)
        t = 0
        treward = 0
        done = False
        nb_actions = 0
        while not done and total_frames < args.nb_frame:
            if (t % args.frame_skip) == 0:
                if np.random.rand() < epsilon:
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

                # sample random minibatch of transitions from replay memories
                idxs = np.random.choice(idxs_rm[:nb_active_rm], size=args.batch_size)

                qamax = np.max(model_target.predict(state1_rm[idxs]), axis=1)
                y_q = model.predict(state0_rm[idxs])
                y_q_target = reward_rm[idxs] + (1.0-terminal_rm[idxs])*args.gamma*qamax
                y_q[idxs_batch, action_rm[idxs]] = y_q_target

                # train on batch
                train_loss = model.train_on_batch(state0_rm[idxs], y_q)
                model_updates += 1

                # update model_target every C updates
                if (model_updates % args.C) == 0:
                    model_target.set_weights(model.get_weights())
                    fname = 'weights/{0}/updates_{1}.h5'.format(args.game, model_updates)
                    model.save_weights(fname)


            t += 1
            epsilon = max(epsilon - epsilon_reduction, args.final_epsilon)

            pbar_episode.total = max(pbar_episode.total, t)
            pbar_episode.set_description('{0:4d}   reward'.format(int(treward)))
            pbar_episode.update(1)
            pbar_frames.update(1)
            if args.render: env.render()

        pbar_episode.close()

        episodes += 1
        total_frames += t
        pbar_frames.set_description('{0:04d} episodes'.format(episodes))

    pbar_frames.close()



if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    game_config = game_config(args)
    train(args, game_config)
