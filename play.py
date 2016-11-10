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
parser.add_argument('--weights-file', dest='weights_file', required=True)

parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)
parser.add_argument('--replay-size', dest='replay_size', type=int, default=10000)
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


def play(args, game_config):
    state_shape = game_config['state_shape']
    env = game_config['enviroment']
    preprocessing = game_config['preprocessing']
    actions = game_config['actions']

    # Initialize action value function with random with random weights
    model = create_model(args, game_config)
    print('weights/{0}/{1}'.format(args.game, args.weights_file))
    model.load_weights('weights/{0}/{1}'.format(args.game, args.weights_file))

    # keep track variables
    t = 0
    epsilon = 0.05
    done = False
    obs = np.zeros(state_shape, dtype=np.int8)
    while not done:
        # if (t % args.frame_skip) == 0:
        #     if np.random.rand() < epsilon:
        #         action_idx = np.random.randint(low=0, high=len(actions))
        #     else:
        qval = model.predict(np.array([obs]), verbose=0)
        action_idx = qval.argmax()

        ob, reward, done, info = env.step(actions[action_idx])

        if (t % args.frame_skip) == 0:
            # update state
            obs[1:] = obs[:-1]
            obs[0] = preprocessing(ob)

        t += 1
        if args.render: env.render()


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    game_config = game_config(args)
    play(args, game_config)
