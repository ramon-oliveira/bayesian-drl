import os
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
