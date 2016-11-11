import gym
import numpy as np
import utils
from config import args
from models import create_model

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


def play(args, game_config):
    state_shape = game_config['state_shape']
    env = game_config['enviroment']
    preprocessing = game_config['preprocessing']
    actions = game_config['actions']

    # Initialize action value function with random with random weights
    model = create_model(args, game_config)
    
    # keep track variables
    t = 0
    epsilon = 0.05
    done = False
    obs = np.zeros(state_shape, dtype=np.int8)
    while not done:
        if (t % args.frame_skip) == 0:
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(low=0, high=len(actions))
            else:
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
    game_config = game_config(args)
    play(args, game_config)
