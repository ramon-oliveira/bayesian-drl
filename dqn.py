import gym
import tqdm
import numpy as np
import utils
from config import args
from models import create_model
from keras.models import model_from_yaml


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
    game_config = game_config(args)
    train(args, game_config)
