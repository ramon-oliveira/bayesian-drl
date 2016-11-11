import sys
import argparse

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

args = parser.parse_args(sys.argv[1:])
