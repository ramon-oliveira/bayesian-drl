import gym
import numpy as np
from skimage import color
from skimage.transform import resize
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import Progbar
import time

def preprocess(obs):
    #max_frames = np.maximum.reduce(obs) #quantos frame???
    xyz = color.rgb2xyz(obs)
    y = xyz[:,:,:,1]
    small = resize(y,(4,84,84))
    state = small.reshape(1, 4, 84, 84)
    return state

def populate_memory(env, D, size=5000):
    n = len(D)
    while n < size:
        obs = np.zeros([m]+list(env.observation_space.shape))
        ob = env.reset()
        ob0 = ob    #this is for fixing the even odd frames problem
        obs[0] = ob0
        State0 = preprocess(obs)
        treward = 0
        for i in range(3000):
            action = env.action_space.sample()
            (ob1, reward, done, _info) = env.step(action)
            ob = np.maximum.reduce([ob0,ob1])   #even odd frame
            ob0 = ob1                           #problem
            obs[1:m] = obs[0:m-1]
            obs[0] = ob
            treward += reward
            ##Set State' = ob and preprocess State'
            State1 = preprocess(obs)
            D.insert(0,[State0,action,reward,State1])
            if len(D)>replay_size:
                D.pop()
            if done:break;
            n = len(D)
            if(len(D)%100==0):
                print("Replay memory lenght",len(D))
            if n > size: break
    return n

def act(env, obs):
    return env.action_space.sample()

if __name__ == '__main__':

    env = gym.make('Pong-v0')

    Total_Frames = 50000000
    Max_ep = 1000000
    num_steps = 5000
    #Lets assume the following parameters are the same

    ##Initialize replay memory D to capacity N (1000000)
    D = list()  ###### TODO: use numpy array
    ##Initialize action value function with random with random weights
    print("creating Q network")
    Q = Sequential()
    Q.add(Convolution2D(32, 8, 8, border_mode='valid', input_shape=(4,84,84)))
    Q.add(MaxPooling2D(pool_size=(4, 4)))
    Q.add(Dropout(0.5))
    Q.add(Activation('relu'))
    Q.add(Convolution2D(64, 4, 4, border_mode='valid'))
    Q.add(MaxPooling2D(pool_size=(2, 2)))
    Q.add(Dropout(0.5))
    Q.add(Activation('relu'))
    Q.add(Convolution2D(64, 3, 3, border_mode='valid'))
    Q.add(Dropout(0.5))
    Q.add(Activation('relu'))
    Q.add(Flatten())
    Q.add(Dense(512))
    Q.add(Activation('relu'))
    Q.add(Dense(6))
    Q.add(Activation('tanh')) #aqui o certo e um htan

    sgd = SGD()
    print("compiling Q network")
    Q.compile(loss="mean_squared_error", optimizer=sgd)
    Q.summary()
    ##Initialize target action-value function ^Q with same wieghts
    print("copying Q to Q_target")
    Q_target = model_from_yaml(Q.to_yaml())
    Q_target.set_weights(Q.get_weights())

    e = 0 #e-greedy policy, drops from e=1 to e=0.1
    k = 4 #The agent sees and selects an action every kth frame
    m = 4 #Number of frames looked at each moment
    replay_size = 10000
    batch_size = 32
    gama = 0.99  #discount factor for future rewards Q function
    C = 16
    render = False

    ##Populates replay memory with some random sequences
    print("populating memory")
    populate_memory(env, D, size=100)

    print("Starting to Train")

    S0 = np.zeros([batch_size, m, 84, 84])
    S1 = np.zeros([batch_size, m, 84, 84])
    A = np.zeros([batch_size], dtype=np.int)
    R = np.zeros([batch_size])

    ##Starts Playing and training
    for episodes in range(Max_ep):
        ##Initialize sequence game and sequence s1 pre-process sequence
        obs = np.zeros([m]+list(env.observation_space.shape))
        ob = env.reset()
        ob0 = ob    #this is for fixing the even odd frames problem
        obs[0] = ob0
        State0 = preprocess(obs)
        treward = 0
        pb = Progbar(1050)
        train_loss = 1e10
        for t in range(num_steps):
            if t%k==0:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    in_obs = preprocess(obs)
                    qval = Q.predict(in_obs, batch_size=1, verbose=0)
                    action = qval.argmax()
            (ob1, reward, done, _info) = env.step(action)
            ob = np.maximum.reduce([ob0,ob1])   #even odd frame
            ob0 = ob1                           #problem
            obs[1:m] = obs[0:m-1]
            obs[0] = ob
            treward += reward

            ##Set State' = ob and preprocess State'
            State1 = preprocess(obs)

            ##### TODO: numpy array
            D.insert(0,[State0,action,reward,State1])
            if len(D)>replay_size:
                D.pop()

            ##train only every few actions
            if t%4==0:
                ##Sample random minibatch of transitions from D
                batch = [D[i] for i in sorted(np.random.randint(0,len(D),32))]
                for i, (s0, a, r, s1) in enumerate(batch):
                    S0[i] = s0
                    S1[i] = s1
                    A[i] = a
                    R[i] = r
                y_Q = Q.predict_on_batch(S0)
                y_Q_target = R +gama*np.max(Q_target.predict_on_batch(S1),1)
                y_Q[:, A] = y_Q_target

                ##train on batch
                train_loss = Q.train_on_batch(S0,y_Q)
                pb.update(t, [['mse', train_loss]])

            ##set e-greedy policy adjust
            if e > 0.1:
                e -= 0.0000009

            ##update Q_target every C trains
            if t and t%C==0:
                Q_target.set_weights(Q.get_weights())

            if render:
                env.render()
            if done:break;
        # pb.update(num_steps, [['mse', train_loss]])
        Q.save_weights('tst.h5', overwrite=True)
        print("Episode",episodes+1,"\tpoints =",treward,"\tframes",t+1)
        print("\n")


