from __future__ import print_function
import random
import numpy as np
import collections
import math
import gym
import tensorflow as tf
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def q_network(input_shape, num_atoms, num_actions, lr):
    Input = tf.keras.layers.Input(shape=(input_shape))
    Layer = tf.keras.layers.Conv2D(32, (8, 8), strides=(4,4), activation='relu')(Input)
    Layer = tf.keras.layers.Conv2D(64, (4, 4), strides=(2,2), activation='relu')(Layer)
    Layer = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(Layer)
    Layer = tf.keras.layers.Flatten()(Layer)
    Layer = tf.keras.layers.Dense(256, activation='relu')(Layer)
    output_actions = []
    for i in range(num_actions):
        output_actions.append(tf.keras.layers.Dense(num_atoms, activation='softmax')(Layer))
    model = tf.keras.models.Model(inputs=Input, outputs=output_actions)
    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=adam)
    return model

class Main:
    def __init__(self, game, state_size, num_actions, num_atoms):
        self.game = game
        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = 0.99
        self.lr = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 100
        self.explore = 1000
        self.frame_per_action = 4
        self.update_target_freq = 1000
        self.timestep_per_train = 4
        self.episodes = 100
        self.num_atoms = num_atoms
        self.v_max = 10
        self.v_min = -10
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]
        self.memory = collections.deque()
        self.max_memory = 1000
        self.model = None
        self.target_model = None

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = env.action_space.sample()
        else:
            action = self.choose_optimal_action(state)
        return action

    def choose_optimal_action(self, state):
        y = self.model.predict(state)
        y_ = np.vstack(y)
        q = np.sum(np.multiply(y_, np.array(self.z)), axis=1)
        action = np.argmax(q)
        return action

    def replay_memory(self, s_t, action, r_t, s_t1, is_terminated, t):
        self.memory.append((s_t, action, r_t, s_t1, is_terminated))
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * math.exp(-t/self.explore)
        if len(self.memory) > self.max_memory:
            self.memory.popleft()
        if t % self.update_target_freq == 0:
            self.update_target_model()

    def train(self):
#        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        num_samples = min(self.batch_size , len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)
        state_inputs = np.zeros(((num_samples,) + self.state_size))
        next_states = np.zeros(((num_samples,) + self.state_size))
        action, reward, done = [], [], []
        for i in range(num_samples):
            state_inputs[i,:,:,:] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            next_states[i,:,:,:] = replay_samples[i][3]
            done.append(replay_samples[i][4])
        z = self.model.predict(next_states)
        z_ = self.target_model.predict(next_states)
        m_prob = self.model.predict(state_inputs)
        optimal_action = []
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) # length (num_atoms x num_actions)
        q = q.reshape((num_samples, num_actions), order='F')
        optimal_action = np.argmax(q, axis=1)
        for i in range(num_samples):
            if done[i]:
                Tz = min(self.v_max, max(self.v_min, reward[i]))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[action[i]][i][int(m_l)] += (m_u - bj)
                m_prob[action[i]][i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[action[i]][i][int(m_l)] += z_[optimal_action[i]][i][j] * (m_u - bj)
                    m_prob[action[i]][i][int(m_u)] += z_[optimal_action[i]][i][j] * (bj - m_l)
        loss = self.model.fit(state_inputs, m_prob, batch_size=self.batch_size, epochs=1, verbose=0)
        return loss.history['loss']

if __name__ == '__main__':
    games = ['Atlantis-v0', 'Alien-v0', 'Amidar-v0', 'Berzerk-v0', 'CrazyClimber-v0']
    games = ['Atlantis-v0']
    filename = 'result.txt'
    file = open(filename,'w')
    for game in games:
        Max_t = 100000
        env = gym.make(game)
        x_t = env.reset()
        num_actions = env.action_space.n
        rows, cols, channels = x_t.shape
        num_atoms = 51
        state_size = (rows, cols, channels)
        agent = Main(game, state_size, num_actions, num_atoms)
        agent.model = q_network(state_size, num_atoms, num_actions, agent.lr)
        agent.target_model = q_network(state_size, num_atoms, num_actions, agent.lr)
        file.write(str(game) + ': ')
        if len(sys.argv)>1 and sys.argv[1] == 'load':
            print('Load model')
            agent.model.load_weights('./weights.dxh')
        is_terminated = 0
        epsilon = agent.initial_epsilon
        GAME = 0
        t = 0
        R = 0
        while GAME < agent.episodes:
            loss = 0
            a_t = np.zeros([num_actions])
            action_idx  = agent.choose_action(x_t)
            a_t[action_idx] = 1
            a_t = a_t.astype(int)
            x_t1, r_t, is_terminated, info = env.step(action_idx)
#            env.render()
            R += r_t
            if t % 10 == 0:
                print(t, R)
            if (is_terminated):
                GAME += 1
                print ('Episode Finish ', GAME)
                result = '(episode=' + str(GAME) +' Reward=' + str(R) + ' t=' + str(t) + '),'
                file.write(result)
                x_t1 = env.reset()
            x_t1 = np.reshape(x_t1, (1, rows, cols, channels))
            agent.replay_memory(x_t, action_idx, r_t, x_t1, is_terminated, t)
            if t > agent.observe and t % agent.timestep_per_train == 0:
                loss = agent.train()
            x_t = x_t1
            t += 1
            if t % 1000 == 0:
                print('Save model')
                agent.model.save_weights('./weights.dxh', overwrite=True)
            state = ''
            if t <= agent.observe:
                state = 'observe'
            elif t > agent.observe and t <= agent.observe + agent.explore:
                state = 'explore'
            else:
                state = 'train'
            if (is_terminated):
                print('TIME', t, '/ GAME', GAME, '/ STATE', state,'/ Reward',R)
                R = 0
            if t >= Max_t:
                print('Write result and break')
#                result = game + ',' + str(R) + ',' + str(t) + '\n'
                result = '(episode=' + str(GAME) +' Reward=' + str(R) + ' t=' + str(t) + '),'
                file.write(result)
                break
        result = '(episode=' + str(GAME) +' Reward=' + str(R) + ' t=' + str(t) + ')\n'
        file.write(result)
