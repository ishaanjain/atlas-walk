import sys
import numpy as np
import tensorflow as tf
from src.Actor import Actor
from src.Critic import Critic
from src.ReplayBuffer import ReplayBuffer
from src.OUExploration import OUNoise

class DDPG:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # build model
        self.exploration = OUNoise(self.action_dim)
        self.ReplayBuffer = ReplayBuffer(self.args.buffer_size)
        self.Actor = Actor(args, env)
        self.Critic = Critic(args, env)

    def init_sess(self, sess):
        self.sess = sess
        self.Actor.init_sess(sess)
        self.Critic.init_sess(sess)

    def action(self, state):
        states = np.expand_dims(state, axis=0)
        actions = self.Actor.predict_action(states)
        return actions[0]

    def noisy_action(self, state):
        return self.action(state) + self.exploration.noise()

    def perceive(self, state, action, reward, done, next_state):
        self.ReplayBuffer.add(state, action, reward, done, next_state)

        if (self.ReplayBuffer.size() > 10000):
            self.train_minibatch()

    def train_minibatch(self):
        states, actions, rewards, dones, next_states = self.ReplayBuffer.get_batch(self.args.batch_size)

        actions = np.reshape(actions, [self.args.batch_size, self.action_dim])
        dones = np.asarray([int(done) for done in dones], dtype=np.float32)

        target_q = self.Critic.predict_target(next_states,
                                              self.Actor.predict_target(next_states))
        target_y = rewards + ((1 - dones) * (self.args.discount_rate * np.squeeze(target_q)))

        self.Critic.train(states, actions, target_y)

        acts = self.Actor.predict(states)
        grads = self.Critic.calcActionGrads(states, acts)
        self.Actor.train(states, grads[0])

        # perform soft updates on target networks
        self.Actor.update_target()
        self.Critic.update_target()
