import numpy as np
import tensorflow as tf
from src.network import Network
from src.OUExploration import OUExploration
from src.ReplayBuffer import ReplayBuffer

class CDQN:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.exploration = OUExploration(env)
        self.buffer = ReplayBuffer(args.buffer_size)
        self.network = Network(args, env, scope='CDQN')
        self.target_network = Network(args, env, scope='target_CDQN')

    def init_sess(self, sess):
        """ Initialize session of continuous deep Q-learning model """
        self.sess = sess

    def copy_q_params(self):
        """ copy Q network parameters into Q' """
        updates = [q_net_var.assign(net_var) for q_net_var, net_var in zip(self.target_network.variables, self.network.variables)]
        self.sess.run(updates)

    def update_target_params(self):
        """ update Q' networks parameters based on Q networks parameters """
        updates = [self.soft_update(q_net_var, net_var) for q_net_var, net_var in zip(self.target_network.variables, self.network.variables)]
        self.sess.run(updates)

    def soft_update(self, q_net_var, net_var):
        update = (self.args.tau * net_var) + ((1 - self.args.tau) * q_net_var)
        return q_net_var.assign(update)

    def noisy_action(self, state):
        """ predict an action based on the current state and add some exploration noise to it """
        observations = np.expand_dims(state, axis=0)
        actions = self.sess.run(self.network.mu,
                                feed_dict={self.network.observations: observations,
                                           self.network.train: False})

        return self.exploration.add_noise(actions[0])

    def perceive(self, state, action, reward, next_state):
        """ store transition in buffer and train on minibatches from buffer """
        self.buffer.add(state, action, reward, next_state)

        if (self.buffer.size() > self.args.batch_size):
            self.train_minibatch()

    def train_minibatch(self):
        print('step')
        """ Train the model on a minibatch of samples from the replay buffer """
        for iteration in range(self.args.update_repeat):
            states, actions, rewards, next_states = self.buffer.get_batch(self.args.batch_size)

            # calculate value for target network
            target_V = self.sess.run(self.target_network.V,
                                     feed_dict={self.target_network.observations: next_states,
                                                self.target_network.train: False})

            # calculate target y
            target_y = rewards + (self.args.discount_rate * np.squeeze(target_V))

            # update network
            self.sess.run([self.network.optimize], feed_dict={self.network.observations: states,
                                                              self.network.actions: actions,
                                                              self.network.target_y: target_y,
                                                              self.network.train: self.args.train})

            # update target network
            self.update_target_params()
