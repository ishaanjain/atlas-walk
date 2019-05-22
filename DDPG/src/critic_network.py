import tensorflow as tf
import numpy as np
import math
import src.ops as ops

LAYER1_SIZE = 400
LAYER2_SIZE = 300
L2 = 0.01


class CriticNetwork:
    """docstring for CriticNetwork"""

    def __init__(self, args, state_dim, action_dim):
        self.args = args
        self.time_step = 0
        # create q network
        self.state_input, \
        self.action_input, \
        self.q_value_output, \
        self.net = self.create_q_network(state_dim, action_dim)

        # create target q network (the same structure with q network)
        self.target_state_input, \
        self.target_action_input, \
        self.target_q_value_output, \
        self.target_variables = self.create_target_q_network(state_dim, action_dim, self.net)

        self.create_training_method()

        self.init_updates()
        self.soft_updates(self.args.tau)

    def init_updates(self):
        self.init_updates = [tar_param.assign(net_param) for tar_param, net_param in
                             zip(self.target_variables, self.net)]

    def soft_updates(self, tau):
        self.soft_updates = [tar_param.assign((tau * net_param) + ((1 - tau) * (tar_param))) for tar_param, net_param in
                             zip(self.target_variables, self.net)]

    def init_sess(self, sess):
        self.sess = sess
        self.sess.run(self.init_updates)

    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder("float", [None, 1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(self.args.critic_learning_rate).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

    def create_q_network(self, state_dim, action_dim):
        # the layer size could be changed
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        with tf.variable_scope("critic", reuse=tf.AUTO_REUSE):
            state_input = tf.placeholder("float", [None, state_dim])
            action_input = tf.placeholder("float", [None, action_dim])

            W1 = self.variable([state_dim, layer1_size], state_dim)
            b1 = self.variable([layer1_size], state_dim)
            W2 = self.variable([layer1_size, layer2_size], layer1_size + action_dim)
            W2_action = self.variable([action_dim, layer2_size], layer1_size + action_dim)
            b2 = self.variable([layer2_size], layer1_size + action_dim)
            W3 = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-3, 3e-3))
            b3 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))

            layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
            layer2 = tf.nn.relu(tf.matmul(layer1, W2) + tf.matmul(action_input, W2_action) + b2)
            q_value_output = tf.identity(tf.matmul(layer2, W3) + b3)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")

        return state_input, action_input, q_value_output, variables

    def create_target_q_network(self, state_dim, action_dim, net):
        # the layer size could be changed
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        with tf.variable_scope("target_critic", reuse=tf.AUTO_REUSE):
            W1 = self.variable([state_dim, layer1_size], state_dim)
            state_input = tf.placeholder("float", [None, state_dim])
            action_input = tf.placeholder("float", [None, action_dim])

            b1 = self.variable([layer1_size], state_dim)
            W2 = self.variable([layer1_size, layer2_size], layer1_size + action_dim)
            W2_action = self.variable([action_dim, layer2_size], layer1_size + action_dim)
            b2 = self.variable([layer2_size], layer1_size + action_dim)
            W3 = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-3, 3e-3))
            b3 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))

            layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
            layer2 = tf.nn.relu(tf.matmul(layer1, W2) + tf.matmul(action_input, W2_action) + b2)
            q_value_output = tf.identity(tf.matmul(layer2, W3) + b3)

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_critic")

        return state_input, action_input, q_value_output, variables

    def update_target(self):
        self.sess.run(self.soft_updates)

    def train(self, y_batch, state_batch, action_batch):
        self.time_step += 1
        self.sess.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.state_input: state_batch,
            self.action_input: action_batch
        })

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })[0]

    def target_q(self, state_batch, action_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch
        })

    def q_value(self, state_batch, action_batch):
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch})

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))


    def randInit(self, f):
        return tf.random_uniform_initializer(-1 / math.sqrt(f), 1 / math.sqrt(f))
