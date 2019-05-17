import tensorflow as tf
import numpy as np
import math
import src.ops as ops


# Hyper Parameters
LAYER1_SIZE = 400
LAYER2_SIZE = 300


class ActorNetwork:
    """docstring for ActorNetwork"""

    def __init__(self, args, sess, state_dim, action_dim):
        self.args = args
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create actor network
        self.state_input, self.action_output, self.net = self.create_network(state_dim, action_dim, self.args.train)

        # create target actor network
        self.target_state_input, self.target_action_output, self.target_update, self.target_net = self.create_target_network(
            state_dim, action_dim, self.args.train, self.net)

        # define training rules
        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        self.update_target()

    # self.load_network()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(self.args.actor_learning_rate).apply_gradients(zip(self.parameters_gradients, self.net))

    def create_network(self, state_dim, action_dim, is_train):
        with tf.variable_scope("actor", reuse=tf.AUTO_REUSE):
            state_input = tf.placeholder("float", [None, state_dim])

            # W1 = self.variable([state_dim, layer1_size], state_dim)
            # b1 = self.variable([layer1_size], state_dim)
            # W2 = self.variable([layer1_size, layer2_size], layer1_size)
            # b2 = self.variable([layer2_size], layer1_size)
            # W3 = tf.Variable(tf.random_uniform([layer2_size, action_dim], -3e-3, 3e-3))
            # b3 = tf.Variable(tf.random_uniform([action_dim], -3e-3, 3e-3))

            # layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
            # layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
            # action_output = tf.tanh(tf.matmul(layer2, W3) + b3)

            W1 = tf.glorot_uniform_initializer()
            b1 = tf.constant_initializer(0.0)
            W2 = tf.glorot_uniform_initializer()
            b2 = tf.constant_initializer(0.0)
            W3 = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
            b3 = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

            layer1 = ops.fully_connected(state_input, LAYER1_SIZE, w_init=W1, b_init=b1, is_training=is_train, scope="layer1")
            layer1_relu = ops.activation_function(layer1, scope="layer1_relu")
            layer2 = ops.fully_connected(layer1_relu, LAYER2_SIZE, w_init=W2, b_init=b2, is_training=is_train, scope="layer2")
            layer2_relu = ops.activation_function(layer2, scope="layer2_relu")
            layer3 = ops.fully_connected(layer2_relu, action_dim, w_init=W3, b_init=b3, is_training=is_train, scope="layer3")
            action_output = ops.activation_function(layer3, scope="action_output", activation=tf.nn.tanh)

        variables = tf.contrib.framework.get_variables(scope="actor")
        print(variables)

        return state_input, action_output, variables

    def create_target_network(self, state_dim, action_dim, is_train, net):
        with tf.variable_scope("actor", reuse=tf.AUTO_REUSE):
            state_input = tf.placeholder("float", [None, state_dim])
            ema = tf.train.ExponentialMovingAverage(decay=1 - self.args.tau)
            target_update = ema.apply(net)
            target_net = [ema.average(x) for x in net]

            # layer1 = ops.fully_connected(state_input, LAYER1_SIZE, w_init=W1, b_init=b1, is_training=is_train, scope="layer1")
            layer1_relu = ops.activation_function(tf.matmul(state_input, target_net[0]) + target_net[1], scope="layer1_relu")
            # layer2 = ops.fully_connected(layer1_relu, LAYER2_SIZE, w_init=W2, b_init=b2, is_training=is_train, scope="layer2")
            layer2_relu = ops.activation_function(tf.matmul(layer1_relu, target_net[6]) + target_net[7], scope="layer2_relu")
            # layer3 = ops.fully_connected(layer2_relu, action_dim, w_init=W3, b_init=b3, is_training=is_train, scope="layer3")
            action_output = ops.activation_function(tf.matmul(layer2_relu, target_net[12]) + target_net[13], scope="action_output", activation=tf.nn.tanh)

        # variables = tf.contrib.framework.get_variables(scope="actor")

        return state_input, action_output, target_update, target_net

        # state_input = tf.placeholder("float", [None, state_dim])
        # ema = tf.train.ExponentialMovingAverage(decay=1 - self.args.tau)
        # target_update = ema.apply(net)
        # target_net = [ema.average(x) for x in net]
        #
        # layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        # layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + target_net[3])
        # action_output = tf.tanh(tf.matmul(layer2, target_net[4]) + target_net[5])
        #
        # return state_input, action_output, target_update, target_net

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch
        })
        # print(self.sess.run([self.net[0], self.net[6], self.net[12]]))


    def actions(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state_batch
        })

    def action(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state]
        })[0]

    def target_actions(self, state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch
        })

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))


'''
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"
	def save_network(self,time_step):
		print 'save actor-network...',time_step
		self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)
'''
