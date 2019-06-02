import math
import src.ops as ops
import numpy as np
import tensorflow as tf

class Critic:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        self.inputs = tf.placeholder(tf.float32, shape=(None, self.observation_size), name='observations')
        self.target_inputs = tf.placeholder(tf.float32, shape=(None, self.observation_size), name='target_observations')
        self.actions = tf.placeholder(tf.float32, shape=(None, self.action_size), name='actions')
        self.target_actions = tf.placeholder(tf.float32, shape=(None, self.action_size), name='target_actions')
        self.y = tf.placeholder(tf.float32, shape=(None), name='target_critic')

        # build critic and target critic network
        self.build_critic(scope='critic')
        self.build_target_critic(scope='target_critic')

        weight_decay = tf.add_n([0.01 * tf.nn.l2_loss(var) for var in self.variables])
        self.loss = ops.mean_squared_error(tf.squeeze(self.output), self.y, scope='loss') + weight_decay
        self.optimize = tf.train.AdamOptimizer(self.args.critic_learning_rate).minimize(self.loss)

        self.QGradsWRTactions = tf.gradients(self.output, self.actions)

    def init_sess(self, sess):
        self.sess = sess

    def predict(self, observations, actions):
        return self.sess.run(self.output, feed_dict={self.inputs: observations,
                                                     self.actions: actions})

    def predict_target(self, observations, actions):
        return self.sess.run(self.target_output,
                             feed_dict={self.target_inputs: observations,
                                        self.target_actions: actions})

    def train(self, observations, actions, target_y):
        self.sess.run(self.optimize, feed_dict={self.inputs: observations,
                                                self.actions: actions,
                                                self.y: target_y})

    def calcActionGrads(self, observations, actions):
        return self.sess.run(self.QGradsWRTactions,
                             feed_dict={self.inputs: observations,
                                        self.actions: actions})

    def update_target(self):
        self.sess.run(self.soft_updates)

    def build_critic(self, scope=None):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer1 = ops.fully_connected(self.inputs, 400, scope='layer1')
            layer1_act = ops.activation_fn(layer1, scope='layer1_act')
            layer2 = ops.fully_connected(layer1_act, 300, use_bias=False, scope='layer2')
            actions_layer = ops.fully_connected(self.actions, 300, scope='actions_layer')
            layer2_act = ops.activation_fn(layer2 + actions_layer, scope='layer2_act')
            output = ops.fully_connected(layer2_act, 1, w_init=tf.random_uniform_initializer(-3e-3, 3e-3),
                                         b_init=tf.random_uniform_initializer(-3e-3, 3e-3), scope='layer3')

            self.output = tf.identity(output)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def build_target_critic(self, scope=None):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer1 = ops.fully_connected(self.target_inputs, 400, scope='layer1')
            layer1_act = ops.activation_fn(layer1, scope='layer1_act')
            layer2 = ops.fully_connected(layer1_act, 300, use_bias=False, scope='layer2')
            actions_layer = ops.fully_connected(self.target_actions, 300, scope='actions_layer')
            layer2_act = ops.activation_fn(layer2 + actions_layer, scope='layer2_act')
            output = ops.fully_connected(layer2_act, 1, w_init=tf.random_uniform_initializer(-3e-3, 3e-3),
                                         b_init=tf.random_uniform_initializer(-3e-3, 3e-3), scope='layer3')

            self.target_output = tf.identity(output)

            self.target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self.soft_updates = [tar_param.assign((self.args.tau * net_param) + ((1 - self.args.tau) * tar_param)) for tar_param, net_param in zip(self.target_variables, self.variables)]

    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
