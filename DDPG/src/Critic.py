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
        #self.init_updates()
        #self.soft_updates(self.args.tau)

        weight_decay = tf.add_n([0.01 * tf.nn.l2_loss(var) for var in self.variables])
        self.loss = ops.mean_squared_error(tf.squeeze(self.output), self.y, scope='loss') + weight_decay
        self.optimize = tf.train.AdamOptimizer(self.args.critic_learning_rate).minimize(self.loss)

        self.QGradsWRTactions = tf.gradients(self.output, self.actions)

    def init_sess(self, sess):
        self.sess = sess

    def init_updates(self):
        self.init_updates = [tar_param.assign(net_param) for tar_param, net_param in zip(self.target_variables, self.variables)]

    def soft_updates(self, tau):
        self.soft_updates = [tar_param.assign((tau * net_param) + ((1 - tau) * (tar_param))) for tar_param, net_param in zip(self.target_variables, self.variables)]

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
            layer2 = ops.fully_connected(layer1_act, 300, scope='layer2')
            actions_layer = ops.fully_connected(self.actions, 300, scope='actions_layer')
            layer2_act = ops.activation_fn(layer2 + actions_layer, scope='layer2_act')
            layer3 = ops.fully_connected(layer2_act, 1, w_init=tf.random_uniform_initializer(-3e-3, 3e-3),
                                         b_init=tf.random_uniform_initializer(-3e-3, 3e-3), scope='layer3')
            output = ops.activation_fn(layer3, fn=tf.nn.tanh, scope='output')

            self.output = output
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def build_target_critic(self, scope=None):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            ema = tf.train.ExponentialMovingAverage(decay=1-self.args.tau)
            self.soft_updates = ema.apply(self.variables)
            target_net = [ema.average(x) for x in self.variables]

            layer1 = tf.nn.relu(tf.matmul(self.target_inputs,target_net[0]) + target_net[1])
            layer2 = tf.matmul(layer1,target_net[2]) + target_net[3]
            actions_layer = tf.matmul(self.target_actions,target_net[4]) + target_net[5]
            layer2_out = tf.nn.relu(layer2 + actions_layer)
            q_value_output = tf.identity(tf.matmul(layer2,target_net[6]) + target_net[7])

            self.target_output = q_value_output
            self.target_variables = target_net

            #layer1 = ops.fully_connected(self.target_inputs, 400, scope='layer1')
            #layer1_act = ops.activation_fn(layer1, scope='layer1_act')
            #layer2 = ops.fully_connected(layer1_act, 300, scope='layer2')
            #actions_layer = ops.fully_connected(self.target_actions, 300, scope='actions_layer')
            #layer2_act = ops.activation_fn(layer2 + actions_layer, scope='layer2_act')
            #layer3 = ops.fully_connected(layer2_act, 1, w_init=tf.random_uniform_initializer(-3e-3, 3e-3),
            #                             b_init=tf.random_uniform_initializer(-3e-3, 3e-3), scope='layer3')
            #output = ops.activation_fn(layer3, fn=tf.nn.tanh, scope='output')

            #self.target_output = output
            #self.target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
