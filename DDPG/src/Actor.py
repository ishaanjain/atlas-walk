import math
import src.ops as ops
import numpy as np
import tensorflow as tf

class Actor:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        self.inputs = tf.placeholder(tf.float32, shape=(None, self.observation_size), name='observations')
        self.target_inputs = tf.placeholder(tf.float32, shape=(None, self.observation_size), name='target_observations')
        self.QGradsWRTaction = tf.placeholder(tf.float32, shape=(None, self.action_size), name='action_gradients')
        self.is_training = tf.placeholder(tf.bool, name="train")

        # build actor and target actor
        self.build_actor(scope='actor')
        self.build_target_actor(scope='target_actor')

        self.actor_gradients = tf.gradients(self.output, self.variables, -self.QGradsWRTaction)
        self.optimize = tf.train.AdamOptimizer(self.args.actor_learning_rate).apply_gradients(zip(self.actor_gradients, self.variables))

    def init_sess(self, sess):
        self.sess = sess

    def predict_action(self, observations):
        return self.sess.run(self.output, feed_dict={self.inputs: observations,
                                                     self.is_training: False})

    def predict(self, observations):
        return self.sess.run(self.output, feed_dict={self.inputs: observations,
                                                     self.is_training: True})

    def predict_target(self, observations):
        return self.sess.run(self.target_output,
                             feed_dict={self.target_inputs: observations,
                                        self.is_training: True})

    def train(self, observations, gradients):
        return self.sess.run(self.optimize,
                             feed_dict={self.inputs: observations,
                                        self.QGradsWRTaction: gradients,
                                        self.is_training: True})

    def update_target(self):
        self.sess.run(self.soft_updates)

    def build_actor(self, scope=None):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer1 = ops.fully_connected(self.inputs, 400, scope='layer1')
            layer1_act = ops.activation_fn(layer1, scope='layer1_act')
            layer2 = ops.fully_connected(layer1_act, 300, scope='layer2')
            layer2_act = ops.activation_fn(layer2, scope='layer2_act')
            layer3 = ops.fully_connected(layer2_act, self.action_size, w_init=tf.random_uniform_initializer(-3e-3, 3e-3),
                                         b_init=tf.random_uniform_initializer(-3e-3, 3e-3), scope='layer3')
            output = ops.activation_fn(layer3, fn=tf.nn.tanh, scope='output')

            self.output = output

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def build_target_actor(self, scope=None):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer1 = ops.fully_connected(self.target_inputs, 400, scope='layer1')
            layer1_act = ops.activation_fn(layer1, scope='layer1_act')
            layer2 = ops.fully_connected(layer1_act, 300, scope='layer2')
            layer2_act = ops.activation_fn(layer2, scope='layer2_act')
            layer3 = ops.fully_connected(layer2_act, self.action_size, w_init=tf.random_uniform_initializer(-3e-3, 3e-3),
                                         b_init=tf.random_uniform_initializer(-3e-3, 3e-3), scope='layer3')
            output = ops.activation_fn(layer3, fn=tf.nn.tanh, scope='output')

            self.target_output = output

            self.target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self.soft_updates = [tar_param.assign((self.args.tau * net_param) + ((1 - self.args.tau) * tar_param)) for tar_param, net_param in zip(self.target_variables, self.variables)]

    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
