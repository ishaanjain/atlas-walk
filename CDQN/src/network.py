import src.ops as ops
import tensorflow as tf

class Network:
    def __init__(self, args, env, scope):
        self.args = args
        self.env = env
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        self.observations = tf.placeholder(tf.float32, shape=(None, self.observation_size), name='observations')
        self.actions = tf.placeholder(tf.float32, shape=(None, self.action_size), name='actions')
        self.target_y = tf.placeholder(tf.float32, shape=(None), name='target_y')
        self.train = tf.placeholder(tf.bool, name='train')

        self.build_model(scope=scope)

    def build_model(self, scope='CDQN'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # create 2 hidden layers as basis for value and advantage functions
            with tf.variable_scope('hidden', reuse=tf.AUTO_REUSE):
                hidden = self.observations

                for idx in range(2):
                    hidden = ops.fully_connected(hidden, 100, w_init=tf.random_uniform_initializer(-0.05, 0.05),
                                                is_training=self.train, scope='hid'+str(idx))
                    hidden = ops.activation_fn(hidden, fn=tf.nn.tanh, scope='hidden_act'+str(idx))

            # value function
            with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
                self.V = ops.fully_connected(hidden, 1, w_init=tf.random_uniform_initializer(-0.05, 0.05),
                                            is_training=self.train, scope='V')

            # neural network layers for advantage function
            with tf.variable_scope('advantage', reuse=tf.AUTO_REUSE):
                linear_layer = ops.fully_connected(hidden, self.action_size**2,
                                                   w_init=tf.random_uniform_initializer(-0.05, 0.05),
                                                   is_training=self.train, scope='linear_layer')

                mu = ops.fully_connected(hidden, self.action_size, w_init=tf.random_uniform_initializer(-0.05, 0.05),
                                        is_training=self.train, scope='mu')
                self.mu = ops.activation_fn(mu, fn=tf.nn.tanh, scope='mu_act')

            # create lower triangular matrix with exponentiated diagonal from linear neural network layer
            self.L = ops.createLowerTriangle(linear_layer, self.action_size)

            # create positive definite matrix by multipting L and L transpose
            self.P = tf.matmul(self.L, tf.transpose(self.L, (0, 2, 1)))

            # create advantage function
            diff = tf.expand_dims(self.actions - mu, -1)
            diff_transpose = tf.transpose(diff, (0, 2, 1))
            self.A = -0.5 * tf.reshape(tf.matmul(diff_transpose, tf.matmul(self.P, diff)), [-1, 1])

            with tf.variable_scope('Q', reuse=tf.AUTO_REUSE):
                self.Q = self.A + self.V

            with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
                self.loss = ops.mean_squared_error(tf.squeeze(self.Q), self.target_y, scope='loss')
                self.optimize = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.loss)

            self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
