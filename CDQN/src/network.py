import tensorflow as tf

class Network:
    def __init__(self, args, env, scope):
        self.args = args
        self.env = env

        self.observations = tf.placeholder(tf.float32, shape=(None, self.env.observation_space.shape[0]), name='observations')
        self.actions = tf.placeholder(tf.float32, shape=(None, self.env.action_space.shape[0]), name='actions')
        self.train = tf.placeholder(tf.bool, name='train')

        self.build_model(scope=scope)

    def init_sess(self, sess):
        self.sess = sess

    def build_model(self, scope='CDQN'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            pass
