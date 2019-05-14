import tensorflow as tf
from src.network import Network
from src.OUExploration import OUExploration

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
        self.network.init_sess(sess)
        self.target_network.init_sess(sess)

    def noisy_action(self, state):
        pass

    def perceive(self, state, action, reward, next_state):
        """ store transition in buffer and train on minibatches from buffer """
        self.buffer.add(state, action, reward, next_state)

        if (self.buffer.size() > self.args.batch_size):
            self.train_minibatch()

    def train_minibatch(self):
        """ Train the model on a minibatch of samples from the replay buffer """
        for iteration in range(self.args.update_repeat):
            states, actions, rewards, next_states = self.buffer.get_batch(self.args.batch_size)

            # update network

            # update target network
