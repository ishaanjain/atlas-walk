import tensorflow as tf

class DDPG:
    def __init__(self, args, env):
        self.args = args
        self.env = env
