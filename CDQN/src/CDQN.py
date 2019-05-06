import tensorflow as tf

class CDQN:
    def __init__(self, args, env):
        self.args = args
        self.env = env
