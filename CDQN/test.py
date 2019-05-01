import os
import sys
import gym
import roboschool
import tensorflow as tf
from datetime import datetime

class Test:
    def __init__(self, args):
        self.args = args

    def test(self):
        # get existing or create new checkpoint path
        if self.args.load_model is not None:
            checkpoint = 'checkpoints/' + self.args.load_model
        else:
            print("Error: Must load in a model to test on")
            sys.exit(1)

        # build graph for CDQN network
        graph = tf.Graph()
        with graph.as_default():
            pass

        with tf.Session(graph=graph) as sess:
            pass
