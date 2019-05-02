import os
import sys
import gym
import roboschool
import numpy as np
import tensorflow as tf
from datetime import datetime
from src.NormalizedEnv import NormalizedEnv
from src.CDQN import CDQN


class Test:
    def __init__(self, args):
        self.args = args

    def test(self):
        # get existing or create new checkpoint path
        if self.args.load_model is not None:
            checkpoint = os.path.expanduser('CDQN/checkpoints/' + self.args.load_model)
        else:
            print("Error: Must load in a model to test on")
            sys.exit(1)

        # create the humanoid environment
        env = NormalizedEnv(gym.make("RoboschoolHumanoid-v1"))

        # build graph for CDQN network
        graph = tf.Graph()
        with graph.as_default():
            model = CDQN(self.args, env)
            saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            ckpt = tf.train.latest_checkpoint(checkpoint)
            saver.restore(sess, ckpt)

            for _ in range(self.args.max_episodes):
                state = env.reset()
                rewards = []

                for _ in range(env.spec.timestep_limit):
                    if self.args.render:
                        env.render()

                    action = model.action(state)
                    state, reward, done, _ = env.step(action)
                    rewards.append(reward)

                    if done:
                        break

                total_reward = np.mean(rewards)
                print("Average reward: {}".format(total_reward))
