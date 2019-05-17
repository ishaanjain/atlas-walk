import os
import sys
import gym
import roboschool
import numpy as np
import tensorflow as tf
from datetime import datetime
from src.NormalizedEnv import NormalizedEnv
from src.CDQN import CDQN


class Train:
    def __init__(self, args):
        self.args = args

    def train(self):
        # get existing or create new checkpoint path
        if self.args.load_model is not None:
            checkpoint = os.path.expanduser('CDQN/checkpoints/' + self.args.load_model)
        else:
            checkpoint_name = datetime.now().strftime('%d%m%Y-%H%M')
            checkpoint = os.path.expanduser('CDQN/checkpoints/' + checkpoint_name)

            try:
                os.makedirs(checkpoint)
            except os.error:
                print('Error: Failed to make new checkpoint directory')
                sys.exit(1)

        # create the humanoid environment
        env = NormalizedEnv(gym.make("RoboschoolHumanoid-v1"))

        # build graph for CDQN network
        graph = tf.Graph()
        with graph.as_default():
            model = CDQN(self.args, env)
            saver = tf.train.Saver(max_to_keep=2)

        with tf.Session(graph=graph) as sess:
            if self.args.load_model is not None: # restore graph and last saved training step
                ckpt = tf.train.get_checkpoint_state(checkpoint)
                meta_graph_path = ckpt.model_checkpoint_path + '.meta'
                restore = tf.train.import_meta_graph(meta_graph_path)
                restore.restore(sess, tf.train.latest_checkpoint(checkpoint))
                step = int(meta_graph_path.split("-")[2].split(".")[0])
                start_episode = step // self.args.max_steps
                start_step = step % self.args.max_steps
            else:
                sess.run(tf.global_variables_initializer())
                start_episode = 0
                start_step = 1

            try:
                model.init_sess(sess)
                model.copy_q_params()

                for episode in range(start_episode, self.args.max_episodes):
                    state = env.reset()
                    rewards = []

                    for step in range(start_step, self.args.max_steps+1):
                        if self.args.render:
                            env.render()

                        action = model.noisy_action(state)
                        next_state, reward, done, _ = env.step(action)
                        model.perceive(state, action, reward, next_state)
                        state = next_state
                        rewards.append(reward)

                        # save once a checkpoint is reached
                        if (((episode*self.args.max_steps) + step) % self.args.checkpoint_frequency == 0):
                            print('Saving models training progress to the `checkpoints` directory...')
                            save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=((episode*self.args.max_steps) + step))
                            print('Model saved as {}'.format(save_path))

                        if (((episode*self.args.max_steps) + step) % self.args.display_frequency == 0):
                            print ('Episode {} - Step {} - Average Reward: {}'.format(episode, step, np.mean(reward)))

                        if done:
                            break

                    model.exploration.reset()  # reset the exploration strategy after every episode

                    total_reward = np.mean(rewards)
                    print("Episode {} - Average reward: {}".format(episode, total_reward))

                env.close()

            except KeyboardInterrupt: # save training progress upon user exit
                env.close()
                print('Saving models training progress to the `checkpoints` directory...')
                save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
                print('Model saved as {}'.format(save_path))
                sys.exit(0)
