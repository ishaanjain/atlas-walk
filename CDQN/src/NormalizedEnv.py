import gym
import roboschool
import numpy as np
from gym.wrappers.time_limit import TimeLimit


class NormalizedEnv(TimeLimit):
    def __init__(self, env):
        TimeLimit.__init__(self, env)

        self.observation_scale = np.ones_like(self.observation_space.high)
        self.observation_shift = np.zeros_like(self.observation_space.high)

        self.action_scale = (self.action_space.high - self.action_space.low) / 2
        self.action_shift = (self.action_space.high + self.action_space.low) / 2

        self.rewards_scale = 1e-1
        self.rewards_shift = 0.

        # update observation and action space
        self.observation_space = gym.spaces.Box(self.normalize_observation(self.observation_space.low),
                                                self.normalize_observation(self.observation_space.high))
        self.action_space = gym.spaces.Box(-np.ones_like(self.action_space.high),
                                            np.ones_like(self.action_space.high))

    def normalize_observation(self, observation):
        return (observation - self.observation_shift) / self.observation_scale

    def normalize_action(self, action):
        return (self.action_scale * action) + self.action_scale

    def normalize_reward(self, rewards):
        return (self.rewards_scale * rewards) + self.rewards_shift

    def step(self, action):
        normalized_action = np.clip(self.normalized_action(action), self.action_space.high, self.action_space.low)

        obs, reward, term, info = TimeLimit.step(self, normalized_action)

        normalized_observation = self.normalized_observation(obs)

        return normalized_observation, reward, term, info
