import numpy as np

class OUExploration:
  # Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py

  def __init__(self, env, sigma=0.3, mu=0, theta=0.15):
    self.action_size = env.action_space.shape[0]
    self.mu = mu
    self.theta = theta
    self.sigma = sigma

    self.state = np.ones(self.action_size) * self.mu
    self.reset()

  def add_noise(self, action):
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
    self.state = x + dx

    return action + self.state

  def reset(self):
    self.state = np.ones(self.action_size) * self.mu
