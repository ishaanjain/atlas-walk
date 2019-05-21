import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()

    def add(self, state, action, reward, done, next_state):
        """ Add a new transition to the replay buffer """
        transition = (state, action, reward, done, next_state)

        if (self.size() < self.buffer_size):
            self.buffer.append(transition)
        else:
            self.buffer.popleft()
            self.buffer.append(transition)

    def get_batch(self, batch_size):
        """ Get a random sample of transitions from the replay buffer """
        batch = list(map(list, zip(*random.sample(self.buffer, batch_size))))

        return batch[0], batch[1], batch[2], batch[3], batch[4]

    def size(self):
        """ Get the current number of transitions in the replay buffer """
        return len(self.buffer)

    def clear(self):
        """ Empty the replay buffer """
        self.buffer = deque()
