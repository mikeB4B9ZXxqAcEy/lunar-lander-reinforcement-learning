import random
from collections import deque, namedtuple


class ReplayBuffer:
    _Experience = namedtuple('Experience', field_names=('state', 'action', 'reward', 'next_state', 'done'))

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self._Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = tuple(e.state for e in experiences if e is not None)
        actions = tuple(e.action for e in experiences if e is not None)
        rewards = tuple(e.reward for e in experiences if e is not None)
        next_states = tuple(e.next_state for e in experiences if e is not None)
        done = tuple(e.done for e in experiences if e is not None)
        return states, actions, rewards, next_states, done

    def __len__(self):
        return len(self.memory)
