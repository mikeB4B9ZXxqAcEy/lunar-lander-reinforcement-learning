import random
import numpy as np
from K import K


class Agent:

    def __init__(self, env, model_A, model_B, memory):
        self.env = env
        self.epsilon = K.EPSILON_INIT
        self.memory = memory
        self.model_A = model_A
        self.model_B = model_B

    def pick_action(self, state, epsilon: float):
        state = np.array([state])
        if np.random.rand() < epsilon:
            # -1 because it's zero based
            action = random.randint(0, K.ACTION_SIZE - 1)
        else:
            q_values = self.model_A.predict(state)
            action = np.argmax(q_values)
        return action

    def train(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Flatten tuple of states to 1d array of states
        states = np.array(states)
        next_states = np.array(next_states)

        state_qs = self.model_A.predict(states)
        next_state_qs = self.model_B.predict(next_states)

        # START -- Implementation of Wang et al. 2015
        inverse_actions = np.logical_not(actions)
        next_state_intrinsic_value = np.amax(next_state_qs, axis=1).reshape(K.BATCH_SIZE, 1)
        dones = np.logical_not(dones).reshape(K.BATCH_SIZE, 1)
        next_state_intrinsic_value = K.DISCOUNT_FACTOR * next_state_intrinsic_value * dones
        rewards = np.array(rewards).reshape(K.BATCH_SIZE, 1)
        next_state_value = next_state_intrinsic_value + rewards

        state_qs = (actions * next_state_value) + (state_qs * inverse_actions)
        # END -- Implementation of Wang et al.

        self.model_A.fit(states, state_qs, batch_size=K.BATCH_SIZE, epochs=1, verbose=0)

    def run_episode(self):
        done = False
        score = 0
        state = self.env.reset()
        while not done:
            self.epsilon *= K.EPSILON_DECAY
            if self.epsilon < K.EPSILON_MIN:
                self.epsilon = K.EPSILON_MIN

            action = self.pick_action(state, self.epsilon)

            next_state, reward, done, info = self.env.step(action)
            score += reward
            self.memory.add(state, self._action_bit_of(action), reward, next_state, done)

            if len(self.memory) > K.BATCH_SIZE:
                self.train()

            state = next_state

        return score

    def _action_bit_of(self, action_idx: int):

        if action_idx < 0 or action_idx >= K.ACTION_SIZE:
            # It's zero based!
            raise ValueError('Invalid action index provided.')
        arr = [0] * K.ACTION_SIZE
        arr[action_idx] = 1
        return np.array(arr)
