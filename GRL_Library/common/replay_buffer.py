"""
    This function is used to define the replay_buffer in the DRL
"""

import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        """
            <Constructor>
            Define replay buffer

            Parameters：
            ------
            size: The maximum capacity of the replay_buffer, when the capacity is exceeded,
            the new data will replace the old data.
        """
        self.size = size  # Define the maximum size of the replay_buffer
        self.buffer = []  # Define the replay_buffer storage list (storage core)
        self.index = 0  # Define the replay_buffer index
        self.length = 0  # Defines the current length of the replay_buffer (run step)

    def add(self, state, action, reward, next_state, done):
        """
            <Data storage function>
            Store data in the replay buffer

            Parameters:
            ------
            state: current state of the moment
            action: the action at the current moment
            reward: the reward received for performing the current action
            next_state: the next state after the current action
            done: whether to terminate
        """
        # Combine the above data and store it in [data]
        data = (state, action, reward, next_state, done)

        # Store data
        if self.index >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data

        # Index update
        self.index = (self.index + 1) % self.size

        # Length update
        self.length = min(self.length + 1, self.size)

    def sample(self, batch_size, n_steps=1):
        """
            <Data sampling function>
            Sampling data in replay_buffer

            Parameters:
            ------
            batch_size: the amount of data to be sampled from the replay_buffer
            n_steps: the number of multi-steps learning steps, which affects the
            number of simultaneous samples.
        """
        # samples initialization, uniform with PER form to record weights and indexes
        # common replay_buffer, indexes randomly generated, weights all-1 matrix
        samples = {'weights': np.ones(shape=batch_size, dtype=np.float32),
                   'indexes': np.random.choice(self.length - n_steps + 1, batch_size, replace=False)}

        # Data sampling
        sample_data = []
        if n_steps == 1:  # single-step learning
            for i in samples['indexes']:
                data_i = self.buffer[i]
                sample_data.append(data_i)
        else:  # multi-step learning
            for i in samples['indexes']:
                data_i = self.buffer[i: i + n_steps]
                sample_data.append(data_i)

        return samples, sample_data

