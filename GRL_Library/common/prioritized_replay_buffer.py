"""
    This function is used to define the prioritized_replay_buffer in the DRL.
"""
import random
import numpy as np


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha, beta, beta_step, epsilon):
        """
            <Constructor>
            Defines the priority_replay_buffer class

            Parameters:
            ------
            capacity: the maximum capacity of the replay_buffer, when the capacity is exceeded, the new
                      data will replace the old data when the capacity is exceeded
            alpha: sampling probability error index
            beta: the significance sampling index
            beta_step: incremental value of beta per sample
            (beta should not exceed 1, and the update rate should be controlled)
            epsilon: a very small value to prevent zero priority.
        """
        # capacity is set to the nth power of 2 to facilitate code writing and debugging
        assert capacity & (capacity - 1) == 0

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_step = beta_step
        self.epsilon = epsilon

        self.max_priority = 1.

        # Summation of a binary tree and finding the minimum value within a range
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        self.next_index = 0
        self.size = 0
        self.buffer = []

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
        # # Combine the above data and store it in [data]
        data = (state, action, reward, next_state, done)

        # Index calculation
        idx = self.next_index

        # Store data
        if idx >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[idx] = data

        # Index update
        self.next_index = (idx + 1) % self.capacity
        # Length update
        self.size = min(self.capacity, self.size + 1)

        # Calculate Pi*a, the new sample gets the maximum priority
        priority_alpha = self.max_priority ** self.alpha

        # priority update
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        """
            <minimum priority function>
            Sets the minimum priority in a binary line segment tree

            Parameters:
            ------
            idx: index of the current transition
            priority_alpha: priority to take the value
        """
        # Leaf node
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Traversing along the parent node to update the tree up to the root of the tree
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx],
                                         self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        """
            <summing priority function>
            Set the summation of priority in a binary line segment tree

            Parameters:
            ------
            idx: index of the current transition
            priority: the priority value to take
        """
        # Leaf node
        idx += self.capacity
        self.priority_sum[idx] = priority

        # Traversing along the parent node to update the tree up to the root of the tree
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + \
                                     self.priority_sum[2 * idx + 1]

    def _sum(self):
        """
            <priority sum function>
            Sums the priority in a binary line tree as follows:
            ∑k(Pk)^alpha
        """
        return self.priority_sum[1]

    def _min(self):
        """
            <minimum priority function>
            Searches for minimum priority in a binary line segment tree, specifically:
            min_k (Pk)^alpha
        """
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """
            <maximum priority search function>
            Search for maximum priority in a bifurcated line segment tree.
        """
        # Search from the root node
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:  # Left node
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]  # Right node
                idx = 2 * idx + 1
        return idx - self.capacity

    def sample(self, batch_size, n_steps=1):
        """
            <Data sampling function
            Sampling data in replay_buffer

            Parameter description.
            ------
            batch_size: the amount of data to be sampled from the replay_buffer
            n_steps: the number of multi-steps learning steps, which affects
            the number of simultaneous samples
        """
        # beta value
        beta = self.beta

        # samples initialization
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }

        # Index
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx

        # min_i Pi
        probability_min = self._min() / self._sum()

        # max_i wi
        max_weight = (probability_min * self.size) ** (-beta)

        # sample weights
        for i in range(batch_size):
            idx = samples['indexes'][i]
            probability = self.priority_sum[idx + self.capacity] / self._sum()  # 计算Pi
            weight = (probability * self.size) ** (-beta)  # 计算权重
            samples['weights'][i] = weight / max_weight  # 为样本赋予权重

        # beta update
        self.beta = min((beta + self.beta_step), 1)

        # sample acquiring
        sample_data = []
        if n_steps == 1:  # single-step learning
            for idx in samples['indexes']:
                # Here the index of data should correspond to the index of priority
                data_i = self.buffer[idx]
                sample_data.append(data_i)
        else:  # multi-step learning
            for idx in samples['indexes']:
                data_i = self.buffer[idx: idx + n_steps]
                sample_data.append(data_i)

        # Here, in addition to the sampled sample data, the index and corresponding
        # weights are also returned for the priority update and loss calculation
        return samples, sample_data

    def update_priority(self, indexes, priorities):
        """
            <priority update function>
            Update priority

            Parameters:
            ------
            indexes: the indexes generated by sample
            priorities: priority specific values
        """
        # Avoid zero priority
        priorities = priorities.detach().cpu().numpy()
        priorities = priorities + self.epsilon

        # priority update
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)
