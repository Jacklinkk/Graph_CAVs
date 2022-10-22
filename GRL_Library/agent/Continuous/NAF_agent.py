"""
    This function is used to define the NAF-agent
"""

import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import copy
import collections
from GRL_Library.common.prioritized_replay_buffer import PrioritizedReplayBuffer

# CUDA configuration
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autograd.Variable(*args, **kwargs)


class NAF(object):
    """
        Defining the NAF class

        Parameters:
        --------
        model: the neural network model used in the agent
        optimizer: optimizer for training the model
        explorer: exploration and action selection strategy
        replay_buffer: experience replay pool
        gamma: discount factor
        batch_size: batch storage length
        warmup_step: random exploration step
        update_interval: current network update interval
        target_update_interval: target network update interval
        target_update_method: target network update method (hard or soft)
        soft_update_tau: target network soft update parameter
        n_steps: Time Difference update step length (integer, 1 for single-step update, the rest for multi-step learning)
        action_min: lower bound on action space
        action_max: upper bound on the action space
        model_name: model name (to be saved and read)
    """

    def __init__(self,
                 model,
                 optimizer,
                 explorer,
                 replay_buffer,
                 gamma,
                 batch_size,
                 warmup_step,
                 update_interval,
                 target_update_interval,
                 target_update_method,
                 soft_update_tau,
                 n_steps,
                 action_min,
                 action_max,
                 model_name):

        self.model = model
        self.optimizer = optimizer
        self.explorer = explorer
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_step = warmup_step
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.n_steps = n_steps
        self.action_min = action_min
        self.action_max = action_max
        self.model_name = model_name

        # GPU configuration
        if USE_CUDA:
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.target_model = copy.deepcopy(model)

        self.time_counter = 0

        self.loss_record = collections.deque(maxlen=100)
        self.q_record = collections.deque(maxlen=100)

    def store_transition(self, state, action, reward, next_state, done):
        """
           <experience storage function>
           Used to store experience data from the agent learning process

           Parameters:
           --------
           state: current state at the moment
           action: the action at the current moment
           reward: the reward received for performing the current action
           next_state: the next state after the current action
           done: whether to terminate or not
        """
        # Call the function that holds the data in the replay_buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

    def sample_memory(self):
        """
           <Experience sampling function>
           Used to sample empirical data from the agent learning process
        """
        # Call the sampling function in replay_buffer
        data_sample = self.replay_buffer.sample(self.batch_size, self.n_steps)
        return data_sample

    def choose_action(self, observation):
        """
           <training action selection function>
           Generates the agent's actions based on environmental observations for the training process

           Parameters:
           --------
           observation: observation of the environment in which the intelligence is located
        """
        # Generate actions
        action, _, _ = self.model(observation)
        action = self.explorer.generate_action(action)
        return action

    def test_action(self, observation):
        """
           <Test action selection function>
           Generate agent's actions based on environmental observations for the test process, and directly select the highest scoring action

           Parameters:
           --------
           observation: observation of the environment in which the intelligence is located
        """
        # Generate actions
        action, _, _ = self.model(observation)
        action = action['action']
        return action

    def compute_loss(self, data_batch):
        """
           <loss calculation function>
           Used to calculate the loss of the predicted and target values, as a basis for the subsequent backpropagation derivation

           Parameters:
           --------
           data_batch: The data sampled from the experience pool for training
        """
        # Initialize the loss matrix
        loss = []
        # TD_error
        TD_error = []

        for elem in data_batch:
            state, action, reward, next_state, done = elem
            action = torch.as_tensor(action, dtype=torch.long).to(self.device)

            # Predicted value
            _, q_predict, _ = self.model(state, action)

            # Save q_predict
            q_predict_save = q_predict.detach().cpu().numpy().reshape(len(q_predict), 1)
            data_useful = np.any(q_predict_save, axis=1)
            self.q_record.append(q_predict_save / (data_useful.sum() + 1))

            # Target value
            _, _, value = self.target_model(next_state)
            q_target = reward + self.gamma * value * (1 - done)

            # TD_error
            TD_error_sample = torch.abs(q_target - q_predict)
            TD_error_sample = torch.mean(TD_error_sample)
            # Count the TD_error of the current sample in the total TD_error
            TD_error.append(TD_error_sample)

            loss_sample = F.smooth_l1_loss(q_predict, q_target)

            loss.append(loss_sample)

        # Further processing of TD_error
        TD_error = torch.stack(TD_error)

        # Combine the loses of different samples in a sample into a tensor
        loss = torch.stack(loss)

        return loss

    def compute_loss_multisteps(self, data_batch, n_steps):
        """
           <Multi-step learning loss calculation function>
           Used to calculate the loss of the predicted and target values, as a basis for the subsequent backpropagation derivation

           Parameters:
           --------
           data_batch: the data sampled from the experience pool for training
           n_steps: multi-step learning interval
        """
        loss = []

        TD_error = []

        for elem in data_batch:
            # Take the smaller of n_steps and elem lengths to
            # prevent n-step sequential sampling out of index
            n_steps = min(self.n_steps, len(elem))

            # ------q_predict value------ #
            # First sample
            state, action, reward, next_state, done = elem[0]
            action = torch.as_tensor(action, dtype=torch.long).to(self.device)
            # Predicted value
            _, q_predict, _ = self.model(state, action)
            # Save q_predict
            q_predict_save = q_predict.detach().cpu().numpy().reshape(len(q_predict), 1)
            data_useful = np.any(q_predict_save, axis=1)
            self.q_record.append(q_predict_save / (data_useful.sum() + 1))

            # ------Reward calculation------ #
            reward = [i[2] for i in elem]
            # Discount factor
            n_step_scaling = [self.gamma ** i for i in range(n_steps)]
            # Calculate the reward matrix by multiplying the reward value with the
            # corresponding factor of the discount factor
            R = np.multiply(reward, n_step_scaling)
            # Sum
            R = np.sum(R)

            # ------q_target------ #
            state, action, reward, next_state, done = elem[n_steps - 1]
            # Calculate the value of n-step max_Q
            _, _, value = self.target_model(next_state)
            # Target value
            q_target = R + (self.gamma ** n_steps) * value * (1 - done)

            # ------TD_error for n-step------ #
            TD_error_sample = torch.abs(q_target - q_predict)
            TD_error_sample = torch.mean(TD_error_sample)
            # Count the TD_error of the current sample in the total TD_error
            TD_error.append(TD_error_sample)

            # ------loss------ #
            loss_sample = F.smooth_l1_loss(q_predict, q_target)
            # Add the loss of the current sample to the total loss
            loss.append(loss_sample)

        # Further processing of the TD_error can be done by deciding
        # whether to return this item as required
        TD_error = torch.stack(TD_error)

        # Combine the loses of different samples in a sample into a tensor
        loss = torch.stack(loss)

        return loss

    def loss_process(self, loss, weight):
        """
           <Loss post-processing function>
           Different algorithms require different dimensions of loss data,
           so this function is written for uniform processing

           Parameters:
           --------
           loss: the loss calculated by sample[1, self.batch_size]
        """
        # Calculation of loss based on weights
        weight = torch.as_tensor(weight, dtype=torch.float32).to(self.device)
        loss = torch.mean(loss * weight)

        return loss

    def synchronize_target(self):
        """
           <target_network_sync_function>
           Used to synchronize the target network (target_network)
        """
        if self.target_update_method == "hard":
            self.hard_update()
        elif self.target_update_method == "soft":
            self.soft_update()
        else:
            raise ValueError("Unknown target update method")

    def hard_update(self):
        """
           <target_network_hard_update_function>
           Synchronize the target network (target_network) using the hard_update method
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update(self):
        """
           <target_network soft update function>
           Synchronize the target network (target_network) using the soft_update method
        """
        # The correct soft update parameters must be defined
        assert 0.0 < self.soft_update_tau < 1.0

        # Parameters update
        for target_param, source_param in zip(self.target_model.parameters(),
                                              self.model.parameters()):
            target_param.data.copy_((1 - self.soft_update_tau) *
                                    target_param.data + self.soft_update_tau * source_param.data)

    def learn(self):
        """
           <policy update function>
           Used to implement the agent's learning process
        """
        # ------Whether to return------ #
        if (self.time_counter <= self.warmup_step) or \
                (self.time_counter % self.update_interval != 0):
            self.time_counter += 1
            return

        # ------ calculates the loss ------ #
        # Experience pool sampling, samples include weights and indexes, data_sample is specific sampling data
        samples, data_sample = self.sample_memory()

        if self.n_steps == 1:  # single-step learning
            elementwise_loss = self.compute_loss(data_sample)
        else:  # multi-step learning
            elementwise_loss = self.compute_loss_multisteps(data_sample, self.n_steps)

        # In the case of PrioritizedReplayBuffer, the priority is updated
        # before the total loss is calculated
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            self.replay_buffer.update_priority(samples['indexes'], elementwise_loss)

        loss = self.loss_process(elementwise_loss, samples['weights'])

        self.loss_record.append(float(loss.detach().cpu().numpy()))

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # Whether to update target model
        if self.time_counter % self.target_update_interval == 0:
            self.synchronize_target()

        self.time_counter += 1

    def get_statistics(self):
        """
           <training data fetch function>
           Used to fetch relevant data from the training process
        """
        loss_statistics = np.mean(self.loss_record) if self.loss_record else np.nan
        q_statistics = np.mean(np.absolute(self.q_record)) if self.q_record else np.nan
        return [loss_statistics, q_statistics]

    def save_model(self, save_path):
        """
           <Model save function>
           Used to save the trained model
        """
        save_path = save_path + "/" + self.model_name + ".pt"
        torch.save(self.model, save_path)

    def load_model(self, load_path):
        """
           <Model reading function>
           Used to read the trained model
        """
        load_path = load_path + "/" + self.model_name + ".pt"
        self.model = torch.load(load_path)
