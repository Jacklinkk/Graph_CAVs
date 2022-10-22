"""
    This function is used to define the DoubleNAF-agent
"""


import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import copy
import collections
import GRL_Library.agent.Continuous.NAF_agent as NAF

# CUDA configuration
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autograd.Variable(*args, **kwargs)


class DoubleNAF(NAF.NAF):
    """
        Define the DoubleNAF class, inheriting all the features of the NAF class
    """

    def compute_loss(self, data_batch):
        """
           <loss calculation function>
           Used to calculate the loss of the predicted and target values, as a basis for the subsequent backpropagation derivation

           Parameters:
           --------
           data_batch: The data sampled from the experience pool for training
        """
        loss = []

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

            # Calculate target network objective values based on Double operation
            action_evaluation, _, _ = self.model(next_state)
            action_evaluation = action_evaluation['action']
            _, _, value = self.target_model(next_state, action_evaluation)
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
            # Based on the Double operation, calculate the evaluation action of
            # the current network in the n_steps state S+1
            action_evaluation, _, _ = self.model(next_state)
            action_evaluation = action_evaluation['action']
            # Calculate the value of n-step max_Q
            _, _, value = self.target_model(next_state, action_evaluation)
            # Target value
            q_target = R + (self.gamma ** n_steps) * value * (1 - done)

            # ------TD_error of n-step------ #
            TD_error_sample = torch.abs(q_target - q_predict)
            TD_error_sample = torch.mean(TD_error_sample)
            # Count the TD_error of the current sample in the total TD_error
            TD_error.append(TD_error_sample)

            # ------loss------ #
            loss_sample = F.smooth_l1_loss(q_predict, q_target)

            loss.append(loss_sample)

        # Further processing of the TD_error can be done by deciding
        # whether to return this item as required
        TD_error = torch.stack(TD_error)

        # Combine loses of different samples in sample into tensor
        loss = torch.stack(loss)

        return loss
