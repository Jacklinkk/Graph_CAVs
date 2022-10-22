"""
    This function is used to define the DoubleDQN-agent
"""

import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import copy
import collections
import GRL_Library.agent.Discrete.DQN_agent as DQN

# CUDA configuration
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autograd.Variable(*args, **kwargs)


# Create the DoubleDQN class by inheritance
class DoubleDQN(DQN.DQN):
    """
       Define the DoubleDQN class, inheriting all the features of the DQN class
   """

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

        for elem in data_batch:
            state, action, reward, next_state, done = elem
            action = torch.as_tensor(action, dtype=torch.long, device=self.device)

            # Predicted value
            q_predict = self.model(state)
            q_predict = q_predict.gather(1, action.unsqueeze(1)).squeeze(1)

            # Save q_predict
            q_predict_save = q_predict.detach().cpu().numpy().reshape(len(q_predict), 1)
            data_useful = np.any(q_predict_save, axis=1)
            self.q_record.append(q_predict_save / (data_useful.sum() + 1))

            # Calculate the action value of the current network in state S+1
            q_evaluation = self.model(next_state)
            action_evaluation = torch.argmax(q_evaluation, dim=1)

            # ------Target value------ #
            # Calculate the q value of the target network in state S+1
            q_next = self.target_model(next_state)
            # Selecting actions based on the assessed action
            q_next = q_next.gather(1, action_evaluation.unsqueeze(1)).squeeze(1)
            # Calculating target values
            q_target = reward + self.gamma * q_next * (1 - done)

            loss_sample = F.smooth_l1_loss(q_predict, q_target)

            loss.append(loss_sample)

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
            # ------q_predict------ #
            # The first sample
            state, action, reward, next_state, done = elem[0]
            action = torch.as_tensor(action, dtype=torch.long, device=self.device)
            # predicted value
            q_predict = self.model(state)
            q_predict = q_predict.gather(1, action.unsqueeze(1)).squeeze(1)
            # Save q_predict
            q_predict_save = q_predict.detach().cpu().numpy().reshape(len(q_predict), 1)
            data_useful = np.any(q_predict_save, axis=1)
            self.q_record.append(q_predict_save / (data_useful.sum() + 1))

            # ------Reward calculation------ #
            reward = [i[2] for i in elem]
            # Discount factor
            n_step_scaling = [self.gamma ** i for i in range(n_steps)]
            # Calculate the reward matrix by multiplying the reward value with
            # the corresponding factor of the discount factor
            R = np.multiply(reward, n_step_scaling)

            R = np.sum(R)

            # ------q_target------ #
            state, action, reward, next_state, done = elem[n_steps - 1]
            # Calculate the q-value of the current network in the n_steps state S+1, and the maximum action
            q_evaluation = self.model(next_state)
            action_evaluation = torch.argmax(q_evaluation, dim=1)
            # Calculate the q value of the target network at n_steps state S+1
            q_next = self.target_model(next_state)
            # Selecting actions based on the assessed action
            q_next = q_next.gather(1, action_evaluation.unsqueeze(1)).squeeze(1)
            # Target value
            q_target = R + (self.gamma ** n_steps) * q_next * (1 - done)

            # ------Calculate n-step TD_error------ #
            TD_error_sample = torch.abs(q_target - q_predict)
            TD_error_sample = torch.mean(TD_error_sample)
            # Count the TD_error of the current sample in the total TD_error
            TD_error.append(TD_error_sample)

            # ------loss------ #
            loss_sample = F.smooth_l1_loss(q_predict, q_target)
            loss.append(loss_sample)

        TD_error = torch.stack(TD_error)

        loss = torch.stack(loss)

        return loss
