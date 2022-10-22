"""
    This function is used to define the DistributionalDQN-agent
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


# Creating the DistributionalDQN class by inheritance
class DistributionalDQN(DQN.DQN):
    """
        Defines the DistributionalDQN class, inheriting all the features of the DQN class

        Additional parameters:
        --------
        V_min: minimum value of the distributed value interval
        V_max: maximum value of the distributed value interval
        n_atoms: number of distributed samples
    """

    def __init__(self, model, optimizer, explorer, replay_buffer, gamma, batch_size, warmup_step,
                 update_interval, target_update_interval, target_update_method,
                 soft_update_tau, n_steps, V_min, V_max, n_atoms, model_name):
        super().__init__(model, optimizer, explorer, replay_buffer,
                         gamma, batch_size, warmup_step, update_interval,
                         target_update_interval, target_update_method,
                         soft_update_tau, n_steps, model_name)
        self.V_min = V_min
        self.V_max = V_max
        self.n_atoms = n_atoms

        # Calculating the conditional distribution support
        self.support = torch.linspace(self.V_min, self.V_max, self.n_atoms).to(self.device)

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

        TD_error = []

        # z_delta
        delta_z = float(self.V_max - self.V_min) / (self.n_atoms - 1)

        # Number of agents
        num_agents = self.model.num_agents

        # Index of agents
        index_dist = [i for i in range(num_agents)]

        for elem in data_batch:
            state, action, reward, next_state, done = elem

            next_action = self.target_model(next_state).argmax(1)
            next_dist = self.target_model.dist(next_state)
            # Get the specific value of the distribution based on the action
            next_dist = next_dist[index_dist, next_action, :]

            # Calculation of distribution-related parameters
            t_z = reward + self.gamma * self.support * (1 - done)
            t_z = t_z.clamp(min=self.V_min, max=self.V_max)
            b = (t_z - self.V_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (torch.linspace(0, (num_agents - 1) * self.n_atoms, num_agents).
                      long().unsqueeze(1).expand(num_agents, self.n_atoms).to(self.device))

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_ \
                (0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_ \
                (0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

            # Predicted value
            dist = self.model.dist(state)

            # Calculating KL dispersion as loss
            log_p = torch.log(dist[index_dist, action, :])
            loss_sample = -(proj_dist * log_p).sum(1).mean()

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

        # z_delta
        delta_z = float(self.V_max - self.V_min) / (self.n_atoms - 1)

        # Number of agents
        num_agents = self.model.num_agents

        # Index of agents
        index_dist = [i for i in range(num_agents)]

        for elem in data_batch:
            # Take the smaller of n_steps and elem lengths to prevent n-step sequential sampling out of index
            n_steps = min(self.n_steps, len(elem))

            # ------q_target and distribution------ #
            # The n-th sample
            state, action, reward, next_state, done = elem[n_steps - 1]
            # Calculation of movements and distribution
            next_action = self.target_model(next_state).argmax(1)
            next_dist = self.target_model.dist(next_state)
            # Get the specific value of the distribution based on the action
            next_dist = next_dist[index_dist, next_action, :]

            # ------Reward calculation------ #
            # Reward value
            reward = [i[2] for i in elem]
            # Discount factor
            n_step_scaling = [self.gamma ** i for i in range(n_steps)]
            # Calculate the reward matrix by multiplying the reward value with
            # the corresponding factor of the discount factor
            R = np.multiply(reward, n_step_scaling)
            # Sum
            R = np.sum(R)

            # ------Calculation of distribution-related parameters------ #
            t_z = R + (self.gamma ** self.n_steps) * self.support * (1 - done)
            t_z = t_z.clamp(min=self.V_min, max=self.V_max)
            b = (t_z - self.V_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (torch.linspace(0, (num_agents - 1) * self.n_atoms, num_agents).
                      long().unsqueeze(1).expand(num_agents, self.n_atoms).to(self.device))

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_ \
                (0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_ \
                (0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

            # ------q_predict distribution------ #
            state, action, reward, next_state, done = elem[0]
            dist = self.model.dist(state)

            # ------loss------ #
            # Calculating KL dispersion as loss
            log_p = torch.log(dist[index_dist, action, :])
            loss_sample = -(proj_dist * log_p).sum(1).mean()

            loss.append(loss_sample)

        loss = torch.stack(loss)

        return loss
