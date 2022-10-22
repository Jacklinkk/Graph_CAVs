"""
    This function is used to define the A2C agent
"""

import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import collections
from GRL_Library.agent.Discrete import AC_agent

# CUDA configuration
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autograd.Variable(*args, **kwargs)


class A2C(AC_agent.AC):
    """
        Define class A2C (Advanced Actor-Critic), inheriting all properties of AC.
    """

    def learn(self, state, reward, next_state, done):
        """
           <policy update function
           Used to implement the agent's learning process

           Parameters:
           --------
           state: current state
           reward: the reward after the action is performed
           next_state: the state after the action has been performed
           done: whether the current turn is complete or not
        """
        # ------ optimizer gradient initialization ------ #
        self.optimizer.zero_grad()

        # ------critic network value------ #
        _, next_critic_value = self.model(next_state)
        _, critic_value = self.model(state)

        # ------TD_error------#
        # Reward calculation
        reward = torch.as_tensor(reward, dtype=torch.float32).to(self.device)
        # TD_target calculation
        y_t = reward + self.gamma * next_critic_value * (1 - done)

        # ------loss calculation------#
        # loss of actor network
        self.log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32).to(self.device)
        self.log_probs = torch.reshape(self.log_probs, (len(y_t), 1))
        # Here (y_t - critic_value) is introduced as the calculation of advantage,
        # which is where A2C differs from AC
        actor_loss = -1 * torch.mul(self.log_probs, (y_t - critic_value))
        actor_loss = torch.mean(actor_loss)
        # loss of critic network
        critic_loss = F.smooth_l1_loss(y_t, critic_value)
        # Record loss
        self.loss_record.append(float((actor_loss + critic_loss).detach().cpu().numpy()))

        # ------Backward update------ #
        (actor_loss + critic_loss).backward()
        self.optimizer.step()
