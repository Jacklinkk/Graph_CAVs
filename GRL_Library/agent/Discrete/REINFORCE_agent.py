"""
    This function is used to define the REINFORCE_agent
"""

import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import collections

# CUDA configuration
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autograd.Variable(*args, **kwargs)


class REINFORCE(object):
    """
        Defining the REINFORCE class

        Parameters:
        --------
        model: the neural network model used in the agent
        optimizer: the optimizer to train the model
        gamma: discount factor
        model_name: the name of the model (to be saved and read)
    """

    def __init__(self,
                 model,
                 optimizer,
                 gamma,
                 model_name):

        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.model_name = model_name

        # GPU configuration
        if USE_CUDA:
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        # Set the current emulation step counter
        self.time_counter = 0

        # Set up training data recording matrix
        self.loss_record = collections.deque(maxlen=100)

        # Record data
        self.reward_memory = []
        self.action_memory = []

    def choose_action(self, observation):
        """
           <Action selection function>
           Generate agent's action based on environment observation

           Parameter:
           --------
           observation: the environment observation of the smart body
        """
        # Action generation
        probabilities = F.softmax(self.model(observation), dim=-1)
        action_probabilities = torch.distributions.Categorical(probabilities)
        action = action_probabilities.sample()
        log_probabilities = action_probabilities.log_prob(action)
        self.action_memory.append(log_probabilities)

        return action

    def store_rewards(self, reward):
        """
           <reward storage function>
           Store the rewards during agent interaction

           Parameters:
           --------
           reward: the reward for the interaction between the agent and the environment
        """
        self.reward_memory.append(reward)

    def learn(self):
        """
           <policy update function>
           Used to implement the agent's learning process
        """
        # ------Optimizer gradient initialization------ #
        self.optimizer.zero_grad()

        # ------Reward calculation------#
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        G = torch.tensor(G, dtype=torch.float).to(self.device)

        # ------Loss calculation------#
        loss = []
        for g, logprob in zip(G, self.action_memory):
            loss.append(torch.mean(-logprob * g))
        loss = torch.stack(loss).sum()
        # loss = torch.abs(loss)
        self.loss_record.append(float(loss.detach().cpu().numpy()))

        # ------Backward------ #
        loss.backward()
        self.optimizer.step()

        # ------Data logging reset------ #
        self.action_memory = []
        self.reward_memory = []

    def get_statistics(self):
        """
           <training data acquisition function>
           Used to get the relevant data during training
        """
        loss_statistics = np.mean(self.loss_record) if self.loss_record else np.nan
        return [loss_statistics]

    def save_model(self, save_path):
        """
           <Model saving function>
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
