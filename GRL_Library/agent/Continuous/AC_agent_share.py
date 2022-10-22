"""
    This function is used to define the Actor-Critics agent
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


class AC(object):
    """
        Defining the AC class (Actor-Critic)

        Parameters:
        --------
        model: the neural network model used in the agent
        optimizer: optimizer for training the model
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

        self.time_counter = 0

        self.loss_record = collections.deque(maxlen=100)

        # Log value of the probability of recording the action
        self.log_probs = []

    def choose_action(self, observation):
        """
           <Action selection function>
           Generates the agent's action based on environmental observations

           Parameters:
           --------
           observation: observation of the environment where the smartbody is located
        """
        # Generate action
        action, log_probs, _ = self.model(observation)
        self.log_probs = log_probs

        return action

    def learn(self, state, reward, next_state, done):
        """
           <policy update function>
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
        _, _, next_critic_value = self.model(next_state)
        _, _, critic_value = self.model(state)

        # ------TD_error------#
        # Reward calculation
        reward = torch.as_tensor(reward, dtype=torch.float32).to(self.device)
        # TD_target
        y_t = reward + self.gamma * next_critic_value * (1 - done)

        # ------loss calculation------#
        # loss of actor network
        self.log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32).to(self.device)
        self.log_probs = torch.reshape(self.log_probs, (len(y_t), 1))
        actor_loss = -1 * torch.mul(self.log_probs, critic_value)
        actor_loss = torch.mean(actor_loss)
        # loss of critic network
        critic_loss = F.smooth_l1_loss(y_t, critic_value)

        # ------Backward update------ #
        (actor_loss + critic_loss).backward()
        print("actor_loss:", actor_loss)
        print("critic_loss:", critic_loss)
        self.optimizer.step()

        # Record loss
        self.loss_record.append(float((actor_loss + critic_loss).detach().cpu().numpy()))

    def get_statistics(self):
        """
           <training data fetch function>
           Used to fetch relevant data from the training process
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
           <model reading function>
           Used to read the trained model
        """
        load_path = load_path + "/" + self.model_name + ".pt"
        self.model = torch.load(load_path)
