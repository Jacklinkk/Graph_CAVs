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
        actor_model: the neural network model used by the actor
        actor_optimizer: the actor's optimizer
        critic_model: the neural network model used by critic
        critic_optimizer: critic's optimizer
        gamma: discount factor
        model_name: the name of the model (to be saved and read)
    """

    def __init__(self,
                 actor_model,
                 actor_optimizer,
                 critic_model,
                 critic_optimizer,
                 gamma,
                 model_name):

        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer
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
        action, log_probs = self.actor_model(observation)
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
        # ------critic network value------ #
        next_critic_value = self.critic_model(next_state)
        critic_value = self.critic_model(state)

        # ------TD_error------#
        # Reward
        reward = torch.as_tensor(reward, dtype=torch.float32).to(self.device)
        # TD_target
        y_t = reward + self.gamma * next_critic_value * (1 - done)

        # ------loss calculation------#
        # loss of actor network
        self.log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32).to(self.device)
        self.log_probs = torch.reshape(self.log_probs, (len(y_t), 1))
        actor_loss = -1 * torch.mul(self.log_probs, critic_value)
        actor_loss = torch.mean(actor_loss)
        # actor network update
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        # loss of critic network
        critic_loss = F.smooth_l1_loss(y_t, critic_value)
        # critic network update
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

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
           <Model save function>
           Used to save the trained model
        """
        save_path_actor = save_path + "/" + self.model_name + "_actor" + ".pt"
        save_path_critic = save_path + "/" + self.model_name + "_critic" + ".pt"
        torch.save(self.actor_model, save_path_actor)
        torch.save(self.critic_model, save_path_critic)

    def load_model(self, load_path):
        """
           <model reading function>
           Used to read the trained model
        """
        load_path_actor = load_path + "/" + self.model_name + "_actor" + ".pt"
        load_path_critic = load_path + "/" + self.model_name + "_critic" + ".pt"
        self.actor_model = torch.load(load_path_actor)
        self.critic_model = torch.load(load_path_critic)
