"""
    This function is used to define the DDPG agent
"""

import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import collections
import copy
from GRL_Library.common.prioritized_replay_buffer import PrioritizedReplayBuffer

# CUDA configuration
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autograd.Variable(*args, **kwargs)


class DDPG(object):
    """
        Define the DDPG class (Deep Deterministic Policy Gradient)

        Parameters:
        --------
        actor_model: the neural network model used by the actor
        actor_optimizer: actor's optimizer
        critic_model: the neural network model used by critic
        critic_optimizer: optimizer for critic
        explore_noise: explore noise
        replay_buffer: experience replay pool
        batch_size: the length of the batch storage
        update_interval: current network update interval
        target_update_interval: target network update interval
        soft_update_tau: soft update parameter for the target network
        n_steps: Time Difference update step length (integer, 1 for single-step update, rest for Multi-step learning)
        gamma: discount factor
    """

    def __init__(self,
                 actor_model,
                 actor_optimizer,
                 critic_model,
                 critic_optimizer,
                 explore_noise,
                 replay_buffer,
                 batch_size,
                 update_interval,
                 target_update_interval,
                 soft_update_tau,
                 n_steps,
                 gamma,
                 model_name):

        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer
        self.explore_noise = explore_noise
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.soft_update_tau = soft_update_tau
        self.n_steps = n_steps
        self.gamma = gamma
        self.model_name = model_name

        # Target network
        self.actor_model_target = copy.deepcopy(actor_model)
        self.critic_model_target = copy.deepcopy(critic_model)

        # GPU configuration
        if USE_CUDA:
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.time_counter = 0

        self.loss_record = collections.deque(maxlen=100)

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
           <Action selection function>
           Generates the agent's action based on environmental observations

           Parameters:
           --------
           observation: observation of the environment where the smartbody is located
        """
        # Generate action
        action = self.actor_model(observation)
        noise = torch.as_tensor(self.explore_noise(), dtype=torch.float32).to(self.device)
        action = action + noise

        return action

    def test_action(self, observation):
        """
           <Test action selection function>
           Generate agent's actions based on environmental observations for the test process, and directly select the highest scoring action

           Parameters:
           --------
           observation: observation of the environment in which the intelligence is located
        """
        # Generate action
        action = self.actor_model(observation)

        return action

    def loss_process(self, loss, weight):
        """
           <Loss post-processing function>
           Different algorithms require different dimensions of loss data,
           so this function is written for uniform processing.

           Parameters:
           --------
           loss: the loss calculated by sample[1, self.batch_size]
           weight: weight factor
        """
        # Calculation of loss based on weight
        weight = torch.as_tensor(weight, dtype=torch.float32).to(self.device)
        loss = torch.mean(loss * weight.detach())

        return loss

    def learn_onestep(self, info_batch, data_batch):
        """
           <loss calculation function>
           Used to calculate the loss of the predicted and target values,
           as a basis for the subsequent backpropagation derivation.

           Parameter description:
           --------
           info_batch: the index and weight information of the sampled samples
           data_batch: the data sampled from the experience pool for training
        """
        # Initialize the loss matrix
        actor_loss = []
        critic_loss = []

        # Extract data from each sample in the order in which it is stored
        # ------loss of critic network------ #
        for elem in data_batch:
            state, action, reward, next_state, done = elem
            action = torch.as_tensor(action, dtype=torch.float32).to(self.device)

            # target value
            action_target = self.actor_model_target(next_state)
            critic_value_next = self.critic_model_target(next_state, action_target)
            critic_value = self.critic_model(state, action)
            critic_value = critic_value.detach()
            critic_target = reward + self.gamma * critic_value_next * (1 - done)

            critic_loss_sample = F.smooth_l1_loss(critic_value, critic_target)
            critic_loss.append(critic_loss_sample)

        # critic network update
        critic_loss_e = torch.stack(critic_loss)
        critic_loss_total = self.loss_process(critic_loss_e, info_batch['weights'])
        self.critic_optimizer.zero_grad()
        critic_loss_total.backward(retain_graph=True)
        self.critic_optimizer.step()

        # ------loss of actor network------ #
        for elem in data_batch:
            state, action, reward, next_state, done = elem

            mu = self.actor_model(state)
            actor_loss_sample = -1 * self.critic_model(state, mu)
            actor_loss_s = actor_loss_sample.mean()
            actor_loss.append(actor_loss_s)

        # actor network update
        actor_loss_e = torch.stack(actor_loss)
        actor_loss_total = self.loss_process(actor_loss_e, info_batch['weights'])
        self.actor_optimizer.zero_grad()
        actor_loss_total.backward(retain_graph=True)
        self.actor_optimizer.step()

        # ------Updating PRE weights------ #
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            self.replay_buffer.update_priority(info_batch['indexes'], (critic_loss_e + actor_loss_e))

        # ------Record loss------ #
        self.loss_record.append(float((critic_loss_total + actor_loss_total).detach().cpu().numpy()))

    def learn_multisteps(self, info_batch, data_batch):
        """
           <Multi-step learning loss calculation function>
           Used to calculate the loss of the predicted and target values,
           as a basis for the subsequent backpropagation derivation.

           Parameters:
           --------
           info_batch: index and weight information of the sampled samples
           data_batch: the data sampled from the experience pool for training
        """
        # Initialize the loss matrix
        actor_loss = []
        critic_loss = []

        # Extract data from each sample in the order in which it is stored
        # ------ calculates the loss of the critic network ------ #
        for elem in data_batch:
            # Take the smaller of n_steps and elem lengths to prevent
            # n-step sequential sampling out of index
            n_steps = min(self.n_steps, len(elem))

            # Take the sample data required for the calculation of the current and target values
            state, action, reward, next_state, done = elem[0]
            action = torch.as_tensor(action, dtype=torch.float32).to(self.device)
            state_, action_, reward_, next_state_, done_ = elem[n_steps - 1]

            # ------ Calculate the reward ------ #
            # Reward value
            reward = [i[2] for i in elem]
            # Discount factor
            n_step_scaling = [self.gamma ** i for i in range(n_steps)]
            # Calculate the reward matrix by multiplying the reward value with
            # the corresponding factor of the discount factor
            R = np.multiply(reward, n_step_scaling)
            # Sum
            R = np.sum(R)

            # Target value
            action_target = self.actor_model_target(next_state_)
            critic_value_next = self.critic_model_target(next_state_, action_target)
            critic_value = self.critic_model(state, action)
            critic_value = critic_value.detach()
            critic_target = R + self.gamma * critic_value_next * (1 - done)

            critic_loss_sample = F.smooth_l1_loss(critic_value, critic_target)
            critic_loss.append(critic_loss_sample)

        # critic network update
        critic_loss_e = torch.stack(critic_loss)
        critic_loss_total = self.loss_process(critic_loss_e, info_batch['weights'])
        self.critic_optimizer.zero_grad()
        critic_loss_total.backward(retain_graph=True)
        self.critic_optimizer.step()

        # ------loss of actor network------ #
        for elem in data_batch:
            # Take the sample data needed to calculate the current and target values
            state, action, reward, next_state, done = elem[0]
            action = torch.as_tensor(action, dtype=torch.float32).to(self.device)

            mu = self.actor_model(state)
            actor_loss_sample = -1 * self.critic_model(state, mu)
            actor_loss_s = actor_loss_sample.mean()
            actor_loss.append(actor_loss_s)

        # actor network update
        actor_loss_e = torch.stack(actor_loss)
        actor_loss_total = self.loss_process(actor_loss_e, info_batch['weights'])
        self.actor_optimizer.zero_grad()
        actor_loss_total.backward(retain_graph=True)
        self.actor_optimizer.step()

        # ------Updating PRE weights------ #
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            self.replay_buffer.update_priority(info_batch['indexes'], (critic_loss_e + actor_loss_e))

        # ------Record los------ #
        self.loss_record.append(float((critic_loss_total + actor_loss_total).detach().cpu().numpy()))

        return critic_loss, actor_loss

    def synchronize_target(self):
        """
           <target network update function>
           soft_update_tau = 1 for hard update, soft update for the rest
        """
        # The correct soft update parameter must be defined
        assert 0.0 < self.soft_update_tau <= 1.0

        # Parameters update
        for target_param, source_param in zip(self.critic_model_target.parameters(),
                                              self.critic_model.parameters()):
            target_param.data.copy_((1 - self.soft_update_tau) *
                                    target_param.data + self.soft_update_tau * source_param.data)

        for target_param, source_param in zip(self.actor_model_target.parameters(),
                                              self.actor_model.parameters()):
            target_param.data.copy_((1 - self.soft_update_tau) *
                                    target_param.data + self.soft_update_tau * source_param.data)

    def learn(self):
        """
           <policy update function>
           Used to implement the agent's learning process
        """
        # ------whether to return------ #
        if (self.time_counter <= 2 * self.batch_size) or \
                (self.time_counter % self.update_interval != 0):
            self.time_counter += 1
            return

        # ------ calculates the loss ------ #
        # Experience pool sampling, samples include weights and indexes,
        # data_sample is the specific sampled data
        samples, data_sample = self.sample_memory()

        # loss matrix
        if self.n_steps == 1:  # single-step learning
            self.learn_onestep(samples, data_sample)
        else:  # multi-step learning
            self.learn_multisteps(samples, data_sample)

        # ------target network update------ #
        if self.time_counter % self.target_update_interval == 0:
            self.synchronize_target()

        self.time_counter += 1

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
