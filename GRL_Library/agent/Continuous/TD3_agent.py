"""
    This function is used to define the TD3 agent
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


class TD3(object):
    """
        Define the TD3 class (Twin Delayed Deep Deterministic Policy Gradient)

        Parameters:
        --------
        actor_model: the neural network model used by the actor
        actor_optimizer: actor's optimizer
        critic_model_1: the neural network model used by critic_1
        critic_optimizer_1: optimizer for critic_1
        critic_model_2: Neural network model used by critic_2
        critic_optimizer_2: optimiser for critic_2
        explore_noise: explore_noise
        warmup: exploration step
        replay_buffer: experience replay pool
        batch_size: batch storage length
        update_interval: current network update interval
        update_interval_actor: actor network update interval
        target_update_interval: target network update interval
        soft_update_tau: target network soft update parameter
        n_steps: Time Difference update step length
        (integer, 1 for single-step update, the rest for Multi-step learning)
        gamma: discount factor
    """

    def __init__(self,
                 actor_model,
                 actor_optimizer,
                 critic_model_1,
                 critic_optimizer_1,
                 critic_model_2,
                 critic_optimizer_2,
                 explore_noise,
                 warmup,
                 replay_buffer,
                 batch_size,
                 update_interval,
                 update_interval_actor,
                 target_update_interval,
                 soft_update_tau,
                 n_steps,
                 gamma,
                 model_name):

        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.critic_model_1 = critic_model_1
        self.critic_optimizer_1 = critic_optimizer_1
        self.critic_model_2 = critic_model_2
        self.critic_optimizer_2 = critic_optimizer_2

        self.explore_noise = explore_noise
        self.warmup = warmup
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.update_interval_actor = update_interval_actor
        self.target_update_interval = target_update_interval
        self.soft_update_tau = soft_update_tau
        self.n_steps = n_steps
        self.gamma = gamma
        self.model_name = model_name

        # target network
        self.actor_model_target = copy.deepcopy(self.actor_model)
        self.critic_model_target_1 = copy.deepcopy(self.critic_model_1)
        self.critic_model_target_2 = copy.deepcopy(self.critic_model_2)

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
        if self.time_counter < self.warmup:
            action = np.random.normal(scale=self.explore_noise,
                                      size=(self.actor_model.num_agents,
                                            self.actor_model.num_outputs))
            action = torch.as_tensor(action, dtype=torch.float32).to(self.device)
        else:
            action = self.actor_model(observation)
            noise = torch.as_tensor(np.random.normal(scale=self.explore_noise)).to(self.device)
            action = action + noise

        action = torch.clamp(action, self.actor_model.action_min, self.actor_model.action_max)
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
        critic_loss_1 = []
        critic_loss_2 = []

        # Extract data from each sample in the order in which it is stored
        # ------loss of critic network------ #
        for elem in data_batch:
            state, action, reward, next_state, done = elem
            action = torch.as_tensor(action, dtype=torch.float32).to(self.device)

            # target value
            action_target = self.actor_model_target(next_state)
            action_target = action_target + \
                            torch.clamp(torch.as_tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
            action_target = torch.clamp(action_target,
                                        self.actor_model.action_min,
                                        self.actor_model.action_max)

            # target critic_value
            q1_next = self.critic_model_target_1(next_state, action_target)
            q2_next = self.critic_model_target_2(next_state, action_target)
            q1 = self.critic_model_1(state, action)
            q1 = q1.detach()
            q2 = self.critic_model_2(state, action)
            q2 = q2.detach()
            critic_value_next = torch.min(q1_next, q2_next)

            critic_target = reward + self.gamma * critic_value_next * (1 - done)

            # loss calculation
            q1_loss = F.smooth_l1_loss(critic_target, q1)
            q2_loss = F.smooth_l1_loss(critic_target, q2)
            critic_loss_1.append(q1_loss)
            critic_loss_2.append(q2_loss)

        # critic network update
        critic_loss_e_1 = torch.stack(critic_loss_1)
        critic_loss_e_2 = torch.stack(critic_loss_2)
        critic_loss_total_1 = self.loss_process(critic_loss_e_1, info_batch['weights'])
        critic_loss_total_2 = self.loss_process(critic_loss_e_2, info_batch['weights'])

        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        (critic_loss_total_1 + critic_loss_total_2).backward(retain_graph=True)
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        # Determining whether to update the actor network
        if self.time_counter % self.update_interval_actor != 0:
            return
        # ------loss of actor network------ #
        for elem in data_batch:
            state, action, reward, next_state, done = elem

            mu = self.actor_model(state)
            actor_loss_sample = -1 * self.critic_model_1(state, mu)
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
            self.replay_buffer.update_priority(info_batch['indexes'], (critic_loss_e_1
                                                                       + critic_loss_e_2
                                                                       + actor_loss_e))

        # ------Record loss------ #
        self.loss_record.append(float((critic_loss_total_1 +
                                       critic_loss_total_2 +
                                       actor_loss_total).detach().cpu().numpy()))

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
        critic_loss_1 = []
        critic_loss_2 = []

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
            action_target = action_target + \
                            torch.clamp(torch.as_tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
            action_target = torch.clamp(action_target,
                                        self.actor_model.action_min,
                                        self.actor_model.action_max)

            # Target critic_value
            q1_next = self.critic_model_target_1(next_state_, action_target)
            q2_next = self.critic_model_target_2(next_state_, action_target)
            q1 = self.critic_model_1(state, action)
            q1 = q1.detach()
            q2 = self.critic_model_2(state, action)
            q2 = q2.detach()
            critic_value_next = torch.min(q1_next, q2_next)

            critic_target = R + self.gamma * critic_value_next * (1 - done)

            # loss calculation
            q1_loss = F.smooth_l1_loss(critic_target, q1)
            q2_loss = F.smooth_l1_loss(critic_target, q2)
            critic_loss_1.append(q1_loss)
            critic_loss_2.append(q2_loss)

        # critic network update
        critic_loss_e_1 = torch.stack(critic_loss_1)
        critic_loss_e_2 = torch.stack(critic_loss_2)
        critic_loss_total_1 = self.loss_process(critic_loss_e_1, info_batch['weights'])
        critic_loss_total_2 = self.loss_process(critic_loss_e_2, info_batch['weights'])

        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        (critic_loss_total_1 + critic_loss_total_2).backward(retain_graph=True)
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        # ------loss of actor network------ #
        for elem in data_batch:
            # Take the sample data needed to calculate the current and target values
            state, action, reward, next_state, done = elem[0]
            action = torch.as_tensor(action, dtype=torch.float32).to(self.device)

            mu = self.actor_model(state)
            actor_loss_sample = -1 * self.critic_model_1(state, mu)
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
            self.replay_buffer.update_priority(info_batch['indexes'], (critic_loss_e_1
                                                                       + critic_loss_e_2
                                                                       + actor_loss_e))

        # ------Record loss------ #
        self.loss_record.append(float((critic_loss_total_1 +
                                       critic_loss_total_2 +
                                       actor_loss_total).detach().cpu().numpy()))

    def synchronize_target(self):
        """
           <target network update function>
           soft_update_tau = 1 for hard update, soft update for the rest
        """
        # The correct soft update parameter must be defined
        assert 0.0 < self.soft_update_tau <= 1.0

        # Parameters update
        for target_param, source_param in zip(self.critic_model_target_1.parameters(),
                                              self.critic_model_1.parameters()):
            target_param.data.copy_((1 - self.soft_update_tau) *
                                    target_param.data + self.soft_update_tau * source_param.data)

        for target_param, source_param in zip(self.critic_model_target_2.parameters(),
                                              self.critic_model_2.parameters()):
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
        if (self.time_counter <= self.warmup) or \
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
        save_path_critic_1 = save_path + "/" + self.model_name + "_critic_1" + ".pt"
        save_path_critic_2 = save_path + "/" + self.model_name + "_critic_2" + ".pt"
        torch.save(self.actor_model, save_path_actor)
        torch.save(self.critic_model_1, save_path_critic_1)
        torch.save(self.critic_model_2, save_path_critic_2)

    def load_model(self, load_path):
        """
           <model reading function>
           Used to read the trained model
        """
        load_path_actor = load_path + "/" + self.model_name + "_actor" + ".pt"
        load_path_critic_1 = load_path + "/" + self.model_name + "_critic_1" + ".pt"
        load_path_critic_2 = load_path + "/" + self.model_name + "_critic_2" + ".pt"
        self.actor_model = torch.load(load_path_actor)
        self.critic_model_1 = torch.load(load_path_critic_1)
        self.critic_model_2 = torch.load(load_path_critic_2)
