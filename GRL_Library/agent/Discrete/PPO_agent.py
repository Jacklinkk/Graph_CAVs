"""
    This function is used to define the PPO agent
"""

import torch as T
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import collections

# CUDA configuration
USE_CUDA = T.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
#     if USE_CUDA else autograd.Variable(*args, **kwargs)
# autograd.set_detect_anomaly(True)


class PPOMemory(object):
    """
        Define PPOMemory class as replay buffer

        Parameter description:
        --------
        state: current state
    """

    def __init__(self, batch_size):
        self.states = []
        self.probs = []  # Action probability
        self.vals = []  # Value
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batch(self):
        """
           <batch sampling function>
           Used to implement empirical sampling of PPOMemory
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return self.states, \
               self.actions, \
               self.probs, \
               self.vals, \
               np.asarray(self.rewards), \
               np.asarray(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        """
           <data storage function>
           Used to store the data of the agent interaction process

           Parameters:
           --------
           state: current state
           action: current action
           probs: action probability
           vals: value of the action
           reward: the reward for performing the action
           done: whether the current round is completed or not
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """
           <data clear function>
           Used to clear the interaction data already stored and free memory
        """
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class PPO(object):
    """
        Define the PPO class (Proximal Policy Optimization)

        Parameter description:
        --------
        actor_model: actor network
        actor_optimizer: actor optimizer
        critic_model: value network
        critic_optimizer: critic optimizer
        gamma: discount factor
        GAE_lambda: GAE (generalized advantage estimator) coefficient
        policy_clip: policy clipping coefficient
        batch_size: sample size
        n_epochs: number of updates per batch
        update_interval: model update step interval
        model_name: model name (used to save and read)
    """

    def __init__(self,
                 actor_model,
                 critic_model,
                 gamma,
                 GAE_lambda,
                 policy_clip,
                 batch_size,
                 n_epochs,
                 update_interval,
                 model_name):

        self.actor_model = actor_model
        self.critic_model = critic_model
        self.gamma = gamma
        self.GAE_lambda = GAE_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.update_interval = update_interval
        self.model_name = model_name

        # GPU configuration
        if USE_CUDA:
            GPU_num = T.cuda.current_device()
            self.device = T.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        # Replay buffer
        self.memory = PPOMemory(self.batch_size)

        # Record data
        self.loss_record = collections.deque(maxlen=100)

    def store_transition(self, state, action, probs, vals, reward, done):
        """
           <Experience storage function>
           Used to store the experience data during the agent learning process

           Parameters:
           --------
           state: the state of the current moment
           action: current moment action
           probs: probability of current action
           vals: the value of the current action
           reward: the reward obtained after performing the current action
           done: whether to terminate or not
        """
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        """
          <Action selection function>
          Generate agent's action based on environment observation

          Parameters:
          --------
          observation: the environment observation of the smart body
       """
        dist = self.actor_model(observation)
        value = self.critic_model(observation)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action))
        action = T.squeeze(action)
        value = T.squeeze(value)

        return action, probs, value

    def learn(self):
        """
           <policy update function>
           Used to implement the agent's learning process
        """
        # ------Training according to the specific value of n_epochs------ #
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batch()

            values = vals_arr

            # ------Training for each epochs------ #
            # advantage
            advantage = T.zeros(len(reward_arr), len(action_arr[1])).to(self.device)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] *
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.GAE_lambda
                advantage[t, :] = a_t
            # advantage = torch.stack(advantage)

            # values
            values = T.stack(values)

            # Training for the collected samples
            # Note: do loss.backward() in a loop, to avoid gradient compounding
            for batch in batches:
                # Initialize the loss matrix
                actor_loss_matrix = []
                critic_loss_matrix = []

                # Train for each index in the batch
                # Calculate actor_loss and update the actor network
                for i in batch:
                    old_probs = old_prob_arr[i].detach()
                    actions = action_arr[i].detach()

                    dist = self.actor_model(state_arr[i])

                    new_probs = dist.log_prob(actions)
                    prob_ratio = new_probs.exp() / old_probs.exp()
                    weighted_probs = advantage[i].detach() * prob_ratio  # PPO1
                    # ------PPO2------#
                    weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[i].detach()
                    # ----------------#
                    actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                    actor_loss_matrix.append(actor_loss)

                actor_loss_matrix = T.stack(actor_loss_matrix)
                actor_loss_mean = T.mean(actor_loss_matrix)

                self.actor_model.optimizer.zero_grad()
                actor_loss_mean.backward()
                self.actor_model.optimizer.step()

                # Calculate critic_loss and update the critic network
                for i in batch:
                    critic_value = self.critic_model(state_arr[i])
                    critic_value = T.squeeze(critic_value)

                    returns = advantage[i] + values[i]
                    returns = returns.detach()
                    critic_loss = F.smooth_l1_loss(returns, critic_value)

                    critic_loss_matrix.append(critic_loss)

                critic_loss_matrix = T.stack(critic_loss_matrix)
                critic_loss_mean = 0.5 * T.mean(critic_loss_matrix)

                # Backward
                self.critic_model.optimizer.zero_grad()
                critic_loss_mean.backward()
                self.critic_model.optimizer.step()

                # Save loss
                self.loss_record.append(float((actor_loss_mean + critic_loss_mean).detach().cpu().numpy()))

        # Buffer clear
        self.memory.clear_memory()

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
        save_path_actor = save_path + "/" + self.model_name + "_actor" + ".pt"
        save_path_critic = save_path + "/" + self.model_name + "_critic" + ".pt"
        T.save(self.actor_model, save_path_actor)
        T.save(self.critic_model, save_path_critic)

    def load_model(self, load_path):
        """
           <Model reading function>
           Used to read the trained model
        """
        load_path_actor = load_path + "/" + self.model_name + "_actor" + ".pt"
        load_path_critic = load_path + "/" + self.model_name + "_critic" + ".pt"
        self.actor_model = T.load(load_path_actor)
        self.critic_model = T.load(load_path_critic)
