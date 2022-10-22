"""Contains an experiment class for running simulations."""
from flow.core.util import emission_to_csv
from GRL_Experiment.Exp_FigureEight.registry import make_create_env
import datetime
import logging
import time
import os
import numpy as np
import json


class Experiment:

    def __init__(self, flow_params, custom_callables=None):
        """Instantiate the Experiment class.

        Parameters
        ----------
        flow_params : dict
            flow-specific parameters
        custom_callables : dict < str, lambda >
            strings and lambda functions corresponding to some information we
            want to extract from the environment. The lambda will be called at
            each step to extract information from the env and it will be stored
            in a dict keyed by the str.
        """
        self.custom_callables = custom_callables or {}

        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params)

        # Create the environment.
        self.env = create_env()

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")

    def run(self, num_HVs, num_AVs, training, testing, action_min, action_max, Graph):

        import torch
        import torch.nn
        from GRL_Library.common import replay_buffer
        from GRL_Library.agent.Continuous import TD3_agent
        from GRL_Utils.Train_and_Test_DDPG import Training_GRLModels, Testing_GRLModels

        # Initialize GRL model
        N = num_HVs + num_AVs
        F = 2
        A = 1
        assert isinstance(Graph, bool)
        if Graph:
            from GRL_Net.Model_Continuous.DDPG import OUActionNoise, Graph_Actor_Model, Graph_Critic_Model
            actor = Graph_Actor_Model(N, F, A, action_min, action_max)
            critic_1 = Graph_Critic_Model(N, F, A, action_min, action_max)
            critic_2 = Graph_Critic_Model(N, F, A, action_min, action_max)
        else:
            from GRL_Net.Model_Continuous.DDPG import OUActionNoise, NonGraph_Actor_Model, NonGraph_Critic_Model
            actor = NonGraph_Actor_Model(N, F, A, action_min, action_max)
            critic_1 = NonGraph_Critic_Model(N, F, A, action_min, action_max)
            critic_2 = NonGraph_Critic_Model(N, F, A, action_min, action_max)

        # Initialize optimizer
        lr = 1e-4
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)  # 需要定义学习率
        critic_optimizer_1 = torch.optim.Adam(critic_1.parameters(), lr=lr)  # 需要定义学习率
        critic_optimizer_2 = torch.optim.Adam(critic_2.parameters(), lr=lr)  # 需要定义学习率
        # Noisy
        explore_noise = 0.1
        # Replay_buffer
        replay_buffer = replay_buffer.ReplayBuffer(size=10 ** 6)
        # Discount factor
        gamma = 0.9

        warmup = 500
        # Initialize GRL agent
        GRL_TD3 = TD3_agent.TD3(
            actor,
            actor_optimizer,
            critic_1,
            critic_optimizer_1,
            critic_2,
            critic_optimizer_2,
            explore_noise,  # noisy
            warmup,  # warmup
            replay_buffer,  # replay buffer
            batch_size=32,  # batch_size
            update_interval=100,  # model update interval (< actor model)
            update_interval_actor=500,  # actor model update interval
            target_update_interval=5000,  # target model update interval
            soft_update_tau=0.01,  # soft update factor
            n_steps=1,  # multi-steps
            gamma=gamma,  # discount factor
            model_name="DQN_model"  # model name
        )

        # Training
        n_episodes = 10
        max_episode_len = 2500
        save_dir = '../GRL_TrainedModels/TD3/NOG5'
        debug_training = False
        if training:
            Training_GRLModels(actor, GRL_TD3, self.env, n_episodes, max_episode_len, save_dir, debug_training)

        # Testing
        test_episodes = 10
        load_dir = '../GRL_TrainedModels/TD3/NOG5'
        debug_testing = False
        if testing:
            Testing_GRLModels(actor, GRL_TD3, self.env, test_episodes, load_dir, debug_testing)
