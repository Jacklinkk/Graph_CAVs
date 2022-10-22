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

        import torch.nn
        from GRL_Library.common import replay_buffer, explorer_continuous
        from GRL_Library.agent.Continuous import DoubleNAF_agent
        from GRL_Utils.Train_and_Test_NAF import Training_GRLModels, Testing_GRLModels

        # Initialize GRL model
        N = num_HVs + num_AVs
        F = 2
        A = 1
        assert isinstance(Graph, bool)
        if Graph:
            from GRL_Net.Model_Continuous.Q_Net import Graph_Model
            GRL_Net = Graph_Model(N, F, A, action_min, action_max)
        else:
            from GRL_Net.Model_Continuous.Q_Net import NonGraph_Model
            GRL_Net = NonGraph_Model(N, F, A, action_min, action_max)

        # Initialize optimizer
        optimizer = torch.optim.Adam(GRL_Net.parameters(), lr=1e-2)
        # Replay_buffer
        replay_buffer = replay_buffer.ReplayBuffer(size=10 ** 6)
        # Discount factor
        gamma = 0.9
        # Exploration
        explorer = explorer_continuous.LinearDecayEpsilonGreedy(start_epsilon=0.5, end_epsilon=0.01, decay_step=5000)

        # Initialize GRL agent
        warmup = 5000  # warmup
        GRL_NAF = DoubleNAF_agent.DoubleNAF(
            GRL_Net,  # model
            optimizer,  # optimizer
            explorer,  # exploration
            replay_buffer,  # replay buffer
            gamma,  # discount factor
            batch_size=32,  # batch_size
            warmup_step=warmup,  # warmup
            update_interval=100,  # model update interval
            target_update_interval=5000,  # target model update interval
            target_update_method='soft',  # update method
            soft_update_tau=0.1,  # soft update factor
            n_steps=1,  # multi-steps
            action_min=action_min,  # Action space Lower boundary
            action_max=action_max,  # Action space higher boundary
            model_name="DQN_model"  # model name
        )

        # Training
        n_episodes = 150
        max_episode_len = 2500
        save_dir = '../GRL_TrainedModels/DoubleNAF/DQN5'
        debug_training = False
        if training:
            Training_GRLModels(GRL_Net, GRL_NAF, self.env, n_episodes, max_episode_len, save_dir, warmup, debug_training)

        # Testing
        test_episodes = 10
        load_dir = 'Test_Models/DQN/DQN_3'
        debug_testing = False
        if testing:
            Testing_GRLModels(GRL_Net, GRL_NAF, self.env, test_episodes, load_dir, debug_testing)