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
        from GRL_Library.agent.Continuous import REINFORCE_agent
        from GRL_Utils.Train_and_Test_REINFORCE import Training_GRLModels, Testing_GRLModels

        # Initialize GRL model
        N = num_HVs + num_AVs
        F = 2
        A = 1
        assert isinstance(Graph, bool)
        if Graph:
            from GRL_Net.Model_Continuous.REINFORCE import Graph_Model
            GRL_Net = Graph_Model(N, F, A, action_min, action_max)
        else:
            from GRL_Net.Model_Continuous.REINFORCE import NonGraph_Model
            GRL_Net = NonGraph_Model(N, F, A, action_min, action_max)

        # Initialize optimizer
        optimizer = torch.optim.Adam(GRL_Net.parameters(), lr=1e-4)
        # Discount factor
        gamma = 0.9

        # Initialize GRL agent
        GRL_REINFORCE = REINFORCE_agent.REINFORCE(
            GRL_Net,  # model
            optimizer,  # optimizer
            gamma,  # discount factor
            model_name="DQN_model"  # model name
        )

        # Training
        n_episodes = 150
        max_episode_len = 2500
        save_dir = '../GRL_TrainedModels/REINFORCE/NOG5'
        debug_training = False
        if training:
            Training_GRLModels(GRL_Net, GRL_REINFORCE, self.env, n_episodes, max_episode_len, save_dir, debug_training)

        # Testing
        test_episodes = 10
        load_dir = '../GRL_TrainedModels/REINFORCE/NOG5'
        debug_testing = False
        if testing:
            Testing_GRLModels(GRL_Net, GRL_REINFORCE, self.env, test_episodes, load_dir, debug_testing)