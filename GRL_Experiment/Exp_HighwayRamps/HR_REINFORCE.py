"""Contains an experiment class for running simulations."""
import datetime
import logging

from GRL_Experiment.Exp_HighwayRamps.registry import make_create_env


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

    def run(self, num_HVs, num_AVs, training, testing, Graph):

        import torch
        from GRL_Library.agent.Discrete import REINFORCE_agent
        from GRL_Utils.Train_and_Test_REINFORCE import Training_GRLModels, Testing_GRLModels

        # Initialize GRL model
        N = num_HVs + num_AVs
        F = 2 + self.env.net_params.additional_params['highway_lanes'] \
            + self.env.n_unique_intentions
        A = 3
        assert isinstance(Graph, bool)
        if Graph:
            from GRL_Net.Model_Discrete.REINFORCE import Graph_Model
            GRL_Net = Graph_Model(N, F, A)
        else:
            from GRL_Net.Model_Discrete.REINFORCE import NonGraph_Model
            GRL_Net = NonGraph_Model(N, F, A)

        # Initialize optimizer
        optimizer = torch.optim.Adam(GRL_Net.parameters(), eps=0.001)  # 需要定义学习率
        # Initialize discount factor
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
