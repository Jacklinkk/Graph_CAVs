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
        from GRL_Library.agent.Discrete import PPO_agent
        from GRL_Utils.Train_and_Test_PPO import Training_GRLModels, Testing_GRLModels

        # Initialize GRL model
        N = num_HVs + num_AVs
        F = 2 + self.env.net_params.additional_params['highway_lanes'] \
            + self.env.n_unique_intentions
        A = 3
        lr = 0.001
        assert isinstance(Graph, bool)
        if Graph:
            from GRL_Net.Model_Discrete.PPO import Graph_Actor_Model, Graph_Critic_Model
            GRL_actor = Graph_Actor_Model(N, F, A, lr)
            GRL_critic = Graph_Critic_Model(N, F, A, lr)
        else:
            from GRL_Net.Model_Discrete.PPO import NonGraph_Actor_Model, NonGraph_Critic_Model
            GRL_actor = NonGraph_Actor_Model(N, F, A, lr)
            GRL_critic = NonGraph_Critic_Model(N, F, A, lr)

        # Discount factor
        gamma = 0.9
        # GAE factor
        GAE_lambda = 0.95
        # Policy clip factor
        policy_clip = 0.2

        # Initialize GRL agent
        GRL_PPO = PPO_agent.PPO(
            GRL_actor,  # actor model
            GRL_critic,  # critic model
            gamma,  # discount factor
            GAE_lambda,  # GAE factor
            policy_clip,  # policy clip factor
            batch_size=32,  # batch_size < update_interval
            n_epochs=5,  # update times for one batch
            update_interval=100,  # update interval
            model_name="DQN_model"  # model name
        )

        # Training
        n_episodes = 150
        max_episode_len = 2500
        save_dir = '../GRL_TrainedModels/PPO/DQN5'
        debug_training = False
        if training:
            Training_GRLModels(GRL_PPO, self.env, n_episodes, max_episode_len, save_dir, debug_training)

        # Testing
        test_episodes = 10
        load_dir = '../GRL_TrainedModels/PPO/DQN5'
        debug_testing = False
        if testing:
            Testing_GRLModels(GRL_PPO, self.env, test_episodes, load_dir, debug_testing)
